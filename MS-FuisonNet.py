import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from einops import rearrange

# Thanks for https://github.com/ljbuaa/VisualDecoding

class ConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size,with_bn=True,with_relu=True,stride=1,padding=0,bias=True):
        super().__init__()
        self.with_bn=with_bn
        self.with_relu=with_relu
        self.conv2d = nn.Conv2d(in_channels, out_channels, kernel_size,stride=stride,padding=padding,bias=bias)
        self.batchNorm=None
        self.relu=None
        if with_bn:
            self.batchNorm=nn.BatchNorm2d(out_channels)
        if with_relu:
            self.relu=nn.ELU()
    def forward(self, x):
        out=self.conv2d(x)
        if self.with_bn:
            out=self.batchNorm(out)
        if self.with_relu:
            out=self.relu(out)
        return out

class MultiHeadAttention(nn.Module):
    def __init__(self, emb_size, num_heads, dropout):
        super().__init__()
        self.emb_size = emb_size
        self.num_heads = num_heads
        self.keys = nn.Linear(emb_size, emb_size)
        self.queries = nn.Linear(emb_size, emb_size)
        self.values = nn.Linear(emb_size, emb_size)
        self.att_drop = nn.Dropout(dropout)
        self.projection = nn.Linear(emb_size, emb_size)

    def forward(self, x: Tensor, mask: Tensor = None) -> Tensor:
        queries = rearrange(self.queries(x), "b n (h d) -> b h n d", h=self.num_heads)
        keys = rearrange(self.keys(x), "b n (h d) -> b h n d", h=self.num_heads)
        values = rearrange(self.values(x), "b n (h d) -> b h n d", h=self.num_heads)
        energy = torch.einsum('bhqd, bhkd -> bhqk', queries, keys)
        if mask is not None:
            fill_value = torch.finfo(torch.float32).min
            energy.mask_fill(~mask, fill_value)

        scaling = self.emb_size ** (1 / 2)
        att = F.softmax(energy / scaling, dim=-1)
        att = self.att_drop(att)
        out = torch.einsum('bhal, bhlv -> bhav ', att, values)
        out = rearrange(out, "b h n d -> b n (h d)")
        out = self.projection(out)
        return out

class FeedForwardBlock(nn.Sequential):
    def __init__(self, emb_size, expansion, drop_p):
        super().__init__(
            nn.Linear(emb_size, expansion * emb_size),
            nn.GELU(),
            nn.Dropout(drop_p),
            nn.Linear(expansion * emb_size, emb_size),
        )

#Signal Channel Attention Weighting
class SCAW(nn.Module):
    def __init__(self, channel, reduction = 1):
        super().__init__()
        self.channel = channel
        self.fc = nn.Sequential(
            nn.Linear(channel, channel // reduction, bias=False),
            nn.ELU(inplace  = True),
            nn.Linear(channel // reduction, channel, bias=False),
            nn.Sigmoid()
        )
        self.conv1d1 = nn.Conv1d(in_channels=17,out_channels=10, stride=16,kernel_size=16)
        self.conv1d2 = nn.Conv1d(in_channels=7, out_channels=10, kernel_size=10, stride=10)
    def forward(self, x):
        if len(x.shape)==3:
            b, c, t = x.size()
            xstd=((x-x.mean(-1).view(b,c,1))**2)
            xstd = F.normalize(xstd.sum(-1),dim=-1)
            attn = self.fc(xstd).view(b, c, 1)
        else:
            b, c, s, t = x.size()
            xstd=((x-x.mean(-1).view(b,c,s,1))**2)
            xstd = F.normalize(xstd.sum(-1),dim=-1)
            attn = self.fc(xstd).view(b, c, s, 1)

        out = x * attn.expand_as(x)

        if self.channel==17:
            out = self.conv1d1(out)
        elif self.channel==7:
            out = self.conv1d2(out)
        return out


class STSTransformerBlock(nn.Module):
    def __init__(self, emb_size1,emb_size2,num_heads=5,drop_p=0.5,forward_expansion=4,forward_drop_p=0.5):
        super().__init__()
        self.emb_size = emb_size1
        self.att_drop1 = nn.Dropout(drop_p)
        self.projection1 = nn.Linear(emb_size1, emb_size1)
        self.projection2 = nn.Linear(emb_size1, emb_size1)
        self.drop1=nn.Dropout(drop_p)
        self.drop2=nn.Dropout(drop_p)

        self.layerNorm1=nn.LayerNorm(emb_size1)
        self.layerNorm2=nn.LayerNorm(emb_size2)

        self.queries1 = nn.Linear(emb_size1, emb_size1)
        self.values1 = nn.Linear(emb_size1, emb_size1)
        self.keys2 = nn.Linear(emb_size2, emb_size2)
        self.values2 = nn.Linear(emb_size2, emb_size2)

        self.layerNorm3=nn.LayerNorm(emb_size1+emb_size2)
        self.mha=MultiHeadAttention(emb_size1+emb_size2, num_heads, 0.5)
        self.drop3=nn.Dropout(drop_p)

        self.ffb=nn.Sequential(
            nn.LayerNorm(emb_size1+emb_size2),
            FeedForwardBlock(
                emb_size1+emb_size2, expansion=forward_expansion, drop_p=forward_drop_p),
            nn.Dropout(drop_p)
        )

    def forward(self, x1, x2):
        x1=rearrange(x1, 'b e (h) (w) -> b (h w) e ')
        x2=rearrange(x2, 'b e (h) (w) -> b (h w) e ')
        res1=x1
        res2=x2

        x1 = self.layerNorm1(x1)
        x2 = self.layerNorm2(x2)
        queries1 = self.queries1(x1)
        values1 = self.values1(x1)
        keys2 = self.keys2(x2)
        values2 = self.values2(x2)

        energy = torch.einsum('bqd, bkd -> bqk', keys2, queries1)
        scaling = self.emb_size ** (1 / 2)
        att = F.softmax(energy / scaling, dim=-1)
        att = self.att_drop1(att)

        out1 = torch.einsum('bal, blv -> bav ', att, values1)
        out1 = self.projection1(out1)
        x1 = self.drop1(out1)
        x1+=res1

        out2 = torch.einsum('bal, blv -> bav ', att, values2)
        out2 = self.projection2(out2)
        x2 = self.drop2(out2)
        x2+=res2

        x=torch.cat((x1,x2),dim=-1)
        res = x
        x=self.layerNorm3(x)
        x=self.mha(x)
        x=self.drop3(x)
        x += res

        res = x
        x = self.ffb(x)
        x += res
        x = rearrange(x, 'b t e -> b e 1 t')
        return x

class MS_FusionNet(nn.Module):
    def __init__(self,classNum=2,channele=17,channelo=7):
        super().__init__()

        self.scawe = SCAW(channele)
        self.scawo = SCAW(channelo)

        self.eeg1 = ConvBlock(1, 40, (10, 1))
        self.eeg2 = nn.Sequential(
            ConvBlock(40, 30, (1, 13)),
            ConvBlock(30, 10, (1, 11)),
        )
        self.eegAvgPool = nn.Sequential(
            nn.AdaptiveAvgPool2d((1, 8)),
            nn.Dropout2d(0.5)
        )

        self.eog1 = ConvBlock(1, 40, (10, 1))
        self.eog2 = nn.Sequential(
            ConvBlock(40, 30, (1, 13)),
            ConvBlock(30, 10, (1, 11)),
        )
        self.eogAvgPool = nn.Sequential(
            nn.AdaptiveAvgPool2d((1, 8)),
            nn.Dropout2d(0.5)
        )

        self.fuseConv1=nn.Sequential(
            ConvBlock(80,60,(1,13)),
            ConvBlock(60,20,(1,11)),
            nn.AdaptiveAvgPool2d((1,8)),
            nn.Dropout2d(0.5)
        )

        self.fuseAvgPool=nn.Sequential(
            nn.AdaptiveAvgPool2d((1,8)),
            nn.Dropout2d(0.5)
        )
        self.fusinTB=STSTransformerBlock(40,40)

        self.feaLen=(40)*8
        self.classify = nn.Sequential(
            nn.Linear(self.feaLen, classNum),
        )
    def forward(self, eeg, eog):

        weeg = self.scawe(eeg)
        weog = self.scawo(eog)

        eeg1_out = self.eeg1(weeg.unsqueeze(1))
        eeg2_out1 = self.eeg2[0](eeg1_out)
        eeg2_out2 = self.eeg2[1](eeg2_out1)
        eeg_out = self.eegAvgPool(eeg2_out2)
        eeg_out = eeg_out.squeeze()

        eog1_out = self.eog1(weog.unsqueeze(1))
        # print(eog1_out.shape)
        eog2_out1 = self.eog2[0](eog1_out)
        eog2_out2 = self.eog2[1](eog2_out1)
        eog_out = self.eegAvgPool(eog2_out2)
        eog_out = eog_out.squeeze()

        fuse_out1 = self.fusinTB(eeg1_out,eog1_out)
        fuse_out1 = self.fuseConv1(fuse_out1).squeeze()
        out = torch.cat((eeg_out,eog_out,fuse_out1),dim=1)
        out = self.classify(out.reshape(-1,self.feaLen))

        return out#,fuse_out1

if __name__ == '__main__':
    x=torch.randn(32,17,1600) # EEG data with 17 channel * 1600 timepoint
    y=torch.randn(32,7,1000) #  EOG data with 7 channel * 1000 timepoint
    model=MS_FusionNet()
    pre_y=model(x,y)
    print("pre_y.shape:",pre_y.shape)