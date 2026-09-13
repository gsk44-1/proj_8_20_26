import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms.functional as TF
import numpy as np


#the double well portion of the gradient, nonlinear pointwise portion
def cubic_iter(x, s=15, num_iter=3): 
    #s is alpha in the double well paper = 2tau lambda / epsilon
    xx = x  # we start with v_0 = u_n
    for _ in range(num_iter):
        out = (x - 2.*s.xx**3 + 3.*s*xx**2)/(1.+s)
        xx = out
    return out

def laplace_kern(in_chan, out_chan):
    weight=torch.zeros(out_chan, in_chan, 3, 3, requires_grad=False)
    weight[:, :, 1, 0] = 1
    weight[:, :, 1, 2] = 1
    weight[:, :, 0, 1] = 1
    weight[:, :, 2, 1] = 1
    weight[:, :, 1, 1] = -4
    return weight

class DoubleConv(nn.Module):
    def __init__(self, in_chan, out_chan):
        super(DoubleConv, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_chan, out_chan, 3, 1, 1, bias=False),
            nn.BatchNorm2d(out_chan),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_chan, out_chan, 3, 1, 1, bias=False),
            nn.BatchNorm2d(out_chan),
            nn.ReLU(inplace=True)
        )
    def forward(self, x):
        return self.conv(x)

class UNET(nn.Module):
    def __init__(
            self, in_chan=3, out_chan=1, features=[64, 128, 256],
    ):
        super(UNET, self).__init__()
        self.ups=nn.ModuleList()
        self.downs=nn.ModuleList()
        self.pool=nn.MaxPool2d(kernel_size=2, stride=2)

        for feature in features:
            self.downs.append(DoubleConv(in_chan, feature))
            in_chan=feature


        self.ups.append(
            nn.ConvTranspose2d(features[-1]*2, features[-1], kernel_size=2, stride=2)
        )
        self.ups.append(
            DoubleConv(features[-1]*2, feature[-1])
        ) #we have twice the number of expected channels because of concatenation

        for j in reversed(range(len(features)-1)):
            self.ups.append(
                nn.ConvTranspose2d(features[j+1], features[j], kernel_size=2, stride=2)
            )
            self.ups.append(DoubleConv(features[j]*2, features[j]))

        self.bottleneck = DoubleConv(features[-1], features[-1]*2)
        self.final_conv=nn.Conv2d(features[0], out_chan, kernel_size=1)

        self.BNfinal=nn.BatchNorm2d(out_chan)
        self.sig=nn.Sigmoid()

    def forward(self,x):
        
        skip_connections=[]
            
        for down in self.downs:
            x=down(x)
            skip_connections.append(x)
            x=self.pool(x)
                
        x=self.bottleneck(x)
            
        skip_connections=skip_connections[::-1]
            
        for idx in range(0,len(self.ups),2):
                x=self.ups[idx](x)
                skip_connection=skip_connections[idx//2]
                if x.shape!=skip_connection.shape:
                    x=TF.resize(x,size=skip_connection.shape[2:])
                
                
                concat_skip=torch.cat((skip_connection,x),dim=1)
                x=self.ups[idx+1](concat_skip)
                
        return self.BNfinal(self.final_conv(x))


class ConvBlock(nn.Module):
    def __init__(self):
        super(ConvBlock,self).__init__()
        self.conv1=nn.Conv2d(1,1,3,1,1,padding_mode='circular', bias=False)
        self.BN1=nn.BatchNorm2d(1)
        self.diff=laplace_kern(1,1)
        self.convDiff=nn.Conv2d(1, 1, kernel_size=3, stride=1, padding=1, padding_mode='circular', bias=False)
        self.convDiff.weight=torch.nn.Parameter(self.diff,requires_grad=False)
        self.sig = nn.Sigmoid()
        self.tanh = nn.Tanh()
        self.ReLU = nn.ReLU(inplace=True)
        
    def forward(self,x,Ff):
        out=self.conv1(x)
        out=self.BN1(out)
        
        out=x+0.5*out+0.5*self.convDiff(x)+0.5*Ff #time step is 0.5?

        out=self.sig(out)
        out=cubic_iter(out)
        return out


class DNI(nn.Module):
    def __init__(self,features=[64,128,256],num_blocks=1):
        super(DNI,self).__init__()
        self.layer1=nn.Conv2d(3, 1, kernel_size=3, stride=1, padding=1, padding_mode='circular',bias=False)
        self.final=nn.Conv2d(1, 1, kernel_size=3, stride=1, padding=1, padding_mode='circular')
        self.BN1=nn.BatchNorm2d(1)
        self.sig = nn.Sigmoid()
        self.tanh = nn.Tanh()
        self.blocks=nn.ModuleList()
        self.num_blocks=num_blocks
        self.features=features
        self.F=UNET(features=self.features) #this will be replaced with a function of both x and f
        
        for idx in range(self.num_blocks):
            self.blocks.append(ConvBlock())
            
            
    def forward(self,x):
        out=self.layer1(x)
        out=self.BN1(out)
        out=self.sig(out)
        out=cubic_iter(out)
        Ff=self.F(x)
        for idx in range(self.num_blocks):
            out=self.blocks[idx](out,Ff)
        
        out=self.final(out)

        out=self.sig(out)


        return out