import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.init as init
from torch.autograd import Variable, grad
import numpy as np
import math

# helpers functions

def exists(x):
    return x is not None

def default(val, d):
    if exists(val): 
        return val
    return d() if callable(d) else d

def cast_tuple(t, length = 1):
    if isinstance(t, tuple):
        return t
    return ((t,) * length)

def divisible_by(numer, denom):
    return (numer % denom) == 0

def identity(t, *args, **kwargs):
    return t

def cycle(dl):
    while True:
        for data in dl:
            yield data


def num_to_groups(num, divisor):
    groups = num // divisor
    remainder = num % divisor
    arr = [divisor] * groups
    if remainder > 0:
        arr.append(remainder)
    return arr


# normalization functions

def Upsample2D(dim, dim_out = None, upsample_factor = (2,2)):
    return nn.Sequential(
        nn.Upsample(scale_factor = upsample_factor, mode = 'nearest'),
        nn.Conv2d(dim, default(dim_out, dim), 3, padding = 1)
    )

def Downsample2D(dim, dim_out = None):
    return nn.Sequential(
        nn.MaxPool2d(kernel_size=(2,2), stride=(2,2), padding=0),
        nn.Conv2d(dim, default(dim_out, dim), 1)
    )
   

# building block modules
class ConvBlock2D(nn.Module):  # input dimension is dim, output dimension is dim_out
    def __init__(self, dim, dim_out, groups = 8, dilation = None):
        super().__init__()
        if dilation == None:
            self.conv = nn.Conv2d(dim, dim_out, 3, padding = 1)
        else:
            self.conv = nn.Conv2d(dim, dim_out, 3, padding = dilation, dilation = dilation)
        self.norm = nn.GroupNorm(groups, dim_out)  
        self.act = nn.ReLU()

    def forward(self, x):
        x = self.conv(x)
        x = self.norm(x)
        x = self.act(x)
        return x
    

# model: UNet3D

class Unet2D(nn.Module):
    def __init__(
        self,
        init_dim = 16,
        channels = 1,
        dim_mults = (2,4,8,16),
        num_classes = 2,
    ):
        super().__init__()
    
        self.channels = channels
        input_channels = channels

        self.init_conv = nn.Conv2d(input_channels, init_dim, 3, padding = 1) # if want input and output to have same dimension, Kernel size to any odd number (e.g., 3, 5, 7, etc.). Padding to (kernel size - 1) / 2.

        dims = [init_dim, *map(lambda m: init_dim * m, dim_mults)]  # if initi_dim = 16, then [16, 32, 64, 128, 256]

        in_out = list(zip(dims[:-1], dims[1:])) 
        print('in out is : ', in_out)
        # [(16,32), (32,64), (64,128), (128,256)]. Each tuple in in_out represents a pair of input and output dimensions for different stages in a neural network 

        # layers
        self.downs = nn.ModuleList([])
        self.ups = nn.ModuleList([])
        num_resolutions = len(in_out) # 4

        for ind, (dim_in, dim_out) in enumerate(in_out):
            print(' in downsampling path, ind is: ', ind, ' dim_in is: ', dim_in, ' dim_out is: ', dim_out)

            # in each downsample stage, 
            # we have two conv blocks and then downsampling layer (downsample x and y by 2, then increase the feature number by 2)
            self.downs.append(nn.ModuleList([
                ConvBlock2D(dim_in, dim_in),
                ConvBlock2D(dim_in, dim_in),
                Downsample2D(dim_in, dim_out ) 
            ]))



        for ind, (dim_in, dim_out) in enumerate(reversed(in_out)):
            print(' in upsampling path, ind is: ', ind, ' dim_in is: ', dim_in, ' dim_out is: ', dim_out)
          
            # in each upsample stage,
            # we have a resnetblock and then upsampling layer (upsample x and y by 2, then decrease the feature number by 2)
            self.ups.append(nn.ModuleList([
                ConvBlock2D(dim_out, dim_out) if ind == 0 else ConvBlock2D(dim_out * 2, dim_out),
                ConvBlock2D(dim_out, dim_out),
                Upsample2D(dim_out, dim_in)  
            ]))

        self.out_dim = num_classes

        self.final_block = ConvBlock2D(init_dim * 2, init_dim)
        self.final_conv = nn.Conv2d(init_dim, self.out_dim, 1)  # output channel is initial channel number

    def forward(self, x):

        x = self.init_conv(x)
        # print('initial x shape is: ', x.shape)

        h = []
        for block1, block2, downsample in self.downs:
            x = block1(x)
            x = block2(x)
            h.append(x)
            x = downsample(x)
      
        
        for block1, block2, upsample in self.ups:
            x = block1(x)
            x = block2(x)
            x = upsample(x)
            x = torch.cat((x, h.pop()), dim = 1)

        x = self.final_block(x)
        final_image = self.final_conv(x)
        # print('final image shape is: ', final_image.shape)
      
        return final_image
