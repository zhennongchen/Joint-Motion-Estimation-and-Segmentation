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

def Upsample3D(dim, dim_out = None, upsample_factor = (2,2,1)):
    return nn.Sequential(
        nn.Upsample(scale_factor = upsample_factor, mode = 'nearest'),
        nn.Conv3d(dim, default(dim_out, dim), 3, padding = 1)
    )

def Downsample3D(dim, dim_out = None):
    return nn.Sequential(
        nn.MaxPool3d(kernel_size=(2,2, 1), stride=(2,2, 1), padding=0),
        nn.Conv3d(dim, default(dim_out, dim), 1)
    )
   

# building block modules
class ConvBlock3D(nn.Module):  # input dimension is dim, output dimension is dim_out
    def __init__(self, dim, dim_out, groups = 8, dilation = None):
        super().__init__()
        if dilation == None:
            self.conv = nn.Conv3d(dim, dim_out, 3, padding = 1)
        else:
            self.conv = nn.Conv3d(dim, dim_out, 3, padding = dilation, dilation = dilation)
        self.norm = nn.GroupNorm(groups, dim_out)  
        self.act = nn.ReLU()

    def forward(self, x):
        x = self.conv(x)
        x = self.norm(x)
        x = self.act(x)
        return x
    

# model: UNet3D

class Unet3D(nn.Module):
    def __init__(
        self,
        init_dim = 16,
        channels = 1,

        out_dim = None,
        dim_mults = (2,4,8,16),
        self_condition = False,   # use the prediction from the previous iteration as the condition of next iteration
        attn_dim_head = 32,
        attn_heads = 4,
        full_attn = (None, None, None, True),
    ):
        super().__init__()
    
        self.channels = channels
        input_channels = channels

        self.init_conv = nn.Conv3d(input_channels, init_dim, 3, padding = 1) # if want input and output to have same dimension, Kernel size to any odd number (e.g., 3, 5, 7, etc.). Padding to (kernel size - 1) / 2.

        dims = [init_dim, *map(lambda m: init_dim * m, dim_mults)]  # if initi_dim = 16, then [16, 32, 64, 128, 256]

        in_out = list(zip(dims[:-1], dims[1:])) 
        print('in out is : ', in_out)
        # [(16,32), (32,64), (64,128), (128,256)]. Each tuple in in_out represents a pair of input and output dimensions for different stages in a neural network 


        # attention

        num_stages = len(dim_mults)
        full_attn  = cast_tuple(full_attn, num_stages)
        attn_heads = cast_tuple(attn_heads, num_stages)
        attn_dim_head = cast_tuple(attn_dim_head, num_stages)

        assert len(full_attn) == len(dim_mults)

        # layers

        self.downs = nn.ModuleList([])
        self.ups = nn.ModuleList([])
        num_resolutions = len(in_out) # 4

        for ind, ((dim_in, dim_out), layer_full_attn, layer_attn_heads, layer_attn_dim_head) in enumerate(zip(in_out, full_attn, attn_heads, attn_dim_head)):
            # print(' in downsampling path, ind is: ', ind, ' dim_in is: ', dim_in, ' dim_out is: ', dim_out, ' layer_full_attn is: ', layer_full_attn, ' layer_attn_heads is: ', layer_attn_heads, ' layer_attn_dim_head is: ', layer_attn_dim_head)
            is_last = ind >= (num_resolutions - 1)

            # in each downsample stage, 
            # we have a resnetblock and then downsampling layer (downsample x and y by 2, then increase the feature number by 2)
            self.downs.append(nn.ModuleList([
                ResnetBlock3D(dim_in, dim_in, use_full_attention = layer_full_attn, attn_head = layer_attn_heads, attn_dim_head = layer_attn_dim_head),
                Downsample3D(dim_in, dim_out) if not is_last else nn.Conv3d(dim_in, dim_out, 3, padding = 1)
            ]))

        mid_dim = dims[-1]
        self.mid_block = ResnetBlock3D(mid_dim, mid_dim, use_full_attention = True, attn_head = attn_heads[-1], attn_dim_head = attn_dim_head[-1])

        for ind, ((dim_in, dim_out), layer_full_attn, layer_attn_heads, layer_attn_dim_head) in enumerate(zip(*map(reversed, (in_out, full_attn, attn_heads, attn_dim_head)))):
            # print(' in upsampling path, ind is: ', ind, ' dim_in is: ', dim_in, ' dim_out is: ', dim_out, ' layer_full_attn is: ', layer_full_attn, ' layer_attn_heads is: ', layer_attn_heads, ' layer_attn_dim_head is: ', layer_attn_dim_head)
            is_last = ind == (len(in_out) - 1)
          
            # in each upsample stage,
            # we have a resnetblock and then upsampling layer (upsample x and y by 2, then decrease the feature number by 2)
            self.ups.append(nn.ModuleList([
                ResnetBlock3D(dim_out + dim_in, dim_out, use_full_attention = layer_full_attn, attn_head = layer_attn_heads, attn_dim_head = layer_attn_dim_head),
                Upsample3D(dim_out, dim_in) if not is_last else  nn.Conv3d(dim_out, dim_in, 5, padding = 2)  
            ]))

        self.out_dim = channels

        self.final_res_block = ResnetBlock3D(init_dim * 2, init_dim, use_full_attention = None, attn_head = attn_heads[0], attn_dim_head = attn_dim_head[0])
        self.final_conv = nn.Conv3d(init_dim, self.out_dim, 1)  # output channel is initial channel number

    def forward(self, x):

        x = self.init_conv(x)
        # print('initial x shape is: ', x.shape)
        x_init = x.clone()

        h = []
        for block, downsample in self.downs:
            x = block(x)
            h.append(x)

            x = downsample(x)
        
        x = self.mid_block(x)
        
        for block, upsample in self.ups:
            x = torch.cat((x, h.pop()), dim = 1)   # h.pop() is the output of the corresponding downsample stage
            x = block(x)
            x = upsample(x)

        x = torch.cat((x, x_init), dim = 1)

        x = self.final_res_block(x)
        final_image = self.final_conv(x)
        # print('final image shape is: ', final_image.shape)
      
        return final_image
