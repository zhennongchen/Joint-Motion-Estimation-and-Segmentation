import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.init as init
from torch.autograd import Variable, grad
import numpy as np
import math


# normalization functions

def Upsample2D(dim, dim_out , upsample_factor = (2,2)):
    return nn.Sequential(
        nn.Upsample(scale_factor = upsample_factor, mode = 'nearest'),
        nn.Conv2d(dim, dim_out, 3, padding = 1))


# 
def Downsample2D(dim, dim_out):
    return nn.Sequential(
        nn.MaxPool2d(kernel_size=(2,2), stride=(2,2), padding=0),
        nn.Conv2d(dim, dim_out, 1))
   

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
    
# building convLSTM module
class ConvLSTMCell(nn.Module):
    def __init__(self, input_channels, hidden_channels, kernel_size):
        super(ConvLSTMCell, self).__init__()
        
        self.input_channels = input_channels
        self.hidden_channels = hidden_channels
        self.kernel_size = kernel_size
        self.padding = kernel_size // 2

        self.conv = nn.Conv2d(in_channels=input_channels + hidden_channels,
                              out_channels=4 * hidden_channels,
                              kernel_size=kernel_size, padding=self.padding)

    def forward(self, input_tensor, hidden_state):
        h_cur, c_cur = hidden_state
        combined = torch.cat([input_tensor, h_cur], dim=1)  # concatenate along channel dimension

        combined_conv = self.conv(combined)
        cc_i, cc_f, cc_o, cc_c = torch.split(combined_conv, self.hidden_channels, dim=1)
        
        i = torch.sigmoid(cc_i)
        f = torch.sigmoid(cc_f)
        o = torch.sigmoid(cc_o)
        c = f * c_cur + i * torch.tanh(cc_c)
        h = o * torch.tanh(c)

        return h, c

    def init_hidden(self, batch_size, image_size):
        height, width = image_size
        return (torch.zeros(batch_size, self.hidden_channels, height, width, device=self.conv.weight.device),
                torch.zeros(batch_size, self.hidden_channels, height, width, device=self.conv.weight.device))

    

# model: UNet2D

class Unet2D(nn.Module):
    def __init__(
        self,
        init_dim = 32,
        channels = 1,
        dim_mults = (2,4,8),
        num_classes = 2,
    ):
        super().__init__()
    
        self.channels = channels
        input_channels = channels
        self.num_classes = num_classes

        self.init_conv = nn.Conv2d(input_channels, init_dim, 3, padding = 1) # if want input and output to have same dimension, Kernel size to any odd number (e.g., 3, 5, 7, etc.). Padding to (kernel size - 1) / 2.

        dims = [init_dim, *map(lambda m: init_dim * m, dim_mults)]  # if initi_dim = 16, then [16, 32, 64, 128, 256]

        in_out = list(zip(dims[:-1], dims[1:])) 
        print('in out is : ', in_out)
        # [(16,32), (32,64), (64,128), (128,256)]. Each tuple in in_out represents a pair of input and output dimensions for different stages in a neural network 

        # layers
        self.downs = nn.ModuleList([])
        self.ups = nn.ModuleList([])
        num_resolutions = len(in_out) # 3

        for ind, (dim_in, dim_out) in enumerate(in_out):
            print(' in downsampling path, ind is: ', ind, ' dim_in is: ', dim_in, ' dim_out is: ', dim_out)

            # in each downsample stage, 
            # we have two conv blocks and then downsampling layer (downsample x and y by 2, then increase the feature number by 2)
            self.downs.append(nn.ModuleList([
                ConvBlock2D(dim_in, dim_in),
                ConvBlock2D(dim_in, dim_in),
                Downsample2D(dim_in, dim_out ) 
            ]))

        self.mid_conv = ConvBlock2D(dims[-1], dims[-1])

        self.mid_convLSTM = ConvLSTMCell(dims[-1], dims[-2], 3)

        self.mid_conv2 = ConvBlock2D(dims[-2], dims[-1])

        for ind, (dim_in, dim_out) in enumerate(reversed(in_out)):
            print(' in upsampling path, ind is: ', ind, ' dim_in is: ', dim_in, ' dim_out is: ', dim_out)
          
            # in each upsample stage,
            # we have one upsampling layer (upsample x and y by 2, then decrease the feature number by 2) and then two conv blocks
            self.ups.append(nn.ModuleList([
                Upsample2D(dim_out, dim_in),
                ConvBlock2D(dim_in * 2, dim_in),  # dim_in * 2 is because we concatenate the output of upsampling layer and the output of the corresponding downsample layer
                ConvBlock2D(dim_in, dim_in)
            ]))
               
        self.final_block = nn.Conv2d(init_dim, self.num_classes, 1)


    def forward(self, x):

        x = self.init_conv(x)

        h = []
        for block1, block2, downsample in self.downs:
            x = block1(x)
            x = block2(x)
            h.append(x)
            x = downsample(x)

        x = self.mid_conv(x)
        print('x shape before LSTM is: ', x.shape)
        
        # convLSTM
        # Assuming x is of shape [15, 256, 16, 16] where 15 is the sequence_length
        batch_size, channels, height, width = x.shape
        hidden_state = self.mid_convLSTM.init_hidden(1, (height, width))  # Initialize hidden state

        # Process each time step through the ConvLSTM cell
        output_sequence = []
        for t in range(batch_size):
            hidden_state = self.mid_convLSTM(x[t].unsqueeze(0), hidden_state)  # Process each time step
            output_sequence.append(hidden_state[0].squeeze(0))

        # Reconstruct the tensor from the output sequence
        x = torch.stack(output_sequence)

        print('x shape after convLSTM is: ', x.shape)

        x = self.mid_conv2(x)

        for upsample, block1, block2 in self.ups:
            x = upsample(x)
            x = torch.cat([x, h.pop()], dim = 1)
            x = block1(x)
            x = block2(x)

        final_segmentation = self.final_block(x)

        print('final_segmentation shape is: ', final_segmentation.shape)
      
        return final_segmentation



