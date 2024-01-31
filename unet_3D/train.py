import sys 
sys.path.append('/workspace/Documents')
import os
import torch
import numpy as np
import network as Unet


# build model

model = Unet.Unet3D(
    init_dim = 16,
    channels = 1, 
    dim_mults = (2,4,8,16),
    num_classes = 2,
)

# run the model
a = torch.rand(1, 1, 256,256, 10)
b = model(a)
   