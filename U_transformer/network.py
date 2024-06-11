from typing import List, Tuple
import torch
import torch.nn as nn

import Joint_motion_seg_estimate_CMR.U_transformer.config as config
from Joint_motion_seg_estimate_CMR.U_transformer.EncoderDecoder import ConvBlock, EncoderLayer, DecoderLayer

DEVICE = config.get_device()

class TransformerUNet(nn.Module):
    def __init__(self, channels: Tuple[int], num_heads = 2, num_classes = 2,  is_residual: bool = False, bias = False) -> None:
        super(TransformerUNet, self).__init__()

        self.channels = channels
        self.pos_encoding = PositionalEncoding()
        self.encode = nn.ModuleList([EncoderLayer(channels[i], channels[i + 1], is_residual, bias) for i in range(len(channels) - 2)])
        self.bottle_neck = ConvBlock(channels[-2], channels[-1], is_residual, bias)
        self.mhsa = MultiHeadSelfAttention(channels[-1], num_heads, bias)
        self.mhca = nn.ModuleList([MultiHeadCrossAttention(channels[i], num_heads, channels[i], channels[i + 1], bias) for i in reversed(range(1, len(channels) - 1))])
        self.decode = nn.ModuleList([DecoderLayer(channels[i + 1], channels[i], is_residual, bias) for i in reversed(range(1, len(channels) - 1))])
        self.output = nn.Conv2d(channels[1], num_classes, 1)

        self.init_weights()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # print('initially the x dimension is: ', x.size())
        skip_x_list: List[torch.Tensor] = []
        for i in range(len(self.channels) - 2):
            skip_x, x = self.encode[i](x)
            skip_x_list.append(skip_x)

        # print('after downsampling the x dimension is: ', x.size())
        x = self.bottle_neck(x)
        # print('bottle neck x dimension is: ', x.size())
        x = self.pos_encoding(x)
        # print('after pos encoding x dimension is: ', x.size())
        x = self.mhsa(x)

        for i, skip_x in enumerate(reversed(skip_x_list)):
            x = self.pos_encoding(x)
            skip_x = self.pos_encoding(skip_x)
            skip_x = self.mhca[i](skip_x, x)
            x = self.decode[i](skip_x, x)

        return self.output(x)
    
    def init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_uniform_(m.weight.data)
                if m.bias is not None:
                    m.bias.data.fill_(0.1)

class MultiHeadSelfAttention(nn.Module):
    def __init__(self, embed_dim: int, num_heads: int, bias=False) -> None:
        super(MultiHeadSelfAttention, self).__init__()

        self.mha = nn.MultiheadAttention(embed_dim, num_heads, bias=bias, batch_first=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Original input shape: [BT, C, H, W]
        BT, C, H, W = x.size()
        B = BT // 15  # Since T is known to be 15
        T = 15
        
        # Reshape to [B*H*W, T, C] for temporal attention
        x_temporal = x.view(B, T, C, H, W).permute(0, 3, 4, 1, 2).reshape(B * H * W, T, C)
        
        # Apply multihead self-attention on the temporal dimension
        x_temporal, _ = self.mha(x_temporal, x_temporal, x_temporal, need_weights=False)
        
        # Reshape back to [B, H, W, T, C]
        x_temporal = x_temporal.view(B, H, W, T, C)
        
        # Permute back to [B, T, C, H, W]
        x_temporal = x_temporal.permute(0, 3, 4, 1, 2)
        
        # Reshape to [BT, C, H, W] for spatial attention
        x_spatial = x_temporal.reshape(BT, C, H, W)
        
        # Reshape to [BT, H*W, C] for spatial attention
        x_spatial = x_spatial.permute(0, 2, 3, 1).reshape(BT, H * W, C)
        # print('before spatial mha, x dimension is: ', x_spatial.size())
        
        # Apply multihead self-attention on the spatial dimensions
        x_spatial, _ = self.mha(x_spatial, x_spatial, x_spatial, need_weights=False)
        
        # Reshape back to [BT, H, W, C]
        x_spatial = x_spatial.view(BT, H, W, C)
        
        # Permute back to [BT, C, H, W]
        x_spatial = x_spatial.permute(0, 3, 1, 2)
        
        return x_spatial
    
        # b, c, h, w = x.size()
        # x = x.permute(0, 2, 3, 1).view((b, h * w, c))
        # print('before mha, x dimension is: ', x.size())
        # x, _ = self.mha(x, x, x, need_weights=False)
        # return x.view((b, h, w, c)).permute(0, 3, 1, 2)

class MultiHeadCrossAttention(nn.Module):
    def __init__(self, embed_dim: int, num_heads: int, channel_S: int, channel_Y: int, bias=False) -> None:
        super(MultiHeadCrossAttention, self).__init__()

        self.conv_S = nn.Sequential(
            nn.MaxPool2d(2),
            nn.Conv2d(channel_S, channel_S, 1, bias=bias),
            nn.BatchNorm2d(channel_S),
            nn.ReLU()
        )

        self.conv_Y = nn.Sequential(
            nn.Conv2d(channel_Y, channel_S, 1, bias=bias),
            nn.BatchNorm2d(channel_S),
            nn.ReLU()
        )

        self.mha = nn.MultiheadAttention(embed_dim, num_heads, bias=bias, batch_first=True)

        self.upsample = nn.Sequential(
            nn.Conv2d(channel_S, channel_S, 1, bias=bias).apply(lambda m: nn.init.xavier_uniform_(m.weight.data)),
            nn.BatchNorm2d(channel_S),
            nn.Sigmoid(),
            nn.ConvTranspose2d(channel_S, channel_S, 2, 2, bias=bias)
        )

    def forward(self, s: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        s_enc = s
        s = self.conv_S(s)
        y = self.conv_Y(y)

        b, c, h, w = s.size()
        s = s.permute(0, 2, 3, 1).view((b, h * w, c))

        b, c, h, w = y.size()
        y = y.permute(0, 2, 3, 1).view((b, h * w, c))

        y, _ = self.mha(y, y, s, need_weights=False)
        y = y.view((b, h, w, c)).permute(0, 3, 1, 2)
        
        y = self.upsample(y)

        return torch.mul(y, s_enc)

class PositionalEncoding(nn.Module):
    def __init__(self) -> None:
        super(PositionalEncoding, self).__init__()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, c, h, w = x.size()
        pos_encoding = self.positional_encoding(h * w, c)
        pos_encoding = pos_encoding.permute(1, 0).unsqueeze(0).repeat(b, 1, 1)
        x = x.view((b, c, h * w)) + pos_encoding
        return x.view((b, c, h, w))

    def positional_encoding(self, length: int, depth: int) -> torch.Tensor:
        depth = depth / 2

        positions = torch.arange(length, dtype=config.DTYPE, device=DEVICE)
        depths = torch.arange(depth, dtype=config.DTYPE, device=DEVICE) / depth

        angle_rates = 1 / (10000**depths)
        angle_rads = torch.einsum('i,j->ij', positions, angle_rates)

        pos_encoding = torch.cat((torch.sin(angle_rads), torch.cos(angle_rads)), dim=-1)

        return pos_encoding