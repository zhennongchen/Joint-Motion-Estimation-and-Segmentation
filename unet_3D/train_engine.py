import os
from tqdm import tqdm
import torch
import nibabel as nb
import numpy as np
from typing import Iterable
import logging
from einops import rearrange
import torch.nn.functional as F
import Joint_motion_seg_estimate_CMR.functions_collection as ff


def train_loop(args, model, data_loader_train, optimizer):

    model.train(True)

    # define loss
    if args.turn_zero_seg_slice_into is not None:
        print('ignore index: ', args.turn_zero_seg_slice_into)
        seg_criterion = torch.nn.CrossEntropyLoss(ignore_index = args.turn_zero_seg_slice_into)
    else:
        seg_criterion = torch.nn.CrossEntropyLoss()

    loss_list = []
    ce_loss_list = []
    dice_loss_list = []

    for batch_idx, batch in enumerate(data_loader_train, 1):
        with torch.cuda.amp.autocast():
            # image
            batch_image = batch['image']
            image_input = torch.clone(batch_image).to("cuda")
            print('image_input shape: ', image_input.shape)

            # segmentation
            batch_seg = batch['mask']

            optimizer.zero_grad()
            seg_pred = model(image_input)
            print('seg_pred shape: ', seg_pred.shape, ' unique: ', torch.unique(seg_pred))

            # CE loss
            seg_gt_CE = rearrange(batch_seg, 'b c h w d -> (b d) c h w ').to("cuda")
            seg_pred_CE = rearrange(seg_pred, 'b c h w d -> (b d) c h w ')
            ce_loss = seg_criterion(seg_pred_CE,seg_gt_CE.squeeze(1).long())

            

        loss_list.append(loss.item())
        seg_loss_list.append(seg_loss.item())
        dice_loss_list.append(Dice_loss.item())

        if batch_idx % 30 == 0:
            print('in this iteration loss: ', loss.item(), 'flow_loss: ', flow_loss.item(), 'seg_loss: ', seg_loss.item(), 'warp_seg_loss: ', warp_seg_loss.item(), 'Dice_loss: ', Dice_loss.item())

    return sum(loss_list) / len(loss_list), sum(flow_loss_list) / len(flow_loss_list), sum(seg_loss_list) / len(seg_loss_list), sum(warp_seg_loss_list) / len(warp_seg_loss_list), sum(dice_loss_list) / len(dice_loss_list)

# def train_loop_motion(args, model, data_loader_train, optimizer):
#     epoch_loss = []
#     for batch_idx, batch in enumerate(data_loader_train, 1):
#         batch_image = rearrange(batch['image'], 'b c h w -> c b h w')

#         image_target = torch.clone(batch_image)[:1,:]
#         image_target = torch.repeat_interleave(image_target, 15, dim=0).to("cuda")

#         image_source = torch.clone(batch_image).to("cuda")

#         optimizer.zero_grad()
#         net = model(image_target, image_source, image_target)

#         flow_loss = flow_criterion(net['fr_st'], image_source) + 0.01 * ff.huber_loss(net['out'])

#         flow_loss.backward()
#         optimizer.step()
            
#         epoch_loss.append(flow_loss.item())

#         if (batch_idx + 1) % 10 == 0:
#             print('batch index: ', batch_idx + 1, 'average loss so far: ',sum(epoch_loss) / len(epoch_loss))

#     return sum(epoch_loss) / len(epoch_loss)