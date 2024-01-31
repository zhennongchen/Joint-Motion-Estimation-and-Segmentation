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
    flow_criterion = torch.nn.MSELoss()
    if args.turn_zero_seg_slice_into is not None:
        print('ignore index: ', args.turn_zero_seg_slice_into)
        seg_criterion = torch.nn.CrossEntropyLoss(ignore_index = args.turn_zero_seg_slice_into)
    else:
        seg_criterion = torch.nn.CrossEntropyLoss()

    loss_list = []
    flow_loss_list = []
    seg_loss_list = []
    warp_seg_loss_list = []
    dice_loss_list = []

    for batch_idx, batch in enumerate(data_loader_train, 1):
        with torch.cuda.amp.autocast():
            # image
            batch_image = rearrange(batch['image'], 'b c h w d -> (b d) c h w')
            image_target = torch.clone(batch_image)[:1,:]
            image_target = torch.repeat_interleave(image_target, 15, dim=0).to("cuda")

            image_source = torch.clone(batch_image).to("cuda")

            # segmentation
            batch_seg = rearrange(batch['mask'], 'b c h w d -> (b d) c h w ')
            seg_gt = torch.clone(batch_seg).to("cuda")
            # print('seg_gt shape: ', seg_gt.shape, ' unique: ', torch.unique(seg_gt))

            optimizer.zero_grad()
            net = model(image_target, image_source, image_target)


            flow_loss = flow_criterion(net['fr_st'], image_source) + 0.01 * ff.huber_loss(net['out'])
            seg_loss = seg_criterion(net['outs'],seg_gt.squeeze(1).long())
            # warp seg loss
            seg_time0 = torch.clone(batch_seg)[:1,:]
            seg_time0 = torch.repeat_interleave(seg_time0, 15, dim=0).to("cuda")
            warp_seg_loss = seg_criterion(net['warped_outs'], seg_time0.squeeze(1).long())

            loss = args.loss_weight[0] * flow_loss +  args.loss_weight[1] * seg_loss + args.loss_weight[2] * warp_seg_loss
            loss.backward()
            optimizer.step()

            # calculate Dice loss as well
            pred_seg = net['outs']
            mask_for_dice = rearrange(batch['mask'], 'b c h w d -> (b c) (h w d) ').to("cuda")

            Dice_loss = ff.customized_dice_loss(pred_seg, mask_for_dice.long(), num_classes = args.num_classes, exclude_index = args.turn_zero_seg_slice_into)

            # pred_softmax = F.softmax(net["outs"],dim = 1)
            # pred_seg = np.rollaxis(pred_softmax.argmax(1).detach().cpu().numpy(), 0, 3)
            # print('unique pred_seg: ', np.unique(pred_seg))
            

        loss_list.append(loss.item()) 
        flow_loss_list.append(flow_loss.item())
        seg_loss_list.append(seg_loss.item())
        warp_seg_loss_list.append(warp_seg_loss.item())
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