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
    seg_loss_dice_list = []
    seg_loss_ce_list = []   
    warp_seg_loss_dice_list = []
    warp_seg_loss_ce_list = []
    # dice_loss_list = []

    for batch_idx, batch in enumerate(data_loader_train, 1):
        with torch.cuda.amp.autocast():
            if batch_idx == 1 or batch_idx % args.accum_iter == 0 or batch_idx == len(data_loader_train) or batch_idx == len(data_loader_train) - 1:
                optimizer.zero_grad()
            
            # image
            batch_image = rearrange(batch['image'], 'b c h w d -> (b d) c h w')
            image_tf_0 = torch.clone(batch_image)[:1,:]
            image_tf_0 = torch.repeat_interleave(image_tf_0, 15, dim=0).to(torch.float16).to("cuda")

            image_tf_all = torch.clone(batch_image).to(torch.float16).to("cuda")

            # segmentation
            batch_seg = rearrange(batch['mask'], 'b c h w d -> (b d) c h w ')
            seg_gt_tf_all = torch.clone(batch_seg).to("cuda")

            seg_gt_tf_0 = torch.clone(batch_seg)[:1,:]
            seg_gt_tf_0 = torch.repeat_interleave(seg_gt_tf_0, 15, dim=0).to("cuda") 
          

            ### important!! model(image_alltimeframe, image_time0, image_alltimeframe)
            net = model(image_tf_all, image_tf_0, image_tf_all)

            #### calculate loss
            # flow loss (warp images from all time frames to image at time frame 0)
            flow_loss = flow_criterion(net['fr_st'], image_tf_0) + 0.01 * ff.huber_loss(net['out'])

            # seg loss (all time frame segmentation), use dice loss
            seg_loss_dice = ff.customized_dice_loss(net['outs'], seg_gt_tf_all.long(), num_classes = args.num_classes, exclude_index = args.turn_zero_seg_slice_into)
            seg_loss_ce = seg_criterion(net['outs'],seg_gt_tf_all.squeeze(1).long())

            # warp seg loss (warp segs from all time frames to seg at time frame 0)
            warp_seg_loss_dice = ff.customized_dice_loss(net['warped_outs'], seg_gt_tf_0.long(), num_classes = args.num_classes, exclude_index = args.turn_zero_seg_slice_into)
            warp_seg_loss_ce = seg_criterion(net['warped_outs'], seg_gt_tf_0.squeeze(1).long())
           
            # loss = args.loss_weight[0] * flow_loss +  args.loss_weight[1] * seg_loss + args.loss_weight[2] * warp_seg_loss
            loss = args.loss_weight[0] * flow_loss +  args.loss_weight[1] * seg_loss_dice + args.loss_weight[2] * seg_loss_ce + args.loss_weight[3] * warp_seg_loss_dice + args.loss_weight[4] * warp_seg_loss_ce

            if batch_idx == 1 or batch_idx % args.accum_iter == 0 or batch_idx == len(data_loader_train) or batch_idx == len(data_loader_train) - 1:
                loss.backward()
                optimizer.step()

            # calculate Dice loss as well
            # Dice_loss = ff.customized_dice_loss(net['outs'], torch.clone(batch_seg).to("cuda").long(), num_classes = args.num_classes, exclude_index = args.turn_zero_seg_slice_into)

            if batch_idx % 30 == 0 or batch_idx == len(data_loader_train):
                print('in this iteration loss: ', np.round(loss.item(),3), 'flow_loss: ', np.round(flow_loss.item(),3), 'seg_loss_dice: ', np.round(seg_loss_dice.item(),3), 'seg_loss_ce: ', np.round(seg_loss_ce.item(),3), 'warp_seg_loss_dice: ', np.round(warp_seg_loss_dice.item(),3), 'warp_seg_loss_ce: ', np.round(warp_seg_loss_ce.item(),3))

                pred_softmax = F.softmax(net["outs"],dim = 1)
                pred_seg1 = np.rollaxis(pred_softmax.argmax(1).detach().cpu().numpy(), 0, 3)

                pred_softmax_warp = F.softmax(net["warped_outs"],dim = 1)
                pred_seg_warp = np.rollaxis(pred_softmax_warp.argmax(1).detach().cpu().numpy(), 0, 3)
                print('unique pred_seg and pred_seg1: ', np.unique(pred_seg1), np.unique(pred_seg_warp))


        loss_list.append(loss.item()) 
        flow_loss_list.append(flow_loss.item())
        seg_loss_dice_list.append(seg_loss_dice.item())
        seg_loss_ce_list.append(seg_loss_ce.item())
        warp_seg_loss_dice_list.append(warp_seg_loss_dice.item())
        warp_seg_loss_ce_list.append(warp_seg_loss_ce.item())


    return sum(loss_list) / len(loss_list), sum(flow_loss_list) / len(flow_loss_list), sum(seg_loss_dice_list) / len(seg_loss_dice_list), sum(seg_loss_ce_list) / len(seg_loss_ce_list), sum(warp_seg_loss_dice_list) / len(warp_seg_loss_dice_list), sum(warp_seg_loss_ce_list) / len(warp_seg_loss_ce_list)