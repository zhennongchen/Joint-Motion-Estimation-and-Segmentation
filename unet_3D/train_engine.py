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
            print('in train image_input shape: ', image_input.shape)

            # segmentation
            batch_seg = batch['mask']

            optimizer.zero_grad()
            seg_pred = model(image_input)
            print('in train seg_pred shape: ', seg_pred.shape)

            # # CE loss
            # seg_gt_CE = rearrange(torch.clone(batch_seg), 'b c h w d -> (b d) c h w').to("cuda")
            # seg_pred_CE = rearrange(torch.clone(seg_pred), 'b c h w d -> (b d) c h w')
            # print('in train seg_gt_CE ad seg_pred_CE shape: ', seg_gt_CE.shape, seg_pred_CE.shape)
            seg_gt_CE = torch.clone(batch_seg).to("cuda")
            print('unique seg_gt_CE: ', torch.unique(seg_gt_CE))
            print('shape seg_gt_CE: ', seg_gt_CE.shape)
            ce_loss = seg_criterion(seg_pred, seg_gt_CE.squeeze(0).long())

            # Dice loss
            dice_loss = ff.customized_dice_loss(seg_pred,torch.clone(batch_seg).to("cuda").long(), num_classes = args.num_classes, exclude_index = args.turn_zero_seg_slice_into)

            # loss = args.loss_weight[0] * ce_loss + args.loss_weight[1] * dice_loss
            loss = ce_loss
            pred_softmax = F.softmax(seg_pred,dim = 1)
            print('pred_softmax shape: ', pred_softmax.shape)
            pred_seg_softmax = pred_softmax.argmax(1).detach().cpu().numpy()
            print('pred_seg_softmax shape: ', pred_seg_softmax.shape)
            print('unique pred_seg_softmax: ', np.unique(pred_seg_softmax))

            loss.backward()
            optimizer.step()

        loss_list.append(loss.item())
        ce_loss_list.append(ce_loss.item())
        dice_loss_list.append(dice_loss.item())

        if batch_idx % 30 == 0:
            print('in this iteration loss: ', loss.item(), ' ce_loss: ', ce_loss.item(), ' dice_loss: ', dice_loss.item())

    return sum(loss_list) / len(loss_list), sum(ce_loss_list) / len(ce_loss_list), sum(dice_loss_list) / len(dice_loss_list)
