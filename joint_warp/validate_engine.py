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


def valid_loop(args, model, data_loader_valid):

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

    for batch_idx, batch in enumerate(data_loader_valid, 1):
        with torch.cuda.amp.autocast():
            # image
            batch_image = rearrange(batch['image'], 'b c h w d -> (b d) c h w')
            image_tf_0 = torch.clone(batch_image)[:1,:]
            image_tf_0 = torch.repeat_interleave(image_tf_0, 15, dim=0).to("cuda")

            image_tf_all = torch.clone(batch_image).to("cuda")

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
            seg_loss = ff.customized_dice_loss(net['outs'], seg_gt_tf_all.long(), num_classes = args.num_classes, exclude_index = args.turn_zero_seg_slice_into)
            # seg_loss = seg_criterion(net['outs'],seg_gt_tf_all.squeeze(1).long())

            # warp seg loss (warp segs from all time frames to seg at time frame 0)
            warp_seg_loss = ff.customized_dice_loss(net['warped_outs'], seg_gt_tf_0.long(), num_classes = args.num_classes, exclude_index = args.turn_zero_seg_slice_into)
            # warp_seg_loss = seg_criterion(net['warped_outs'], seg_gt_tf_0.squeeze(1).long())
           
            loss = args.loss_weight[0] * flow_loss +  args.loss_weight[1] * seg_loss + args.loss_weight[2] * warp_seg_loss

            # calculate Dice loss as well
            pred_seg = net['outs']
            Dice_loss = ff.customized_dice_loss(pred_seg, torch.clone(batch_seg).to("cuda").long(), num_classes = args.num_classes, exclude_index = args.turn_zero_seg_slice_into)


        loss_list.append(loss.item()) 
        flow_loss_list.append(flow_loss.item())
        seg_loss_list.append(seg_loss.item())
        warp_seg_loss_list.append(warp_seg_loss.item())
        dice_loss_list.append(Dice_loss.item())
        torch.cuda.synchronize()

    return sum(loss_list) / len(loss_list), sum(flow_loss_list) / len(flow_loss_list), sum(seg_loss_list) / len(seg_loss_list),  sum(warp_seg_loss_list) / len(warp_seg_loss_list), sum(dice_loss_list) / len(dice_loss_list)


def pred_save(batch, output,args):

    pred_softmax = F.softmax(output["outs"],dim = 1)
    pred_seg = np.rollaxis(pred_softmax.argmax(1).detach().cpu().numpy(), 0, 3)
                        

    original_shape = np.array([x.item() for x in batch["original_shape"]])
    centroid = batch["centroid"].numpy().flatten()
              
    crop_start_end_list = []
    for dim, size in enumerate([args.img_size, args.img_size]):
        start = max(centroid[dim] - size // 2, 0)
        end = start + size
        # Adjust the start and end if they are out of bounds
        if end > original_shape[dim]:
            end = original_shape[dim]
            start = max(end - size, 0)
        crop_start_end_list.append([start, end])
     

    final_pred_seg = np.zeros(original_shape)
    final_pred_seg[crop_start_end_list[0][0]:crop_start_end_list[0][1], crop_start_end_list[1][0]:crop_start_end_list[1][1]] = pred_seg

    # save original image and ground truth segmentation
    if args.full_or_nonzero_slice[0:4] == 'full':
        original_image_file = batch["image_full_slice_file"][0]
        original_seg_file = batch["seg_full_slice_file"][0]
    elif args.full_or_nonzero_slice[0:4] == 'nonz':
        original_image_file = batch["image_nonzero_slice_file"][0]
        original_seg_file = batch["seg_nonzero_slice_file"][0]
    elif args.full_or_nonzero_slice[0:4] == 'loos':
        original_image_file = batch["image_nonzero_slice_file_loose"][0]
        original_seg_file = batch["seg_nonzero_slice_file_loose"][0]
                    
    slice_index = batch["slice_index"].item()

    affine = nb.load(original_image_file).affine
    original_image = nb.load(original_image_file).get_fdata()[:,:,slice_index,:]
    original_seg = nb.load(original_seg_file).get_fdata()[:,:,slice_index,:]

    save_folder = os.path.join(args.output_dir, 'predicts'); ff.make_folder([save_folder])

    patient_id = batch["patient_id"][0]
   
    save_folder_sub = os.path.join(save_folder, patient_id, 'epoch-' + str(args.pretrained_model_epoch)); ff.make_folder([os.path.dirname(save_folder_sub),save_folder_sub])

    nb.save(nb.Nifti1Image(final_pred_seg, affine), os.path.join(save_folder_sub, 'pred_seg_%s.nii.gz' % slice_index))
    nb.save(nb.Nifti1Image(original_image, affine), os.path.join(save_folder_sub, 'original_image_%s.nii.gz' % slice_index))
    nb.save(nb.Nifti1Image(original_seg, affine), os.path.join(save_folder_sub, 'original_seg_%s.nii.gz' % slice_index))






# def valid_loop_motion(args, model, data_loader_valid):
#     valid_loss = []
#     for batch_idx, batch in enumerate(data_loader_valid, 1):
#         batch_image = rearrange(batch['image'], 'b c h w -> c b h w')

#         image_target = torch.clone(batch_image)[:1,:]
#         image_target = torch.repeat_interleave(image_target, 15, dim=0).to("cuda")

#         image_source = torch.clone(batch_image).to("cuda")

#         net = model(image_target, image_source, image_target)

#         flow_loss = flow_criterion(net['fr_st'], image_source) + 0.01 * ff.huber_loss(net['out'])

#         valid_loss.append(flow_loss.item())

#         torch.cuda.synchronize()

#     return sum(valid_loss) / len(valid_loss)