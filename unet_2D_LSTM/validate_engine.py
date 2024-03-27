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
    if args.turn_zero_seg_slice_into is not None:
        seg_criterion = torch.nn.CrossEntropyLoss(ignore_index = 10)
    else:
        seg_criterion = torch.nn.CrossEntropyLoss()

    loss_list = []
    ce_loss_list = []
    dice_loss_list = []

    assert len(data_loader_valid) == len(args.dataset_valid)

    # iterate over dataset
    for i in range(len(data_loader_valid)):
        current_data_loader = data_loader_valid[i] # this is an iterable
        current_dataset_name = args.dataset_valid[i][0]
        current_slice_type = args.dataset_valid[i][1]
        print('in validation: current slice type: ', current_slice_type, ' current dataset name: ', current_dataset_name)

        for batch_idx, batch in enumerate(current_data_loader, 1):
            with torch.cuda.amp.autocast():
                
                # image
                batch_image = rearrange(batch['image'], 'b c h w d -> (b d) c h w')
                image_input = torch.clone(batch_image).to(torch.float16).to("cuda")

                # segmentation
                batch_seg = rearrange(batch['mask'], 'b c h w d -> (b d) c h w').to("cuda")

                seg_pred = model(image_input)

                # CE loss
                ce_loss = seg_criterion(seg_pred, torch.clone(batch_seg).squeeze(1).long()) 

                # Dice loss
                dice_loss = ff.customized_dice_loss(seg_pred,torch.clone(batch_seg).squeeze(1).long(), num_classes = args.num_classes, exclude_index = args.turn_zero_seg_slice_into)

                loss = args.loss_weight[0] * ce_loss + args.loss_weight[1] * dice_loss


            loss_list.append(loss.item())
            ce_loss_list.append(ce_loss.item())
            dice_loss_list.append(dice_loss.item())
            torch.cuda.synchronize()


    return [sum(loss_list) / len(loss_list), sum(ce_loss_list) / len(ce_loss_list), sum(dice_loss_list) / len(dice_loss_list)]


def pred_save(batch, output,args, save_folder):

    pred_softmax = F.softmax(output,dim = 1)
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

    patient_id = batch["patient_id"][0]
   
    save_folder_sub = os.path.join(save_folder, patient_id, 'epoch-' + str(args.pretrained_model_epoch)); ff.make_folder([os.path.dirname(save_folder_sub),save_folder_sub])

    nb.save(nb.Nifti1Image(final_pred_seg, affine), os.path.join(save_folder_sub, 'pred_seg_%s.nii.gz' % slice_index))
    nb.save(nb.Nifti1Image(original_image, affine), os.path.join(save_folder_sub, 'original_image_%s.nii.gz' % slice_index))
    nb.save(nb.Nifti1Image(original_seg, affine), os.path.join(save_folder_sub, 'original_seg_%s.nii.gz' % slice_index))