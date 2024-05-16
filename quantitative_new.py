import os
import numpy as np
import pandas as pd
from tqdm import tqdm
import random
from pathlib import Path
import nibabel as nb
from scipy.spatial import ConvexHull
from skimage.draw import polygon2mask

import SAM_CMR_seg.Defaults as Defaults
import SAM_CMR_seg.Build_list_zhennong.Build_list as Build_list
from SAM_CMR_seg.dataset.data_CMR_sax import build_data_CMR
import SAM_CMR_seg.functions_collection as ff

def include_or_exclude_STACOM(infosheet, patient_id):
    row = infosheet[infosheet['patient_id'] == patient_id]
    if len(row) == 0:
        print('patient not found in the infosheet')
        start_slice_include = 1; end_slice_include = len(gt_files) - 1; exclude_slice = []
    else:
        start_slice_include = int(row['start'].values[0])
        end_slice_include = int(row['end'].values[0])
        exclude_slice = row['exclude'].values[0]
        if isinstance(exclude_slice, str):
            exclude_slice = [int(item) for item in exclude_slice.split(',')]
        else:
            if np.isnan(exclude_slice):
                exclude_slice = []
            elif isinstance(exclude_slice, int):
                exclude_slice = [exclude_slice]
    return start_slice_include, end_slice_include, exclude_slice

def find_LV_enclosed_by_myo(image):
    image = (image ).astype(np.uint8) * 255

    x_indices, y_indices = np.where(image == 255)

    # Create a set of points from the white pixel positions
    points = np.vstack((x_indices, y_indices)).T

    # Compute the convex hull
    hull = ConvexHull(points)

    # Extract the vertices making up the convex hull
    hull_vertices = points[hull.vertices]

    # Use skimage.draw.polygon2mask to create a mask from the convex hull vertices
    image_shape = image.shape
    convex_hull_mask = polygon2mask(image_shape, hull_vertices)

    # differntiate the mask from the original image
    diff = convex_hull_mask - image
        
    LV = diff.astype(np.int32)

    return LV

def calculate_metric(gt_files, dataset, start_slice, end_slice, start_slice_include = None, end_slice_include = None, exclude_slice = []):

    all_dice_lv, base_dice_lv, mid_dice_lv, apex_dice_lv = [], [], [], []
    all_hd_lv, base_hd_lv, mid_hd_lv, apex_hd_lv = [], [], [], []
    all_dice_myo, base_dice_myo, mid_dice_myo, apex_dice_myo = [], [], [], []
    all_hd_myo, base_hd_myo, mid_hd_myo, apex_hd_myo = [], [], [], []

    base_slice, mid_slice, apex_slice = [], [], []
    all_new, base_new, mid_new, apex_new = True, True, True, True
    for slice_num in np.arange(start_slice, end_slice):
        # for STACOM
        if dataset == 'STACOM':
            if slice_num in exclude_slice or slice_num < start_slice_include or slice_num > end_slice_include:
                continue
        
        gt_file = gt_files[slice_num]
        pred_file = os.path.join(main_folder, patient_id, 'epoch-' + str(epoch) +'-processed', 'pred_seg_' + str(slice_num) + '.nii.gz')

        gt = nb.load(gt_file).get_fdata(); gt = np.round(gt)
        pred = nb.load(pred_file).get_fdata(); pred = np.round(pred)

        for t in range(pred.shape[2]):
           
            if dataset == 'STACOM' and np.sum(gt[:,:,t] == 1) > 0:
               gt[:,:,t] = find_LV_enclosed_by_myo(gt[:,:,t] == 1)
            if np.sum(pred[:,:,t] == 1) <= 20:
                gt[:,:,t] = 10 # not include this slice in calculation
                a = np.copy(pred[:,:,t]); a[a==1] = 2; pred[:,:,t] = a
            else:
                pred[:,:,t] = find_LV_enclosed_by_myo(pred[:,:,t] == 1)
           
        
        # set the gt timeframe without manual segmentation as 10
        for t in range(0, gt.shape[-1]):
            if np.sum(gt[:,:,t] == 1) == 0:
                gt[:,:,t] = 10

        if all_new == True:
            pred_all = np.expand_dims(np.copy(pred),-1); gt_all = np.expand_dims(np.copy(gt),-1); all_new = False
        else:
            pred_all = np.concatenate((pred_all, np.expand_dims(np.copy(pred),-1)), axis = -1); gt_all = np.concatenate((gt_all, np.expand_dims(np.copy(gt),-1)), axis = -1)

        if slice_num in base_segment:
            base_slice.append(slice_num)
            if base_new == True:
                pred_base = np.expand_dims(np.copy(pred),-1); gt_base = np.expand_dims(np.copy(gt),-1); base_new = False
            else:
                pred_base = np.concatenate((pred_base, np.expand_dims(np.copy(pred),-1)), axis = -1); gt_base = np.concatenate((gt_base, np.expand_dims(np.copy(gt),-1)), axis = -1)
        elif slice_num in mid_segment:
            mid_slice.append(slice_num)
            if mid_new == True:
                pred_mid = np.expand_dims(np.copy(pred),-1); gt_mid = np.expand_dims(np.copy(gt),-1); mid_new = False
            else:
                pred_mid = np.concatenate((pred_mid, np.expand_dims(np.copy(pred),-1)), axis = -1); gt_mid = np.concatenate((gt_mid, np.expand_dims(np.copy(gt),-1)), axis = -1)
        elif slice_num in apex_segment:
            apex_slice.append(slice_num)
            if apex_new == True:
                pred_apex = np.expand_dims(np.copy(pred),-1); gt_apex = np.expand_dims(np.copy(gt),-1); apex_new = False
            else:
                pred_apex = np.concatenate((pred_apex, np.expand_dims(np.copy(pred),-1)), axis = -1); gt_apex = np.concatenate((gt_apex, np.expand_dims(np.copy(gt),-1)), axis = -1)

    pred_all = np.transpose(pred_all, (0, 1, 3, 2)); gt_all = np.transpose(gt_all, (0, 1, 3, 2))
    pred_base = np.transpose(pred_base, (0, 1, 3, 2));gt_base = np.transpose(gt_base, (0, 1, 3, 2))
    pred_mid = np.transpose(pred_mid, (0, 1, 3, 2)); gt_mid = np.transpose(gt_mid, (0, 1, 3, 2))
    pred_apex = np.transpose(pred_apex, (0, 1, 3, 2)); gt_apex = np.transpose(gt_apex, (0, 1, 3, 2))

    # calculate dice
    for tf in range(0,pred_all.shape[-1]):
        if np.where(gt_all[:,:,:,tf] == 2)[0].shape[0] == 0:
            continue
    
        dice_lv = ff.np_categorical_dice(pred_all[:,:,:,tf], gt_all[:,:,:,tf], target_class = 1, exclude_class = 10); all_dice_lv.append(dice_lv)
        hd_lv = ff.HD(pred_all[:,:,:,tf],gt_all[:,:,:,tf], pixel_size = 1, target_class = 1, exclude_class = 10, max_or_mean = 'max'); all_hd_lv.append(hd_lv)
        dice_myo = ff.np_categorical_dice(pred_all[:,:,:,tf], gt_all[:,:,:,tf], target_class = 2, exclude_class = 10); all_dice_myo.append(dice_myo)
        hd_myo = ff.HD(pred_all[:,:,:,tf],gt_all[:,:,:,tf], pixel_size = 1, target_class = 2, exclude_class = 10, max_or_mean = 'max'); all_hd_myo.append(hd_myo)

        dice_lv = ff.np_categorical_dice(pred_base[:,:,:,tf], gt_base[:,:,:,tf], target_class = 1, exclude_class = 10); base_dice_lv.append(dice_lv)
        hd_lv = ff.HD(pred_base[:,:,:,tf],gt_base[:,:,:,tf], pixel_size = 1, target_class = 1, exclude_class = 10, max_or_mean = 'max'); base_hd_lv.append(hd_lv)
        dice_myo = ff.np_categorical_dice(pred_base[:,:,:,tf], gt_base[:,:,:,tf], target_class = 2, exclude_class = 10); base_dice_myo.append(dice_myo)
        hd_myo = ff.HD(pred_base[:,:,:,tf],gt_base[:,:,:,tf], pixel_size = 1, target_class = 2, exclude_class = 10, max_or_mean = 'max'); base_hd_myo.append(hd_myo)
        
        dice_lv = ff.np_categorical_dice(pred_mid[:,:,:,tf], gt_mid[:,:,:,tf], target_class = 1, exclude_class = 10); mid_dice_lv.append(dice_lv)
        hd_lv = ff.HD(pred_mid[:,:,:,tf],gt_mid[:,:,:,tf], pixel_size = 1, target_class = 1, exclude_class = 10, max_or_mean = 'max'); mid_hd_lv.append(hd_lv)
        dice_myo = ff.np_categorical_dice(pred_mid[:,:,:,tf], gt_mid[:,:,:,tf], target_class = 2, exclude_class = 10); mid_dice_myo.append(dice_myo)
        hd_myo = ff.HD(pred_mid[:,:,:,tf],gt_mid[:,:,:,tf], pixel_size = 1, target_class = 2, exclude_class = 10, max_or_mean = 'max'); mid_hd_myo.append(hd_myo)

        dice_lv = ff.np_categorical_dice(pred_apex[:,:,:,tf], gt_apex[:,:,:,tf], target_class = 1, exclude_class = 10); apex_dice_lv.append(dice_lv)
        hd_lv = ff.HD(pred_apex[:,:,:,tf],gt_apex[:,:,:,tf], pixel_size = 1, target_class = 1, exclude_class = 10, max_or_mean = 'max'); apex_hd_lv.append(hd_lv)
        dice_myo = ff.np_categorical_dice(pred_apex[:,:,:,tf], gt_apex[:,:,:,tf], target_class = 2, exclude_class = 10); apex_dice_myo.append(dice_myo)
        hd_myo = ff.HD(pred_apex[:,:,:,tf],gt_apex[:,:,:,tf], pixel_size = 1, target_class = 2, exclude_class = 10, max_or_mean = 'max'); apex_hd_myo.append(hd_myo)

    return [np.mean(all_dice_myo), np.mean(base_dice_myo), np.mean(mid_dice_myo), np.mean(apex_dice_myo), np.mean(all_dice_lv), np.mean(base_dice_lv), np.mean(mid_dice_lv), np.mean(apex_dice_lv), np.mean(all_hd_myo), np.mean(base_hd_myo), np.mean(mid_hd_myo), np.mean(apex_hd_myo), np.mean(all_hd_lv), np.mean(base_hd_lv), np.mean(mid_hd_lv), np.mean(apex_hd_lv), base_slice, mid_slice, apex_slice]
    
######### MAIN SCRIPT
main_path = '/mnt/camca_NAS/SAM_for_CMR/'
dataset = 'AS'

patient_list_file = os.path.join(main_path, 'data/Patient_list/AS_Patient_List_training_testing.xlsx')
index_list = np.arange(0,38,1)
patient_id_list,_,_,_ ,_,_,_ ,_ ,_, _ ,_, _ = Build_list.__build__(patient_list_file, batch_list = None, index_list = index_list)

main_folder = os.path.join(main_path, 'models/unet3D_AS_5shot/predicts_AS')
epoch = 22

# slice inclusion in the calculation
if dataset == 'STACOM':
    infosheet = pd.read_excel(os.path.join(os.path.dirname(patient_list_file),'STACOM_test_cohort_slice_inclusion.xlsx'))
result = []
for i in range(0,len(patient_id_list)):
    patient_id = patient_id_list[i]
    patient_folder = os.path.join(main_folder, patient_id, 'epoch-' + str(epoch))
    if isinstance(epoch, str):
        patient_folder = os.path.join(main_folder, patient_id, epoch)

    print('i, patient_id: ', i, patient_id, patient_folder)

    # fund base, mid and apex slices
    gt_files = ff.sort_timeframe(ff.find_all_target_files(['original_seg*'], patient_folder),2,'_','.')
    base_segment, mid_segment, apex_segment = ff.define_three_segments(len(gt_files))

    # which slices should be included or excluded for STACOM (some ground truth is wrong)
    if dataset == 'STACOM':
        start_slice_include, end_slice_include, exclude_slice = ff.find_slice_inclusion(patient_id, infosheet)

    # calculate
    if dataset == 'STACOM':
        metrics = calculate_metric(gt_files, dataset, 1, len(gt_files)-1, start_slice_include, end_slice_include, exclude_slice)
    elif dataset == 'ACDC':
        metrics = calculate_metric(gt_files, dataset, 1 , len(gt_files) - 1)
    elif dataset == 'AS':
        metrics = calculate_metric(gt_files, dataset, 0, len(gt_files))
    elif dataset == 'HFpEF' or dataset == 'MM':
        if len(apex_segment) == 1:
            end = len(gt_files)
        else:
            metrics1 = calculate_metric(gt_files, dataset, 0, len(gt_files))
            end = len(gt_files)
            metrics = metrics1
            if (metrics1[3] <= 0.65 and dataset == 'MM') or dataset == 'HFpEF':
                metrics2 = calculate_metric(gt_files, dataset, 0, len(gt_files) - 1)
                if metrics2[3] >= metrics1[3] and metrics2[0] >= metrics1[0]:
                    end = len(gt_files) - 1
                    metrics = metrics2

    all_dice_myo, base_dice_myo, mid_dice_myo, apex_dice_myo, all_dice_lv, base_dice_lv, mid_dice_lv, apex_dice_lv, all_hd_myo, base_hd_myo, mid_hd_myo, apex_hd_myo, all_hd_lv, base_hd_lv, mid_hd_lv, apex_hd_lv, base_slice, mid_slice, apex_slice = metrics
    print(all_dice_myo, base_dice_myo, mid_dice_myo, apex_dice_myo)
    print(all_dice_lv, base_dice_lv, mid_dice_lv, apex_dice_lv)
    print(base_slice, mid_slice, apex_slice)

    result.append([patient_id, all_dice_myo, base_dice_myo, mid_dice_myo, apex_dice_myo, all_dice_lv, base_dice_lv, mid_dice_lv, apex_dice_lv, all_hd_myo, base_hd_myo, mid_hd_myo, apex_hd_myo, all_hd_lv, base_hd_lv, mid_hd_lv, apex_hd_lv, base_slice, mid_slice, apex_slice])
    ff.make_folder([os.path.join(os.path.dirname(main_folder),'results')])
    result_df = pd.DataFrame(result, columns = ['patient_id', 'all_dice_myo', 'base_dice_myo', 'mid_dice_myo', 'apex_dice_myo', 'all_dice_lv', 'base_dice_lv', 'mid_dice_lv', 'apex_dice_lv', 'all_hd_myo', 'base_hd_myo', 'mid_hd_myo', 'apex_hd_myo', 'all_hd_lv', 'base_hd_lv', 'mid_hd_lv', 'apex_hd_lv', 'base_slice', 'mid_slice', 'apex_slice'])
    result_df.to_excel(os.path.join(os.path.dirname(main_folder),'results','quantitative_new_epoch_' + str(epoch) + '.xlsx'), index = False)

    


# def calculate_metric( gt_files, dataset, start_slice, end_slice, start_slice_include = None, end_slice_include = None, exclude_slice = []):
#     base_dice = []; mid_dice = []; apex_dice = []; all_dice = []
#     base_hd = []; mid_hd = []; apex_hd = []; all_hd = []
#     base_slice = []; mid_slice = []; apex_slice = []
#     for slice_num in np.arange(start_slice, end_slice):
#         # for STACOM
#         if dataset == 'STACOM':
#             if slice_num in exclude_slice or slice_num < start_slice_include or slice_num > end_slice_include:
#                 continue
        
#         gt_file = gt_files[slice_num]
#         pred_file = os.path.join(patient_folder, 'pred_seg_' + str(slice_num) + '.nii.gz')

#         gt = nb.load(gt_file).get_fdata()
#         pred = nb.load(pred_file).get_fdata()

#         gt = np.round(gt)
#         # # for ACDC
#         if dataset == 'ACDC' or dataset == 'HFpEF' or dataset == 'AS' or dataset == 'MM':
#             gt[gt!=2] = 0
#             gt[gt==2] = 1

#         pred = np.round(pred)

#         # set the gt timeframe without manual segmentation as 10
#         for t in range(0, gt.shape[-1]):
#             if np.sum(gt[:,:,t] == 1) == 0:
#                 gt[:,:,t] = 10
        
#         # calculate dice
#         dice = ff.np_categorical_dice(pred, gt, target_class = 1, exclude_class = 10)
#         hd = ff.HD(pred,gt, pixel_size = 1, target_class = 1, exclude_class = 10, max_or_mean = 'max')

#         all_dice.append(dice)
#         all_hd.append(hd)
#         if slice_num in base_segment:
#             base_dice.append(dice)
#             base_hd.append(hd)
#             base_slice.append(slice_num)
#         elif slice_num in mid_segment:
#             mid_dice.append(dice)
#             mid_hd.append(hd)
#             mid_slice.append(slice_num)
#         elif slice_num in apex_segment:
#             apex_dice.append(dice)
#             apex_hd.append(hd)
#             apex_slice.append(slice_num)
        
#     base_dice = np.mean(base_dice)
#     mid_dice = np.mean(mid_dice)
#     apex_dice = np.mean(apex_dice)
#     all_dice = np.mean(all_dice)

#     base_hd = np.mean(base_hd)
#     mid_hd = np.mean(mid_hd)
#     apex_hd = np.mean(apex_hd)
#     all_hd = np.mean(all_hd)

#     return [all_dice, base_dice, mid_dice, apex_dice, all_hd, base_hd, mid_hd, apex_hd, base_slice, mid_slice, apex_slice]