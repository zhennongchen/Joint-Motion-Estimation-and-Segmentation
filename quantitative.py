import os
import numpy as np
import pandas as pd
from tqdm import tqdm
import random
from pathlib import Path
import nibabel as nb

import Joint_motion_seg_estimate_CMR.Defaults as Defaults
import Joint_motion_seg_estimate_CMR.Build_list_zhennong.Build_list as Build_list
import Joint_motion_seg_estimate_CMR.functions_collection as ff

def nan_cal(x):
    # x is an array, remove the nan from the array
    new_array = []
    for item in x:
        if item == 'nan':
            item = np.nan
        else:
            item = float(item)
        if not np.isnan(item):
            new_array.append(item)
    x = np.array(new_array)
    return np.mean(x), np.std(x)


def calculate_metric( gt_files, dataset, start_slice, end_slice):
    base_dice = []; mid_dice = []; apex_dice = []; all_dice = []
    base_hd = []; mid_hd = []; apex_hd = []; all_hd = []
    base_slice = []; mid_slice = []; apex_slice = []
    for slice_num in np.arange(start_slice, end_slice):
        # print('slice_num: ', slice_num)
        # for STACOM
        if dataset == 'STACOM':
            if slice_num in exclude_slice or slice_num < start_slice or slice_num > end_slice:
                # print('slice excluded')
                continue
        
        gt_file = gt_files[slice_num]
        pred_file = os.path.join(patient_folder, 'pred_seg_' + str(slice_num) + '.nii.gz')
        pred_file2 = os.path.join(main_folder, patient_id, 'epoch-' + str(epoch) + '-processed', 'pred_seg_' + str(slice_num) + '.nii.gz')

        gt = nb.load(gt_file).get_fdata()
        pred = np.round(nb.load(pred_file).get_fdata())
        pred2 = np.round(nb.load(pred_file2).get_fdata())  # for HD calculation

        gt = np.round(gt)
        # # for ACDC
        if dataset == 'ACDC' or dataset == 'HFpEF':
            gt[gt!=2] = 0
            gt[gt==2] = 1
        # set the gt timeframe without manual segmentation as 10
        for t in range(0, gt.shape[-1]):
            if np.sum(gt[:,:,t] == 1) == 0:
                gt[:,:,t] = 10
        
        # calculate dice
        dice = ff.np_categorical_dice(pred, gt, target_class = 1, exclude_class = 10)
        hd = ff.HD(pred2, gt, pixel_size = 1, target_class = 1, exclude_class = 10, max_or_mean = 'max')

        all_dice.append(dice)
        all_hd.append(hd)
        if slice_num in base_segment:
            base_dice.append(dice)
            base_hd.append(hd)
            base_slice.append(slice_num)
        elif slice_num in mid_segment:
            mid_dice.append(dice)
            mid_hd.append(hd)
            mid_slice.append(slice_num)
        elif slice_num in apex_segment:
            apex_dice.append(dice)
            apex_hd.append(hd)
            apex_slice.append(slice_num)
        
    base_dice = np.mean(base_dice)
    mid_dice = np.mean(mid_dice)
    apex_dice = np.mean(apex_dice)
    all_dice = np.mean(all_dice)

    base_hd = np.mean(base_hd)
    mid_hd = np.mean(mid_hd)
    apex_hd = np.mean(apex_hd)
    all_hd = np.mean(all_hd)

    return [all_dice, base_dice, mid_dice, apex_dice, all_hd, base_hd, mid_hd, apex_hd, base_slice, mid_slice, apex_slice]
        

defaults = Defaults.Parameters() 

dataset = 'HFpEF'

# define patient list
patient_list_file = os.path.join(defaults.sam_dir, 'data/Patient_list/HFpEF_Patient_List_training_testing.xlsx')
index_list = np.arange(0,53,1)
patient_id_list,_,_,_ ,_,_,_ ,_ ,_, _ ,_, _ = Build_list.__build__(patient_list_file, batch_list = None, index_list = index_list)

main_folder = os.path.join(defaults.sam_dir, 'models/unet3D_alldata/predicts_HFpEF')
epoch = 293

# slice inclusion in the calculation
if dataset == 'STACOM':
    infosheet = pd.read_excel(os.path.join(os.path.dirname(patient_list_file),'STACOM_test_cohort_slice_inclusion.xlsx'))

# calculate dice
result = []
for i in range(0,len(patient_id_list)):
    patient_id = patient_id_list[i]
    patient_folder = os.path.join(main_folder, patient_id, 'epoch-' + str(epoch))
    if isinstance(epoch, str):
        patient_folder = os.path.join(main_folder, patient_id, epoch)

    print('patient_id: ', patient_id, patient_folder)

    # find how many slices
    gt_files = ff.sort_timeframe(ff.find_all_target_files(['original_seg*'], patient_folder),2,'_','.')
    # print(gt_files, len(gt_files))

    # divide into three even sets
    base_segment, mid_segment, apex_segment = ff.define_three_segments(len(gt_files))
    print(base_segment, mid_segment, apex_segment)

    # which slices should be included or excluded
    # for STACOM
    if dataset == 'STACOM':
        row = infosheet[infosheet['patient_id'] == patient_id]
        if len(row) == 0:
            print('patient not found in the infosheet')
            start_slice = 1; end_slice = len(gt_files) - 1; exclude_slice = []
        else:
            start_slice = int(row['start'].values[0])
            end_slice = int(row['end'].values[0])
            exclude_slice = row['exclude'].values[0]
            if isinstance(exclude_slice, str):
                exclude_slice = [int(item) for item in exclude_slice.split(',')]
            else:
                if np.isnan(exclude_slice):
                    exclude_slice = []
                elif isinstance(exclude_slice, int):
                    exclude_slice = [exclude_slice]

    # calculate metric
    if dataset != 'HFpEF':
        metrics = calculate_metric(gt_files, dataset, 1 , len(gt_files) - 1)
    else:
        metrics1 = calculate_metric(gt_files, dataset, 0, len(gt_files) - 1)
        metrics2 = calculate_metric(gt_files, dataset, 0, len(gt_files))
        # compare base and apex dice, which one is higher?
        end = [len(gt_files) - 1 if (metrics1[3] >= metrics2[3] and metrics1[0] >= metrics2[0]) else len(gt_files)][0]
      
        metrics = calculate_metric(gt_files, dataset, 0, end)
    all_dice, base_dice, mid_dice, apex_dice, all_hd, base_hd, mid_hd, apex_hd, base_slice, mid_slice, apex_slice = metrics

    print(base_slice, mid_slice, apex_slice)
    print(all_dice, base_dice, mid_dice, apex_dice, all_hd, base_hd, mid_hd, apex_hd)
    result.append([patient_id, all_dice, base_dice, mid_dice, apex_dice, all_hd, base_hd, mid_hd, apex_hd, base_slice, mid_slice, apex_slice])

# add one row for mean
a = np.array(result)
all_dice_mean = nan_cal(a[:,1])[0]
base_dice_mean = nan_cal(a[:,2])[0]
mid_dice_mean = nan_cal(a[:,3])[0]
apex_dice_mean = nan_cal(a[:,4])[0]
all_hd_mean = nan_cal(a[:,5])[0]
base_hd_mean = nan_cal(a[:,6])[0]
mid_hd_mean = nan_cal(a[:,7])[0]
apex_hd_mean = nan_cal(a[:,8])[0]
result.append(['mean', all_dice_mean, base_dice_mean, mid_dice_mean, apex_dice_mean, all_hd_mean, base_hd_mean, mid_hd_mean, apex_hd_mean])

# add one row for std
all_dice_std = nan_cal(a[:,1])[1]
base_dice_std = nan_cal(a[:,2])[1]
mid_dice_std = nan_cal(a[:,3])[1]
apex_dice_std = nan_cal(a[:,4])[1]
all_hd_std = nan_cal(a[:,5])[1]
base_hd_std = nan_cal(a[:,6])[1]
mid_hd_std = nan_cal(a[:,7])[1]
apex_hd_std = nan_cal(a[:,8])[1]
result.append(['std', all_dice_std, base_dice_std, mid_dice_std, apex_dice_std, all_hd_std, base_hd_std, mid_hd_std, apex_hd_std])


ff.make_folder([os.path.join(os.path.dirname(main_folder),'results')])
df = pd.DataFrame(result, columns = ['patient_id', 'all_dice', 'base_dice', 'mid_dice', 'apex_dice', 'all_hd', 'base_hd', 'mid_hd', 'apex_hd', 'base_segment', 'mid_segment', 'apex_segment'])
df.to_excel(os.path.join(os.path.dirname(main_folder),'results', 'HFpEF_test_epoch_' + str(epoch) + '.xlsx'))


