import os
import numpy as np
import pandas as pd
from tqdm import tqdm
import random
from pathlib import Path
import nibabel as nb

import sam_cmr.Defaults as Defaults
import sam_cmr.Build_list_zhennong.Build_list as Build_list
from sam_cmr.dataset.data_CMR import build_data_CMR
import sam_cmr.functions_collection as ff

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

defaults = Defaults.Parameters()

dataset = 'STACOM'

# define patient list
patient_list_file = os.path.join(defaults.sam_dir, 'data/Patient_list/STACOM_Patient_List_training_testing.xlsx')
index_list = np.arange(60,100,1)
patient_id_list,_,_,_ ,_,_,_ ,_ ,_, _ ,_, _ = Build_list.__build__(patient_list_file, batch_list = None, index_list = index_list)

main_folder = os.path.join(defaults.sam_dir, 'models/joint_trial1/predicts')
epoch = 290

# slice inclusion in the calculation
if dataset == 'STACOM':
    infosheet = pd.read_excel(os.path.join(os.path.dirname(patient_list_file),'STACOM_test_cohort_slice_inclusion.xlsx'))

# calculate dice
result = []
for i in range(0,len(patient_id_list)):
    patient_id = patient_id_list[i]
    print('patient_id: ', patient_id)
    patient_folder = os.path.join(main_folder, patient_id, 'epoch-' + str(epoch))

    # find how many slices
    gt_files = ff.sort_timeframe(ff.find_all_target_files(['original_seg*'], patient_folder),2,'_','.')
    # print(gt_files, len(gt_files))

    # divide into three even sets
    arr = np.arange(0, len(gt_files))

    # Calculate the lengths of the segments
    # Ensuring the middle segment has the most elements if the array can't be equally divided
    segment_length = len(gt_files) // 3
    middle_segment_extra = len(gt_files) % 3

    # Create the segments
    base_segment = arr[:segment_length]
    mid_segment = arr[segment_length:2 * segment_length + middle_segment_extra]
    apex_segment = arr[2 * segment_length + middle_segment_extra:]
    print(base_segment, mid_segment, apex_segment)

    # which slices should be included or excluded
    # for STACOM
    if dataset == 'STACOM':
        row = infosheet[infosheet['patient_id'] == patient_id]
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
    

    base_dice = []; mid_dice = []; apex_dice = []; all_dice = []

    for slice_num in np.arange(1, len(gt_files) - 1):
        # print('slice_num: ', slice_num)
        # for STACOM
        if dataset == 'STACOM':
            if slice_num in exclude_slice or slice_num < start_slice or slice_num > end_slice:
                # print('slice excluded')
                continue
        
        gt_file = gt_files[slice_num]
        pred_file = os.path.join(patient_folder, 'pred_seg_' + str(slice_num) + '.nii.gz')

        gt = nb.load(gt_file).get_fdata()
        pred = nb.load(pred_file).get_fdata()

        gt = np.round(gt)
        # # for ACDC
        if dataset == 'ACDC':
            gt[gt!=2] = 0
            gt[gt==2] = 1

        pred = np.round(pred)

        # set the gt timeframe without manual segmentation as 10
        for t in range(0, gt.shape[-1]):
            if np.sum(gt[:,:,t] == 1) == 0:
                gt[:,:,t] = 10
        
        # calculate dice
        dice = ff.np_categorical_dice(pred, gt, target_class = 1, exclude_class = 10)

        all_dice.append(dice)
        if slice_num in base_segment:
            base_dice.append(dice)
        elif slice_num in mid_segment:
            mid_dice.append(dice)
        elif slice_num in apex_segment:
            apex_dice.append(dice)
        
    base_dice = np.mean(base_dice)
    mid_dice = np.mean(mid_dice)
    apex_dice = np.mean(apex_dice)
    all_dice = np.mean(all_dice)

    print(all_dice, base_dice, mid_dice, apex_dice)
    result.append([patient_id, all_dice, base_dice, mid_dice, apex_dice])

# add one row for mean
a = np.array(result)
all_dice_mean = nan_cal(a[:,1])[0]
base_dice_mean = nan_cal(a[:,2])[0]
mid_dice_mean = nan_cal(a[:,3])[0]
apex_dice_mean = nan_cal(a[:,4])[0]
result.append(['mean', all_dice_mean, base_dice_mean, mid_dice_mean, apex_dice_mean])

# add one row for std
all_dice_std = nan_cal(a[:,1])[1]
base_dice_std = nan_cal(a[:,2])[1]
mid_dice_std = nan_cal(a[:,3])[1]
apex_dice_std = nan_cal(a[:,4])[1]
result.append(['std', all_dice_std, base_dice_std, mid_dice_std, apex_dice_std])


ff.make_folder([os.path.join(os.path.dirname(main_folder),'results')])
df = pd.DataFrame(result, columns = ['patient_id', 'all_dice', 'base_dice', 'mid_dice', 'apex_dice'])
df.to_excel(os.path.join(os.path.dirname(main_folder),'results', 'dice_test_epoch_' + str(epoch) + '.xlsx'))





        

    

