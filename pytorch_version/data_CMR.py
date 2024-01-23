#!/usr/bin/env python

import os

from torch.utils.data import Dataset, DataLoader

import sam_cmr.Defaults as Defaults
import sam_cmr.Build_list_zhennong.Build_list as Build_list
import sam_cmr.dataset.CMR.dataset_zhennong as dataset_zhennong

defaults = Defaults.Parameters()

#### just put zhennong's dataloader here

######### to run the code, you need to first in the terminal run . ./set_defaults.sh, then run this file
# the dataloader script is dataset/CMR/dataset_zhennong.py, you can run dataset_zhennong.ipynb for debugging
        
def build_data_CMR(args,dataset_name, train_batch_list, train_index_list, full_or_nonzero_slice, shuffle, augment_list, augment_frequency, return_arrays_or_dictionary = 'dictionary', sample_more_base = 0, sample_more_apex = 0):
    # based on our ACDC dataset
    # we select the patients either by batch_list or by index_list using the spreadsheet file ACDC_Patient_List_training_testing.xlsx
    # for batch (think it as the group /folds in cross validation), batch 0,1,2,3,4 are used for training/cross-validation and batch 5 is used for testing
    # for index, we have 0-100 for training and 100-150 for testing
    
    if dataset_name == 'ACDC':
        data_path = os.path.join(defaults.sam_dir,'data/ACDC_database/temporal')
        patient_list_file = os.path.join(defaults.sam_dir, 'data/Patient_list/ACDC_Patient_List_training_testing.xlsx')
        # print('patient_list_file: ',patient_list_file)

    # based on our STACOM dataset
    # we select the patients either by batch_list or by index_list using the spreadsheet file STACOM_Patient_List_training_testing.xlsx
    # It contains 100 cases, divided into 5 batches, each batch contains 20 cases
    if dataset_name == 'STACOM':
        data_path = os.path.join(defaults.sam_dir,'data/STACOM_database/temporal')
        patient_list_file = os.path.join(defaults.sam_dir, 'data/Patient_list/STACOM_Patient_List_training_testing.xlsx')
    
    ##### select train and valid dataset (using either batch_list or index_list, if not using one then set it to None)

    _,_,_,image_full_slice_file_list_train ,seg_full_slice_file_list_train ,image_nonzero_slice_file_list_train ,seg_nonzero_slice_file_list_train ,image_nonzero_slice_file_loose_list_train ,seg_nonzero_slice_file_loose_list_train, total_slice_num_list_train ,nonzero_slice_num_list_train, nonzero_slice_num_loose_list_train = Build_list.__build__(patient_list_file, batch_list = train_batch_list, index_list = train_index_list)
    # _,_,_,image_full_slice_file_list_valid ,seg_full_slice_file_list_valid ,image_nonzero_slice_file_list_valid ,seg_nonzero_slice_file_list_valid ,total_slice_num_list_valid ,nonzero_slice_num_list_valid = Build_list.__build__(patient_list_file, batch_list = valid_batch_list, index_list = valid_index_list)
    print('image_full_slice_file_list_train num: ',image_full_slice_file_list_train.shape, ', seg_full_slice_file_list_train num: ',seg_full_slice_file_list_train.shape)
    # print('image_full_slice_file_list_valid num: ',image_full_slice_file_list_valid.shape, ', seg_full_slice_file_list_valid num: ',seg_full_slice_file_list_valid.shape)
    
    # select whether you want the "full_slice" or "nonzero_slice" data, full_slice means all the slices in the image, nonzero_slice means only the slices with manual segmentation at both ED and ES

    if full_or_nonzero_slice[0:4] == 'full': # full
        image_file_list_train = image_full_slice_file_list_train
        seg_file_list_train = seg_full_slice_file_list_train
        total_slice_num_list_train = total_slice_num_list_train

        # image_file_list_valid = image_full_slice_file_list_valid
        # seg_file_list_valid = seg_full_slice_file_list_valid
        # total_slice_num_list_valid = total_slice_num_list_valid

    elif full_or_nonzero_slice[0:4] == 'nonz': # nonzero
        image_file_list_train = image_nonzero_slice_file_list_train
        seg_file_list_train = seg_nonzero_slice_file_list_train
        total_slice_num_list_train = nonzero_slice_num_list_train

        # image_file_list_valid = image_nonzero_slice_file_list_valid
        # seg_file_list_valid = seg_nonzero_slice_file_list_valid
        # total_slice_num_list_valid = nonzero_slice_num_list_valid
    
    elif full_or_nonzero_slice[0:4] == 'loos': # nonzero loose
        image_file_list_train = image_nonzero_slice_file_loose_list_train
        seg_file_list_train = seg_nonzero_slice_file_loose_list_train
        total_slice_num_list_train = nonzero_slice_num_loose_list_train

    if dataset_name == 'ACDC':
        relabel_LV = True
        only_myo = True
    elif dataset_name == 'STACOM':
        relabel_LV = False 
        only_myo = False
    
    # print('turn_zero_seg_slice_into: ',args.turn_zero_seg_slice_into)
    dataset_train = dataset_zhennong.Dataset_CMR(patient_list_file,
                                                 image_file_list_train,
                                                 seg_file_list_train,
                                                 total_slice_num_list_train,
                                                 return_arrays_or_dictionary = return_arrays_or_dictionary, # 'dictionary' or 'arrays'

                                                 seg_include_lowest_piexel = 100,
                                                 turn_zero_seg_slice_into = args.turn_zero_seg_slice_into,
                                                 
                                                 relabel_LV = relabel_LV,
                                                 only_myo = only_myo,
                                                 center_crop_according_to_which_class = [1],
                                                 image_shape = [defaults.x_dim, defaults.y_dim],
                                                 shuffle = shuffle, 
                                                 image_normalization = True,
                                                 augment_list = augment_list, # a list of augmentation methods and their range: v range = None for brightness, contrast, sharpness
                                                 augment_frequency = augment_frequency,
                                                 sample_more_base = sample_more_base,
                                                 sample_more_apex= sample_more_apex,)
    
    # valid_dataset = dataset_zhennong.Dataset_CMR(patient_list_file,
    #                                              image_file_list_valid,
    #                                              seg_file_list_valid,
    #                                              total_slice_num_list_valid,
    #                                              return_arrays_or_dictionary = 'dictionary', # 'dictionary' or 'arrays'

    #                                              seg_include_lowest_piexel = 100,
    #                                              turn_zero_seg_slice_into = args.turn_zero_seg_slice_into,
                                                 
    #                                              relabel_LV = relabel_LV,
    #                                              center_crop_according_to_which_class = [1],
    #                                              image_shape = [defaults.x_dim, defaults.y_dim],
    #                                              shuffle = False, 
    #                                              image_normalization = True,
    #                                              augment_list = [], # a list of augmentation methods and their range: v range = None for brightness, contrast, sharpness
    #                                              augment_frequency = 0)
    


    return dataset_train
  
