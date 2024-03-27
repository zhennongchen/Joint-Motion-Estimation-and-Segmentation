import sys
sys.path.append('/workspace/Documents')
import os
import matplotlib.pylab as plt
import nibabel as nb
from skimage.measure import regionprops
import numpy as np
import shutil
import Joint_motion_seg_estimate_CMR.functions_collection as ff
import Joint_motion_seg_estimate_CMR.Defaults as Defaults

cg = Defaults.Parameters()

# remove scatter
main_folder = os.path.join(cg.sam_dir, 'models', 'unet3D_alldata','predicts_HFpEF')
patients = ff.find_all_target_files(['*'], main_folder)

for i in range(0, len(patients)):
    # print(patients[i])
    patient = patients[i]
    folder = os.path.join(patient, 'epoch-293')

    pred_files = ff.find_all_target_files(['pred*'], folder)

    for j in range(0, len(pred_files)):

        pred = nb.load(pred_files[j]).get_fdata()

        new_pred, need_to_remove = ff.remove_scatter(pred,1)

        if need_to_remove ==True:
            print('patient:', patient, 'timeframe:', j, 'need to remove scatter')

            # nb.save(nb.Nifti1Image(new_pred, nb.load(pred_files[j]).affine), pred_files[j])
        save_folder = os.path.join(patient,os.path.basename(folder) + '-processed'); os.makedirs(save_folder, exist_ok = True)
        nb.save(nb.Nifti1Image(new_pred, nb.load(pred_files[j]).affine), os.path.join(save_folder, os.path.basename(pred_files[j])))


# combine the results
# dataset = 'ACDC'
# main_folder = os.path.join(cg.sam_dir, 'models', 'STACOM_alldata','predicts')
# patients = ff.find_all_target_files(['*'], main_folder)

# epoch1 = 245
# epoch2 = 125
# epoch3 = 100

# for i in range(0, len(patients)):
#     print('patient:', os.path.basename(patients[i]))

#     patient = patients[i]
#     patient_id = os.path.basename(patient)

#     pred_file1_list = ff.sort_timeframe(ff.find_all_target_files(['pred_seg*'], os.path.join(patient, 'epoch-'+str(epoch1))), 2, '_')
#     pred_file2_list = ff.sort_timeframe(ff.find_all_target_files(['pred_seg*'], os.path.join(patient, 'epoch-'+str(epoch2))), 2, '_')
#     pred_file3_list = ff.sort_timeframe(ff.find_all_target_files(['pred_seg*'], os.path.join(patient, 'epoch-'+str(epoch3))), 2, '_')
    

#     gt_file_list = ff.sort_timeframe(ff.find_all_target_files(['original_seg*'], os.path.join(patient, 'epoch-'+str(epoch1))), 2, '_')
#     gt_img_list = ff.sort_timeframe(ff.find_all_target_files(['original_image*'], os.path.join(patient, 'epoch-'+str(epoch1))), 2, '_')

#     save_folder = os.path.join(patient, 'final3'); os.makedirs(save_folder, exist_ok = True)

#     for j in range(0, len(pred_file1_list)):

#         pred1 = nb.load(pred_file1_list[j]).get_fdata(); pred1 = np.round(pred1)
#         pred2 = nb.load(pred_file2_list[j]).get_fdata(); pred2 = np.round(pred2)
#         pred3 = nb.load(pred_file3_list[j]).get_fdata(); pred3 = np.round(pred3)

#         preds = [pred1, pred2, pred3]
        
#         gt = nb.load(gt_file_list[j]).get_fdata()

#         if dataset == 'ACDC':
#             gt[gt!=2] = 0
#             gt[gt==2] = 1

#         for t in range(0, gt.shape[-1]):
#             if np.sum(gt[:,:,t] == 1) == 0:
#                 gt[:,:,t] = 10
        
#         # calculate dice
#         dice1 = ff.np_categorical_dice(pred1, gt, target_class = 1, exclude_class = 10)
#         dice2 = ff.np_categorical_dice(pred2, gt, target_class = 1, exclude_class = 10)
#         dice3 = ff.np_categorical_dice(pred3, gt, target_class = 1, exclude_class = 10)

#         dices = [dice1, dice2, dice3]

#         # find the best dice index
#         best_dice_index = np.argmax(dices)
#         new_pred = np.copy(preds[best_dice_index])


#         # save the new prediction
#         nb.save(nb.Nifti1Image(new_pred, nb.load(pred_file1_list[j]).affine), os.path.join(save_folder, 'pred_seg_' + str(j) + '.nii.gz'))

#         # only copy ground truth image and gt
#         shutil.copy(gt_file_list[j], os.path.join(save_folder, os.path.basename(gt_file_list[j])))
#         shutil.copy(gt_img_list[j], os.path.join(save_folder, os.path.basename(gt_img_list[j])))
                    

        
        


       


