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
main_folder = os.path.join(cg.sam_dir, 'models', 'sam_noweights_STACOM','predicts')
patients = ff.find_all_target_files(['*'], main_folder)

for i in range(0, len(patients)):
    # print(patients[i])
    patient = patients[i]
    folder = os.path.join(patient, 'epoch-135')

    pred_files = ff.find_all_target_files(['pred*'], folder)

    for j in range(0, len(pred_files)):

        pred = nb.load(pred_files[j]).get_fdata()

        new_pred, need_to_remove = ff.remove_scatter(pred,1)

        if need_to_remove ==True:
            print('patient:', patient, 'timeframe:', j, 'need to remove scatter')

            # nb.save(nb.Nifti1Image(new_pred, nb.load(pred_files[j]).affine), pred_files[j])
        save_folder = os.path.join(patient,os.path.basename(folder) + '-processed'); os.makedirs(save_folder, exist_ok = True)
        nb.save(nb.Nifti1Image(new_pred, nb.load(pred_files[j]).affine), os.path.join(save_folder, os.path.basename(pred_files[j])))

