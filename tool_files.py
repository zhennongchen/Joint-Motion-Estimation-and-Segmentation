import Joint_motion_seg_estimate_CMR.Defaults as Defaults
import Joint_motion_seg_estimate_CMR.functions_collection as ff

import os
import numpy as np
import nibabel as nb
import pandas as pd
import shutil
import SimpleITK as sitk

defaults = Defaults.Parameters()


# delete files
patient_list = ff.find_all_target_files(['*'],os.path.join(defaults.sam_dir,'models/unet2D_LSTM_trial2_alldata/predicts_HFpEF'))
for patient in patient_list:
    print(patient)
    shutil.rmtree(patient)
