## to run this in terminal, type:
# chmod +x set_defaults.sh
# . ./set_defaults.sh   

## parameters
# define GPU you use
export CUDA_VISIBLE_DEVICES="0"

# volume dimension
export CG_INPUT_X=128 
export CG_INPUT_Y=128 
export CG_INPUT_TF_NUM=10 # always 10

# set number of classes
export CG_NUM_CLASSES=3 #0 for background, 1 for LV, 2 for MYO

# set random seed
export CG_SEED=8

# folders for Zhennong's dataset (change based on your folder paths)
export CG_SAM_DIR="/mnt/camca_NAS/SAM_for_CMR/"
export CG_HFpEF_DIR="/mnt/camca_NAS/HFpEF/data/HFpEF_data/"

