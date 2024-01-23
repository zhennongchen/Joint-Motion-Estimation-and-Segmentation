# System
import os

class Parameters():

  def __init__(self):
    
    # # Dimension of input, for training.
    # self.x_dim = int(os.environ['CG_INPUT_X'])
    # self.y_dim = int(os.environ['CG_INPUT_Y'])
    # self.tf_dim = int(os.environ['CG_INPUT_TF_NUM'])

    # # Seed for randomization.
    # self.seed = int(os.environ['CG_SEED'])

    # # classes:
    # self.num_classes = int(os.environ['CG_NUM_CLASSES'])

    # # folders
    # self.sam_dir = os.environ['CG_SAM_DIR']
    # self.HFpEF_dir = os.environ['CG_HFpEF_DIR']
    # Dimension of input, for training.
    self.x_dim = 128
    self.y_dim = 128
    self.tf_dim = 10

    # Seed for randomization.
    self.seed = 8

    # classes:
    self.num_classes = 3

    # folders
    self.sam_dir ="/mnt/camca_NAS/SAM_for_CMR/"
    self.HFpEF_dir = "/mnt/camca_NAS/HFpEF/data/HFpEF_data/"