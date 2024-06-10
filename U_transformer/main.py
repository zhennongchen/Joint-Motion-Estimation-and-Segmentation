#!/usr/bin/env python
import torch
import torch.optim as optim
from torch.autograd import Variable
import torch.nn as nn
import numpy as np
from tqdm import tqdm
import random
from torch.utils.data import DataLoader
import scipy.io
import os
import pdb
import argparse
import pandas as pd
from einops import rearrange

from Joint_motion_seg_estimate_CMR.unet_2D.network import *
from Joint_motion_seg_estimate_CMR.data.data_CMR_2D import *
from Joint_motion_seg_estimate_CMR.unet_2D.train_engine import *
from Joint_motion_seg_estimate_CMR.unet_2D.validate_engine import *
import Joint_motion_seg_estimate_CMR.Defaults as Defaults
import Joint_motion_seg_estimate_CMR.functions_collection as ff


defaults = Defaults.Parameters()

def get_args_parser():
    parser = argparse.ArgumentParser('joint', add_help=True)
    parser.add_argument('--batch_size', default=1, type=int, help='Batch size per GPU (effective batch size is batch_size * accum_iter * # gpus')

    # Custom parser 
    parser.add_argument('--device', default='cuda', help='device to use for training / testing')
    parser.add_argument('--seed', default=1234, type=int)   
    
    
    ########## important parameters
    trial_name = 'unet2D_trial1'
    main_save_model = os.path.join(defaults.sam_dir, 'models', trial_name)
    pretrained_model_epoch = 64
    parser.add_argument('--output_dir', default = main_save_model, help='path where to save, empty for no saving')
    parser.add_argument('--pretrained_model_epoch', default = pretrained_model_epoch)


    # parser.add_argument('--pretrained_model', default = os.path.join(defaults.sam_dir, 'models', 'unet3D_trial1', 'models', 'model-200.pth'), help='path where to save, empty for no saving')
    if pretrained_model_epoch == None:
        parser.add_argument('--pretrained_model', default = None, help='path where to save, empty for no saving')
    else:
        parser.add_argument('--pretrained_model', default = os.path.join(main_save_model, 'models', 'model-%s.pth' % pretrained_model_epoch), help='path where to save, empty for no saving')

    parser.add_argument('--train_mode', default=False)
    parser.add_argument('--validation', default=True)
    parser.add_argument('--save_prediction', default=True)
    parser.add_argument('--freeze_encoder', default = False) 
    parser.add_argument('--loss_weight', default= [1,0.5]) # [ce_loss, dice_loss]

    if pretrained_model_epoch == None:
        parser.add_argument('--start_epoch', default=1, type=int, metavar='N', help='start epoch')
    else:
        parser.add_argument('--start_epoch', default=pretrained_model_epoch+1, type=int, metavar='N', help='start epoch')
    parser.add_argument('--epochs', default=2000, type=int)
    parser.add_argument('--save_model_file_every_N_epoch', default=2, type = int) 
    parser.add_argument('--lr', type=float, default=1e-4, metavar='LR')
    parser.add_argument('--lr_update_every_N_epoch', default=1000000, type = int) # fixed learning rate
    parser.add_argument('--lr_decay_gamma', default=0.95)
    parser.add_argument('--accum_iter', default=5, type=float)
    
    # Dataset parameters
    parser.add_argument('--img_size', default=128, type=int)    
    parser.add_argument('--num_classes', type=int, default=2)  ######## important!!!!

    parser.add_argument('--dataset_name', default='STACOM')
    parser.add_argument('--full_or_nonzero_slice', default='nonzero') # full means all the slices, nonzero means only the slices with manual segmentation at both ED and ES, loose means the slices with manual segmentation at either ED or ES or both
    parser.add_argument('--turn_zero_seg_slice_into', default=10, type=int)
    parser.add_argument('--augment_list', default=[('brightness' , None), ('contrast', None), ('sharpness', None), ('flip', None), ('rotate', [-20,20]), ('translate', [-5,5]), ('random_crop', [-5,5])], type=list)
    parser.add_argument('--augment_frequency', default=0.5, type=float)

    return parser


def run(args):
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # build some folders
    ff.make_folder([args.output_dir, os.path.join(args.output_dir, 'models'), os.path.join(args.output_dir, 'logs')])

    # Data loading code
    train_index_list = np.arange(0,60,1)  
    valid_index_list = np.arange(60,80,1) # just to monitor the validation loss, will not be used to select any hyperparameters
    train_batch_list = None
    valid_batch_list = None

    dataset_train = build_data_CMR_2D(args, args.dataset_name, 
                    train_batch_list,  train_index_list, full_or_nonzero_slice = args.full_or_nonzero_slice,
                    shuffle = True,
                    augment_list = args.augment_list, augment_frequency = args.augment_frequency,
                    return_arrays_or_dictionary = 'dictionary')
    
    dataset_valid = build_data_CMR_2D(args, args.dataset_name, 
                    valid_batch_list, valid_index_list, full_or_nonzero_slice = args.full_or_nonzero_slice,
                    shuffle = False,
                    augment_list = [], augment_frequency = -0.1,
                    return_arrays_or_dictionary = 'dictionary')

    data_loader_train = torch.utils.data.DataLoader(dataset_train, batch_size = 15, shuffle = False, pin_memory = True, num_workers = 0)# cpu_count()) 
    data_loader_valid = torch.utils.data.DataLoader(dataset_valid, batch_size = 15, shuffle = False, pin_memory = True, num_workers = 0)# cpu_count())

    # build model
    model = Unet2D(init_dim = 16,
        channels = 1,
        dim_mults = (2,4,8,16),
        num_classes = args.num_classes)

    """""""""""""""""""""""""""""""""""""""TRAINING"""""""""""""""""""""""""""""""""""""""
    if args.train_mode == True:
        model = model.to(device)
        optimizer = optim.Adam(model.parameters(),lr=args.lr)
        # load pretrained model
        if args.pretrained_model is not None:
            checkpoint = torch.load(args.pretrained_model)
            model.load_state_dict(checkpoint['model'])
            optimizer.load_state_dict(checkpoint['optimizer'])
            print('loaded pretrained model from: ', args.pretrained_model)
        else:
            print('new train')

        # train loop
        training_log = []
        valid_loss = np.inf; valid_ce_loss = np.inf; valid_dice_loss = np.inf
        
        for epoch in range(args.start_epoch, args.start_epoch + args.epochs):
            print('training epoch:', epoch)

            # update learning rate
            if epoch % args.lr_update_every_N_epoch == 0:
                optimizer.param_groups[0]['lr'] *= args.lr_decay_gamma
            print('learning rate now: ', optimizer.param_groups[0]['lr'])

            # train
            train_loss,  train_ce_loss, train_dice_loss, start_to_only_have_0 = train_loop(args, model, data_loader_train, optimizer)
            
            # on_epoch_end
            dataset_train.on_epoch_end()

            print('end of epoch: ', epoch, 'average loss: ', train_loss, 'ce_loss: ', train_ce_loss, 'dice_loss: ', train_dice_loss)

            # save model
            if epoch % args.save_model_file_every_N_epoch == 0:
                checkpoint_path = os.path.join(args.output_dir, 'models', 'model-%s.pth' % epoch)
                to_save = {'model': model.state_dict(),
                            'optimizer': optimizer.state_dict(),
                            'epoch': epoch,
                            'args': args,}
                torch.save(to_save, checkpoint_path)

            # validate
            if epoch % args.save_model_file_every_N_epoch == 0 and args.validation == True:
                valid_loss, valid_ce_loss, valid_dice_loss = valid_loop(args, model, data_loader_valid)
                print('validation loss: ', valid_loss, 'valid ce_loss: ', valid_ce_loss, 'valid dice_loss: ', valid_dice_loss)

            # save_log
            training_log.append([epoch, train_loss, train_ce_loss, train_dice_loss, optimizer.param_groups[0]['lr'], valid_loss, valid_ce_loss, valid_dice_loss, start_to_only_have_0])
            training_log_df = pd.DataFrame(training_log, columns = ['epoch', 'train_loss', 'train_ce_loss', 'train_dice_loss', 'lr', 'valid_loss', 'valid_ce_loss', 'valid_dice_loss', 'start_to_only_have_0'])
            training_log_df.to_excel(os.path.join(args.output_dir, 'logs', 'training_log.xlsx'), index = False)

    else:
        """""""""""""""""""""""""""""""""""""""INFERENCE"""""""""""""""""""""""""""""""""""""""
        pred_index_list = np.arange(60,100,1)

        with torch.no_grad():
            model = Unet2D(init_dim = 16,
                channels = 1,
                dim_mults = (2,4,8,16),
                num_classes = args.num_classes)
                
            model.to(device)
            pretrained_model = torch.load(args.pretrained_model)
            print('loaded pretrained model from: ', args.pretrained_model)
            model.load_state_dict(pretrained_model['model'])

            for k in range(0, len(pred_index_list)):
                pred_index_list1 = np.arange(pred_index_list[k], pred_index_list[k]+1,1)

                dataset_pred = build_data_CMR_2D(args, args.dataset_name,
                        None, pred_index_list1, 
                        full_or_nonzero_slice = args.full_or_nonzero_slice,
                        shuffle = False,
                        augment_list = [], augment_frequency = -0.1,
                        return_arrays_or_dictionary = 'dictionary')
        
        
                data_loader_pred = torch.utils.data.DataLoader(dataset_pred, batch_size = 1, shuffle = False, pin_memory = True, num_workers = 0)# cpu_count())

                for batch_idx, batch in enumerate(data_loader_pred, 1):
                    # if os.path.isfile(os.path.join(args.output_dir, 'predicts_raw', batch["patient_id"][0], 'epoch-' + str(args.pretrained_model_epoch), 'pred_seg_slice0_tf14.nii.gz')):
                    #     continue

                    # image
                    batch_image = batch['image']
                    image_input = torch.clone(batch_image).to("cuda")
                    output =  model(image_input)
                        
                    pred_save_2D(batch, output,args)

                print('done this patient')
                save_folder_sub = os.path.join(args.output_dir, 'predicts_raw', batch["patient_id"][0], 'epoch-' + str(args.pretrained_model_epoch))
                image_shape = nb.load(batch["image_nonzero_slice_file"][0]).get_fdata().shape
                slice_number = image_shape[2]

                patient_id = batch["patient_id"][0]
                new_save_folder = os.path.join(args.output_dir, 'predicts', patient_id, 'epoch-' + str(args.pretrained_model_epoch))
                ff.make_folder([os.path.dirname(os.path.dirname(new_save_folder)), os.path.dirname(new_save_folder), new_save_folder])
                

                for s in range(0, slice_number):
                    
                    pred_seg = np.zeros((image_shape[0], image_shape[1], 15))
                    for tf in range(0,15):
                        pred_file = os.path.join(save_folder_sub, 'pred_seg_slice' + str(s) + '_tf' + str(tf) + '.nii.gz')
                        p = nb.load(pred_file).get_fdata()
                        pred_seg[:,:,tf] = p
                    
                    original_image = nb.load(batch["image_nonzero_slice_file"][0]).get_fdata()[:,:,s,:]
                    original_seg = nb.load(batch["seg_nonzero_slice_file"][0]).get_fdata()[:,:,s,:]
                    affine = nb.load(batch["image_nonzero_slice_file"][0]).affine

                    # save the image and mask
                    nb.save(nb.Nifti1Image(original_image, affine), os.path.join(new_save_folder, 'original_image_' + str(s) + '.nii.gz'))
                    nb.save(nb.Nifti1Image(original_seg, affine), os.path.join(new_save_folder, 'original_seg_' + str(s) + '.nii.gz'))
                    nb.save(nb.Nifti1Image(pred_seg, affine), os.path.join(new_save_folder, 'pred_seg_' + str(s) + '.nii.gz'))
                        




                

if __name__ == '__main__':
    args = get_args_parser()
    args = args.parse_args()
    print(args)
    run(args)
