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

from Joint_motion_seg_estimate_CMR.unet_3D.network import *
from Joint_motion_seg_estimate_CMR.data.data_CMR import *
from Joint_motion_seg_estimate_CMR.unet_3D.train_engine import *
from Joint_motion_seg_estimate_CMR.unet_3D.validate_engine import *
import Joint_motion_seg_estimate_CMR.Defaults as Defaults
import Joint_motion_seg_estimate_CMR.functions_collection as ff


defaults = Defaults.Parameters()

def get_args_parser():
    parser = argparse.ArgumentParser('joint', add_help=True)
    parser.add_argument('--batch_size', default=1, type=int, help='Batch size per GPU (effective batch size is batch_size * accum_iter * # gpus')

    # Custom parser 
    parser.add_argument('--device', default='cuda', help='device to use for training / testing')
    parser.add_argument('--seed', default=1100, type=int)   
    
    ########## important parameters
    trial_name = 'unet3D_alldata'
    main_save_model = os.path.join(defaults.sam_dir, 'models', trial_name)
    pretrained_model_epoch = 293
    parser.add_argument('--output_dir', default = main_save_model, help='path where to save, empty for no saving')
    parser.add_argument('--pretrained_model_epoch', default = pretrained_model_epoch)

    # parser.add_argument('--pretrained_model', default = os.path.join(defaults.sam_dir, 'models', 'unet3D_alldata', 'models', 'model-280.pth'), help='path where to save, empty for no saving')
    if pretrained_model_epoch == None:
        parser.add_argument('--pretrained_model', default = None, help='path where to save, empty for no saving')
    else:
        parser.add_argument('--pretrained_model', default = os.path.join(main_save_model, 'models', 'model-%s.pth' % pretrained_model_epoch), help='path where to save, empty for no saving')

    parser.add_argument('--train_mode', default=False)
    parser.add_argument('--validation', default=True)
    parser.add_argument('--save_prediction', default=True)
    parser.add_argument('--freeze_encoder', default = False) 
    parser.add_argument('--loss_weight', default= [0,1]) # [ce_loss, dice_loss]

    if pretrained_model_epoch == None:
        parser.add_argument('--start_epoch', default=1, type=int, metavar='N', help='start epoch')
    else:
        parser.add_argument('--start_epoch', default=pretrained_model_epoch+1, type=int, metavar='N', help='start epoch')
    parser.add_argument('--epochs', default= 100, type=int)
    parser.add_argument('--save_model_file_every_N_epoch', default=1, type = int) 
    parser.add_argument('--lr', type=float, default=1e-5, metavar='LR')
    parser.add_argument('--lr_update_every_N_epoch', default=1000000, type = int) # fixed learning rate
    parser.add_argument('--lr_decay_gamma', default=0.95)
    parser.add_argument('--accum_iter', default=5, type=float)
    
    # Dataset parameters
    parser.add_argument('--dataset_names', default=[['STACOM', 'sax'], ['ACDC', 'sax'], ['HFpEF', 'sax'] ], type=list)
    parser.add_argument('--dataset_split',default=[[np.arange(0,100,1) , np.arange(0,0,1)], [np.arange(0,100,1) , np.arange(100,150,1)], [np.arange(0,0,1) , np.arange(0,0,1)]], type=list) # [training_data, validation_data]. for LAX: 0-60 case: 0-224, 60-80: 224-297, 80-100: 297-376
    parser.add_argument('--dataset_train', default= [], type = list)
    parser.add_argument('--dataset_valid', default= [], type = list)

    parser.add_argument('--img_size', default=128, type=int)    
    parser.add_argument('--num_classes', type=int, default=2)  ######## important!!!!
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

    # build model
    model = Unet3D(init_dim = 16,
        channels = 1,
        dim_mults = (2,4,8,16),
        num_classes = args.num_classes)

    """""""""""""""""""""""""""""""""""""""TRAINING"""""""""""""""""""""""""""""""""""""""
    if args.train_mode == True:
        """Load Data from different sources"""
        assert len(args.dataset_names) == len(args.dataset_split), "The length of dataset_names and dataset_split should be the same"
        # define the dataset list for train and validation, separately
        for i in range(len(args.dataset_split)):
            if args.dataset_split[i][0].shape[0] > 0:
                args.dataset_train.append([args.dataset_names[i][0], args.dataset_names[i][1], args.dataset_split[i][0]])
            if args.dataset_split[i][1].shape[0] > 0:
                args.dataset_valid.append([args.dataset_names[i][0], args.dataset_names[i][1], args.dataset_split[i][1]])
        print('now print the dataset_train and dataset_valid')
        print(args.dataset_train)
        print(args.dataset_valid)

        dataset_train = []
        dataset_valid = []
        # do trianing
        for i in range(len(args.dataset_train)):
            current_dataset_name = args.dataset_train[i][0]
            current_slice_type = args.dataset_train[i][1]
            current_index_list = args.dataset_train[i][2]
            dataset_train.append(build_data_CMR(args, current_dataset_name, 
                        None,  current_index_list, 
                        full_or_nonzero_slice = args.full_or_nonzero_slice,
                        shuffle = True,
                        augment_list = args.augment_list, augment_frequency = args.augment_frequency,
                        return_arrays_or_dictionary = 'dictionary', ))
        # do validation
        for i in range(len(args.dataset_valid)):
            current_dataset_name = args.dataset_valid[i][0]
            current_slice_type = args.dataset_valid[i][1]
            current_index_list = args.dataset_valid[i][2]
            dataset_valid.append(build_data_CMR(args, current_dataset_name,
                        None, current_index_list, 
                        full_or_nonzero_slice = args.full_or_nonzero_slice,
                        shuffle = False,
                        augment_list = [], augment_frequency =-0.1,
                        return_arrays_or_dictionary = 'dictionary',))
            
        '''Set up data loader for training and validation set'''
        data_loader_train = []
        data_loader_valid = []
        for i in range(len(dataset_train)):
            data_loader_train.append(torch.utils.data.DataLoader(dataset_train[i], batch_size = 1, shuffle = False, pin_memory = True, num_workers = 0))
        for i in range(len(dataset_valid)):
            data_loader_valid.append(torch.utils.data.DataLoader(dataset_valid[i], batch_size = 1, shuffle = False, pin_memory = True, num_workers = 0))

        # freeze the encoder part
        freeze_list = ["conv_blocks"]
        freeze_keys = []
        if args.freeze_encoder == True:
            for n, value in model.named_parameters():
                if any(freeze_name in n for freeze_name in freeze_list):
                    value.requires_grad = False
                    freeze_keys.append(n)
            else:
                value.requires_grad = True
        else:
            for p in model.parameters():
                p.requires_grad = True
        print('freeze_keys: ', freeze_keys)
        

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
        valid_results = [np.inf] * 3
        
        for epoch in range(args.start_epoch, args.start_epoch + args.epochs):
            print('training epoch:', epoch)

            # update learning rate
            if epoch % args.lr_update_every_N_epoch == 0:
                optimizer.param_groups[0]['lr'] *= args.lr_decay_gamma
            optimizer.param_groups[0]['lr'] = args.lr
            print('learning rate now: ', optimizer.param_groups[0]['lr'])

            # train
            train_results = train_loop(args, model, data_loader_train, optimizer)

            # on_epoch_end:
            for k in range(len(dataset_train)):
               dataset_train[k].on_epoch_end()
            print('end of epoch: ', epoch, 'average loss: ', train_results[0], 'ce_loss: ', train_results[1], 'dice_loss: ', train_results[2])

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
                valid_results = valid_loop(args, model, data_loader_valid)
                print('validation loss: ', valid_results[0], 'ce_loss: ', valid_results[1], 'dice_loss: ', valid_results[2])

            # save_log
            training_log.append([epoch, train_results[0], train_results[1], train_results[2], optimizer.param_groups[0]['lr'], valid_results[0], valid_results[1], valid_results[2]])
            training_log_df = pd.DataFrame(training_log, columns = ['epoch', 'train_loss', 'train_ce_loss', 'train_dice_loss', 'lr', 'valid_loss', 'valid_ce_loss', 'valid_dice_loss'])
            training_log_df.to_excel(os.path.join(args.output_dir, 'logs', 'training_log.xlsx'), index = False)

    else:
        """""""""""""""""""""""""""""""""""""""INFERENCE"""""""""""""""""""""""""""""""""""""""
        pred_index_list = np.arange(0,53,1)
        pred_batch_list = None
        
        dataset_pred = build_data_CMR(args, 'HFpEF',
                    pred_batch_list, pred_index_list, full_or_nonzero_slice = args.full_or_nonzero_slice,
                    shuffle = False,
                    augment_list = [], augment_frequency = -0.1,
                    return_arrays_or_dictionary = 'dictionary')
        
        
        data_loader_pred = torch.utils.data.DataLoader(dataset_pred, batch_size = 1, shuffle = False, pin_memory = True, num_workers = 0)# cpu_count())

        # build model

        with torch.no_grad():
           model = Unet3D(init_dim = 16,
        channels = 1,
        dim_mults = (2,4,8,16),
        num_classes = args.num_classes)
           model.to(device)
           pretrained_model = torch.load(args.pretrained_model)
           print('loaded pretrained model from: ', args.pretrained_model)
           model.load_state_dict(pretrained_model['model'])

           for batch_idx, batch in enumerate(data_loader_pred, 1):
                # image
                batch_image = batch['image']
                image_input = torch.clone(batch_image).to("cuda")

                output =  model(image_input)

                save_folder = os.path.join(args.output_dir, 'predicts_HFpEF');ff.make_folder([save_folder])
                pred_save(batch, output,args, save_folder)
               

if __name__ == '__main__':
    args = get_args_parser()
    args = args.parse_args()
    run(args)
