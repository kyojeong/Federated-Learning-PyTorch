












#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Python version: 3.6


import os
import copy
import time
import pickle
import numpy as np
from tqdm import tqdm
import random

import torch
from tensorboardX import SummaryWriter

from options import args_parser
from update import LocalUpdate, test_inference
from models import MLP, CNNMnist, CNNFashion_Mnist, CNNCifar
from utils import get_dataset, average_weights, exp_details


train_loss, train_accuracy = [[],[],[]], [[],[],[]]

def train_model(index,badnum):
        # BUILD MODEL
        if args.model == 'cnn':
            # Convolutional neural netork
            if args.dataset == 'mnist':
                global_model = CNNMnist(args=args)
            elif args.dataset == 'fmnist':
                global_model = CNNFashion_Mnist(args=args)
            elif args.dataset == 'cifar':
                global_model = CNNCifar(args=args)
    
        elif args.model == 'mlp':
            # Multi-layer preceptron
            img_size = train_dataset[0][0].shape
            len_in = 1
            for x in img_size:
                len_in *= x
                global_model = MLP(dim_in=len_in, dim_hidden=64,
                                   dim_out=args.num_classes)
        else:
            exit('Error: unrecognized model')
    
        # Set the model to train and send it to device.
        global_model.to(device)
        global_model.train()
        print(global_model)
    
        # copy weights
        global_weights = global_model.state_dict()
    
       
         # Training
         
        val_acc_list, net_list = [], []
        cv_loss, cv_acc = [], []
        print_every = 2
        val_loss_pre, counter = 0, 0
        
        for epoch in tqdm(range(args.epochs)):
            local_weights, local_losses = [], []
            print(f'\n | Global Training Round : {epoch+1} |\n')
    
            global_model.train()
            #choose client number
            m = max(int(args.frac * args.num_users), 1)
            print('\nusers choice number: {}'.format(m))
            
            #random to choose the clients
            idxs_users = np.random.choice(range(args.num_users), m, replace=False)
                
            
            print("choose users number: {}\n".format(idxs_users))
            
            
            for idx in idxs_users:
                local_model = LocalUpdate(args=args, dataset=train_dataset,idxs=user_groups[idx], logger=logger)
                w, loss = local_model.update_weights(
                    model=copy.deepcopy(global_model), global_round=epoch,idxnum=idx,badnum=badnum)
                #print('loss: {}\n'.format(loss))
                 
                local_weights.append(copy.deepcopy(w))
                #print(local_weights[0]['conv1.weight'])
                local_losses.append(copy.deepcopy(loss))
               
    
            # average local weights
            global_weights = average_weights(local_weights)
            #print(global_weights['conv1.weight'])
    
            # update global weights
            global_model.load_state_dict(global_weights)
    
            loss_avg = sum(local_losses) / len(local_losses)
            train_loss[index].append(loss_avg)
    
            # Calculate avg training accuracy over all users at every epoch
            list_acc, list_loss = [], []
            global_model.eval()
            for c in range(args.num_users):
                local_model = LocalUpdate(args=args, dataset=train_dataset,
                                          idxs=user_groups[idx], logger=logger)
                acc, loss = local_model.inference(model=global_model)
                list_acc.append(acc)
                list_loss.append(loss)
            train_accuracy[index].append(sum(list_acc)/len(list_acc))
    
        '''
            # print global training loss after every 'i' rounds
            print(f' \nAvg Training Stats after {epoch+1} global rounds:')
            print(f'Training Loss : {np.mean(np.array(train_loss[0]))}')
            print('Train Accuracy: {:.2f}% \n'.format(100*train_accuracy[index][-1]))
        '''
            
        # Test inference after completion of training
        test_acc, test_loss = test_inference(args, global_model, test_dataset)
    
        print(f' \n Results after {args.epochs} global rounds of training:')
        print("|---- Avg Train Accuracy: {:.2f}%".format(100*train_accuracy[index][-1]))
        print("|---- Test Accuracy: {:.2f}%".format(100*test_acc))
        
        # Saving the objects train_loss and train_accuracy:
        print_name = '\ndataset: {}\nmodel: {}\nEpochs: {}\nfraction(C): {}\niid: {}\nlocal_epochs(E): {}\nlocal_batchsize(B): {}\n'.\
        format(args.dataset, args.model, args.epochs, args.frac, args.iid,args.local_ep, args.local_bs)
        
    
       
       
        print(print_name)
    

if __name__ == '__main__':
    start_time = time.time()

    # define paths
    path_project = os.path.abspath('..')
    logger = SummaryWriter('../logs')

    args = args_parser()
    exp_details(args)

    #if args.gpu_id:
     #   torch.cuda.set_device(args.gpu_id)
    device = 'cuda' if args.gpu else 'cpu'

    # load dataset and user groups
    train_dataset, test_dataset, user_groups = get_dataset(args)
    
    

    train_model(0,0)
    train_model(1,20)
    train_model(2,50)

    print(f' \n Results after {args.epochs} global rounds of training contrary:')
    print("|---- Avg Train Accuracy 1: {:.2f}%".format(100*train_accuracy[0][-1]))
    print("|---- Avg Train Accuracy 2: {:.2f}%".format(100*train_accuracy[1][-1]))
    print("|---- Avg Train Accuracy 3: {:.2f}%".format(100*train_accuracy[2][-1]))



    print('\n Total Run Time: {0:0.4f}'.format(time.time()-start_time))

    #PLOTTING (optional)
    import matplotlib
    import matplotlib.pyplot as plt
    matplotlib.use('Agg')

    #Plot Loss curve
    plt.figure()
    plt.title('Training Loss vs Communication rounds')
    plt.plot(range(len(train_loss[0])), train_loss[0], color='r',label='no bad')
    plt.plot(range(len(train_loss[1])), train_loss[1], color='g',label='bad frac 0.2')
    plt.plot(range(len(train_loss[2])), train_loss[2], color='b',label='bad frac 0.5')
    plt.ylabel('Training loss')
    plt.xlabel('Communication Rounds')
    plt.savefig('test_loss.png')
                
    
    # Plot Average Accuracy vs Communication rounds
    plt.figure()
    plt.title('Average Accuracy vs Communication rounds')
    plt.plot(range(len(train_accuracy[0])), train_accuracy[0], color='r',label='no bad')
    plt.plot(range(len(train_accuracy[1])), train_accuracy[1], color='g',label='bad frac 0.2')
    plt.plot(range(len(train_accuracy[2])), train_accuracy[2], color='b',label='bad frac 0.5')
    plt.ylabel('Average Accuracy')
    plt.xlabel('Communication Rounds')
    plt.savefig('test_acc.png')
               
