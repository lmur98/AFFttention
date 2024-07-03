import os
import sys
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch.backends.cudnn as cudnn
import torch.nn.init as init
import torch.utils.data
import torch.utils.data.sampler
import numpy as np
import argparse
import tqdm
import torchnet as tnt
import collections
from torch.utils.tensorboard import SummaryWriter

from similarity_dataset import Similarity_Frames_Dataset
from Siamesse_NN import SiameseR18_5MLP

cudnn.benchmark = True 


def save(epoch, iteration, model, optimizer, parallel):
    print('Saving state, iter:', iteration)
    state_dict = model.state_dict() if not parallel else model.module.state_dict()
    optim_state = optimizer.state_dict()
    checkpoint = {'net':state_dict, 'optimizer':optim_state, 'iter': iteration}
    torch.save(checkpoint, '/home/lmur/hum_obj_int/stillfast/extract_affordances/saved_models/' + 'Ego_4D_Siamese_R18_5MLP_ALL' + '_%d_%d.pth'%(epoch, iteration))

def batch_to_cuda(batch):
    for k in batch:
        batch[k] = batch[k].cuda()
    return batch

def train(iteration, trainloader, valloader, model, optimizer, train_writer, val_writer):
    model.train()
    max_iter = 20000
    total_iters = len(trainloader)
    epoch = iteration//total_iters
    plot_every = int(0.1*len(trainloader))
    loss_meters = collections.defaultdict(lambda: tnt.meter.MovingAverageValueMeter(20))

    while iteration <= max_iter:
        bar = tqdm.tqdm(trainloader, total=len(trainloader))
        for batch in trainloader:
            #----------Training loop----------#
            batch = batch_to_cuda(batch)
            pred, loss_dict = model(batch)
            loss_dict = {k:v.mean() for k,v in loss_dict.items()}
            loss = sum(loss_dict.values())
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            #----------Logging----------#
            _, pred_idx = pred.max(1)
            correct = (pred_idx==batch['label']).float().sum()
            batch_acc = correct/pred.shape[0]
            loss_meters['bAcc'].add(batch_acc.item())

            for k, v in loss_dict.items():
                loss_meters[k].add(v.item())
            loss_meters['total_loss'].add(loss.item())

            if iteration % 100 == 0:
                log_str = 'iter: %d (%d + %d/%d) | '%(iteration, epoch, iteration%total_iters, total_iters)
                log_str += ' | '.join(['%s: %.3f'%(k, v.value()[0]) for k,v in loss_meters.items()])
                print (log_str)

            if iteration % plot_every==0:
                for key in loss_meters:
                    train_writer.add_scalar('train/%s'%key, loss_meters[key].value()[0], int(100*iteration/total_iters))

            iteration += 1
            bar.update(1)
        
        epoch += 1

        if epoch % 1 == 0:
            with torch.no_grad():
                validate(epoch, iteration, valloader, model, optimizer, val_writer)

 
def validate(epoch, iteration, valloader, model, optimizer, val_writer):

    model.eval()

    correct, total = 0, 0
    loss_meters = collections.defaultdict(lambda: tnt.meter.AverageValueMeter())

    for num_passes in tqdm.tqdm(range(2)): # run over validation set 10 times

        for batch in tqdm.tqdm(valloader, total=len(valloader)):
            #--------------Validation loss--------------#
            batch = batch_to_cuda(batch)
            pred, loss_dict = model(batch)
            loss_dict = {k:v.mean() for k,v in loss_dict.items() if v.numel()>0}
            loss = sum(loss_dict.values())

            for k, v in loss_dict.items():
                loss_meters[k].add(v.item())
            loss_meters['total_loss'].add(loss.item())

            _, pred_idx = pred.max(1)
            correct += (pred_idx==batch['label']).float().sum()
            total += pred.size(0)

    accuracy = 1.0 * correct/total

    log_str = '(val) E: %d | iter: %d | A: %.3f | '%(epoch, iteration, accuracy)
    log_str += ' | '.join(['%s: %.3f'%(k, v.value()[0]) for k,v in loss_meters.items()])
    print (log_str)

    val_stats = '_L_%.3f_A_%.3f'%(loss_meters['total_loss'].value()[0], accuracy)
    save(epoch, iteration, model, optimizer, parallel = False)

    val_writer.add_scalar('val/loss', loss_meters['total_loss'].value()[0], epoch)
    val_writer.add_scalar('val/accuracy', accuracy, epoch)

    model.train()

#----------------------------------------------------------------------------------------------------------------------------------------#

if __name__=='__main__':
    load = False
    train_writer = SummaryWriter(log_dir = os.path.join('/home/lmur/hum_obj_int/stillfast/extract_affordances/logs_v2', 'train_all'))
    val_writer = SummaryWriter(log_dir = os.path.join('/home/lmur/hum_obj_int/stillfast/extract_affordances/logs_v2', 'val_all'))
    train_dataset = Similarity_Frames_Dataset('train')
    val_dataset = Similarity_Frames_Dataset('val')
    print('the len of the training is', len(train_dataset))
    print('the len of the validation is', len(val_dataset))
    
    trainloader = torch.utils.data.DataLoader(train_dataset, batch_size = 128, shuffle = True, num_workers = 32, pin_memory=True)
    valloader = torch.utils.data.DataLoader(val_dataset, batch_size = 128, shuffle = False, num_workers = 32, pin_memory=True)
    print('the len of the training loader is', len(trainloader))

    model = SiameseR18_5MLP()
    model.cuda()

    optim_params = list(filter(lambda p: p.requires_grad, model.parameters()))
    print ('Optimizing %d paramters'%len(optim_params))
    optimizer = optim.Adam(optim_params, lr = 2e-4, weight_decay = 1e-4)

    start_iter = 0
    if load:
        load_path = '...'
        checkpoint = torch.load(load_path, map_location='cpu')
        start_iter = checkpoint['iter']
        model.load_state_dict(checkpoint['net'])
        print ('Loaded checkpoint from ', load_path)


    train(start_iter, trainloader, valloader, model, optimizer, train_writer, val_writer)