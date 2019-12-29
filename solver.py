import os
import time
import librosa
import datetime
import numpy as np

import torch
import torch.optim as optim
from torch.utils.data import DataLoader

from tools import report



def compute_n_parameters(model):
    # compute only trainable params
    model_parameters = filter(lambda p: p.requires_grad, model.parameters())
    n_params = sum([np.prod(p.size()) for p in model_parameters])
    return n_params


def train(args, model, loss_func, dataset):
    print('\n - {training phase} - ')
    amount = compute_n_parameters(model)
    print('params amount: {:,d}'.format(amount, ',d'))

    # create exp folder
    path_fig = os.path.join(args.exp_dir, 'runs')
    if not os.path.exists(args.exp_dir):
        os.makedirs(args.exp_dir)
    if not os.path.exists(path_fig):
        os.makedirs(path_fig)

    path_log = os.path.join(args.exp_dir, 'log.txt')
    with open(path_log, 'w') as fp:
        fp.write('\n')

    path_memo = os.path.join(args.exp_dir, 'memo.txt')
    with open(path_memo, 'w') as fp:
        fp.write(args.exp_memo+'\n')

    # device
    model.to(args.device)
    model.train()

    # dataloader
    dataloader = DataLoader(
        dataset, 
        batch_size=args.batch_size,
        num_workers=8,
        shuffle=True)  

    # optimizer
    optimizer = optim.Adam(model.parameters(), lr=args.lr)

    # training config
    counter = 0
    num_batch = len(dataset) // args.batch_size
    acc_batch_time = 0
    time_start_train = time.time()

    # loss
    loss_func.to(args.device)

    # training phase
    print('\n{:=^40}'.format(' start training '))
    for epoch in range(args.epochs):
        for bidx, batch in enumerate(dataloader):
            time_start_batch = time.time()

            # load batch
            x = batch['audio']
            f0 = batch['f0']
            loud = batch['lo']
            x, f0, loud = [it.to(args.device, non_blocking=True) for it in [x, f0, loud]]
            conditions = (f0.transpose(1, 2), loud.transpose(1, 2))
            
            # forward
            x_rec, params, tensors = model(x, conditions)

            # loss
            loss = loss_func(x_rec, x).mean(dim=0)

            # backward
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            # loss
            if counter % args.interval_print_loss == 0:
                acc_batch_time += time.time() - time_start_batch
                train_time = time.time() - time_start_train

                log = '{} - epoch: {}/{} ({:3d}/{:3d}) | t: {:.2f} | loss: {:.6f} | time: {} | iter: {}'.format(
                    args.exp_dir,
                    epoch, args.epochs, 
                    bidx, num_batch, 
                    acc_batch_time,
                    loss.item(), 
                    str(datetime.timedelta(seconds=train_time))[:-5],
                    counter)
                print(log)
                with open(path_log, 'a') as fp:
                    fp.write(log + '\n')
                acc_batch_time = 0

            # plot
            if counter % args.interval_plot == 0:
                print(' [*] plotting...')
                idx = np.random.randint(0, args.batch_size-1)
                x_harmonic, x_noise, x_dry = tensors
                amp, alpha, filter_coeff, reverb = params
                report.plot_runtime_info(
                    x[idx], 
                    x_rec[idx], 
                    args.sr, 
                    (f0[idx], loud[idx]), 
                    (amp[idx], alpha[idx]),
                    (x_harmonic[idx], x_noise[idx], x_dry[idx]), 
                    path_fig,
                    counter)

            # save model
            if counter % args.interval_save_model == 0:
                print(' [*] saving model...')
                torch.save(model.state_dict(), os.path.join(args.exp_dir, 'model.pt'))
                torch.save(optimizer.state_dict(), os.path.join(args.exp_dir, 'optimizer.pt'))

                try:
                    report.make_loss_report(path_log, os.path.join(args.exp_dir, 'loss_report.png'))
                except:
                    pass

            counter += 1

    # done
    print('{:=^40}'.format(' Finished '))

    # save
    torch.save(model, os.path.join(args.exp_dir, 'model.pt'))
    torch.save(optimizer.state_dict(), os.path.join(args.exp_dir, 'optimizer.pt'))
   
    # runtime
    runtime = time.time() - time_start_train
    print('training time:', str(datetime.timedelta(seconds=runtime))+'\n\n')
