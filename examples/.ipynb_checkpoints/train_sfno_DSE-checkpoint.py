# coding=utf-8

# SPDX-FileCopyrightText: Copyright (c) 2022 The torch-harmonics Authors. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause
#
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are met:
#
# 1. Redistributions of source code must retain the above copyright notice, this
# list of conditions and the following disclaimer.
#
# 2. Redistributions in binary form must reproduce the above copyright notice,
# this list of conditions and the following disclaimer in the documentation
# and/or other materials provided with the distribution.
#
# 3. Neither the name of the copyright holder nor the names of its
# contributors may be used to endorse or promote products derived from
# this software without specific prior written permission.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
# AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
# IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
# DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
# FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
# DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
# SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
# CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
# OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
# OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
#

import os
import time

from tqdm import tqdm
from functools import partial

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.cuda import amp

import numpy as np
import pandas as pd

import matplotlib.pyplot as plt


import sys

sys.path.append("../torch_harmonics")
from examples.sfno import PdeDataset
from sht_dse import SHT_DSE, SFNO_dse_fp
from random_sampling import RandomSphericalSampling

# trains with a standard L1 of MSE loss
l1_loss = torch.nn.L1Loss()
mse_loss = torch.nn.MSELoss()

def l1_rel_error(truth, test):
    batch_size = truth.shape[0]
    difference = torch.zeros(batch_size)
    for batch in range(batch_size):
        difference[batch] = torch.mean(torch.abs(truth[batch] - test[batch]))/(torch.mean(torch.abs(truth[batch]))).item() * 100
    return difference

def plot_prediction_vs_target(phi, theta, prd, tar):
    """
    Plots a 3x3 grid with predictions, targets, and their absolute difference.
    
    Parameters:
        phi (array-like): Azimuthal angle data (1D array).
        theta (array-like): Polar angle data (1D array).
        prd (array-like): Predicted values (shape: [n_points, 3]).
        tar (array-like): Target values (shape: [n_points, 3]).
    """
    fig, axes = plt.subplots(3, 3, figsize=(12, 10), constrained_layout=True)
    
    # Compute absolute difference
    diff = np.abs(prd - tar)
    
    # Titles for rows
    row_titles = ["Target", "Prediction", "Absolute Difference"]

    for i in range(3):  # Loop over rows: prd, tar, |prd-tar|
        for j in range(3):  # Loop over columns (channels)
            ax = axes[j, i]
            if i == 0:
                contour = ax.tricontourf(phi, theta, tar[j, :], levels=100, cmap='viridis')
            elif i == 1:
                contour = ax.tricontourf(phi, theta, prd[j, :], levels=100, cmap='viridis')
            else:
                contour = ax.tricontourf(phi, theta, diff[j, :], levels=100, cmap='viridis')
                
            
            cbar = fig.colorbar(contour, ax=ax, orientation='vertical')
            cbar.ax.set_ylabel(f'Channel {j + 1}', rotation=270, labelpad=15)
            
            ax.set_title(f"{row_titles[i]} - Channel {j + 1}")
            ax.set_xlabel('Phi (Azimuthal Angle)')
            ax.set_ylabel('Theta (Polar Angle)')

    plt.savefig("example_prediction.png")
    plt.close('all')
    
# training function
def train_model_fp(model,
                dataloader,
                optimizer,
                gscaler,
                select_random_points,
                theta_index,
                phi_index,
                scheduler=None,
                nepochs=20,
                nfuture=0,
                num_examples=256,
                num_valid=64,
                loss_fn='l1',
                enable_amp=False,
                log_grads=0,
                save_model=False,
                plot_results=True):

    train_start = time.time()

    # count iterations
    iters = 0

    for epoch in range(nepochs):

        # time each epoch
        epoch_start = time.time()

        dataloader.dataset.set_initial_condition('random')
        dataloader.dataset.set_num_examples(num_examples)

        # get the solver for its convenience functions
        solver = dataloader.dataset.solver

        # do the training
        acc_loss = 0
        model.train()

        for inp, tar in dataloader:

            with torch.autocast(device_type="cuda", enabled=enable_amp):
                
                # import pdb; pdb.set_trace()
                # theta_index, phi_index, theta, phi = select_random_points.random_points_on_sphere(5000)
                # sht_transform = SHT_DSE(phi, theta, 22)

                # # Select a random set of points from the input and target grids
                # inp = select_random_points.get_random_sphere_data(inp, theta_index, phi_index)
                # tar = select_random_points.get_random_sphere_data(tar, theta_index, phi_index)

                # inp_sht = sht_transform.inverse(sht_transform.forward(inp))
                # plot_prediction_vs_target(phi, theta, inp[0].cpu(), inp_sht[0].cpu())

                
                # Select a random set of points from the input and target grids
                inp = select_random_points.get_random_sphere_data(inp, theta_index, phi_index)
                tar = select_random_points.get_random_sphere_data(tar, theta_index, phi_index)
                

                prd = model(inp)
                for _ in range(nfuture):
                    prd = model(prd)
                    
                if loss_fn == 'mse':
                    loss = mse_loss(prd, tar)
                elif loss_fn == 'l1':
                    loss = l1_loss(prd, tar)
                else:
                    raise NotImplementedError(f'Unknown loss function {loss_fn}')

            acc_loss += loss.item() * inp.size(0)

            # optimizer.zero_grad()
            # loss.backward()
            # optimizer.step()

            optimizer.zero_grad(set_to_none=True)
            gscaler.scale(loss).backward()

            if log_grads and iters % log_grads == 0:
                log_weights_and_grads(model, iters=iters)

            gscaler.step(optimizer)
            gscaler.update()

            iters += 1

        acc_loss = acc_loss / len(dataloader.dataset)

        dataloader.dataset.set_initial_condition('random')
        dataloader.dataset.set_num_examples(num_valid)

        # perform validation
        valid_loss = 0
        model.eval()
        
        errors = torch.zeros((num_valid))
        with torch.no_grad():
            for index, (inp, tar) in enumerate(dataloader):
                batch_size = inp.shape[0]
                
                # Select a random set of points from the input and target grids
                inp = select_random_points.get_random_sphere_data(inp, theta_index, phi_index)
                tar = select_random_points.get_random_sphere_data(tar, theta_index, phi_index)
                
                prd = model(inp)
                for _ in range(nfuture):
                    prd = model(prd)
                
                
                if loss_fn == 'mse':
                    loss = mse_loss(prd, tar)
                elif loss_fn == 'l1':
                    loss = l1_loss(prd, tar)
                else:
                    raise NotImplementedError(f'Unknown loss function {loss_fn}')

                valid_loss += loss.item() * inp.size(0)
                errors[batch_size*index:batch_size*(index+1)] = l1_rel_error(tar, prd)

                if index == 0 and plot_results:
                    plot_prediction_vs_target(phi_index, theta_index, prd[0].cpu(), tar[0].cpu())

        valid_loss = valid_loss / len(dataloader.dataset)

        if scheduler is not None:
            scheduler.step(valid_loss)
            

        epoch_time = time.time() - epoch_start

        print(f'--------------------------------------------------------------------------------')
        print(f'Epoch {epoch} summary:')
        for param_group in scheduler.optimizer.param_groups:
            print(f"learning rate: {param_group['lr']}")
        print(f'time taken: {epoch_time}')
        print(f'accumulated training loss: {acc_loss}')
        print(f'relative validation loss: {valid_loss}')
        print(f'median relative error: {torch.median(errors).item()}')



    train_time = time.time() - train_start

    print(f'--------------------------------------------------------------------------------')
    print(f'done. Training took {train_time}.')
    return valid_loss

 
# training function
def train_model_vp(model,
                dataloader,
                optimizer,
                gscaler,
                select_random_points,
                scheduler=None,
                nepochs=20,
                nfuture=0,
                num_examples=256,
                num_valid=64,
                loss_fn='l1',
                enable_amp=False,
                log_grads=0,
                save_model=False):

    train_start = time.time()

    # count iterations
    iters = 0

    for epoch in range(nepochs):

        # time each epoch
        epoch_start = time.time()

        dataloader.dataset.set_initial_condition('random')
        dataloader.dataset.set_num_examples(num_examples)

        # get the solver for its convenience functions
        solver = dataloader.dataset.solver

        # do the training
        acc_loss = 0
        model.train()

        for inp, tar in dataloader:

            # Select a random set of points
            theta_index, phi_index, theta, phi = select_random_points.random_points_on_sphere

            with torch.autocast(device_type="cuda", enabled=enable_amp):

                # Select a random set of points from the input and target grids
                inp = select_random_points.get_random_sphere_data(inp, theta, phi)
                tar = select_random_points.get_random_sphere_data(tar, theta, phi)
                sht_transform = SHT_DSE(phi, theta, degree)

                prd = model(inp)
                for _ in range(nfuture):
                    prd = model(prd)
                    
                if loss_fn == 'mse':
                    loss = mse_loss(prd, tar)
                elif loss_fn == 'l1':
                    loss = l1_loss(prd, tar)
                else:
                    raise NotImplementedError(f'Unknown loss function {loss_fn}')

            acc_loss += loss.item() * inp.size(0)

            optimizer.zero_grad(set_to_none=True)
            gscaler.scale(loss).backward()
            # optimizer.zero_grad()
            # loss.backward()
            # optimizer.step()

            if log_grads and iters % log_grads == 0:
                log_weights_and_grads(model, iters=iters)

            gscaler.step(optimizer)
            gscaler.update()

            iters += 1

        acc_loss = acc_loss / len(dataloader.dataset)

        dataloader.dataset.set_initial_condition('random')
        dataloader.dataset.set_num_examples(num_valid)

        # perform validation
        valid_loss = 0
        model.eval()
        
        errors = torch.zeros((num_valid))
        with torch.no_grad():
            for index, (inp, tar) in enumerate(dataloader):
                batch_size = inp.shape[0]
                
                # Select a random set of points from the input and target grids
                inp = select_random_points.get_random_sphere_data(inp, theta, phi)
                tar = select_random_points.get_random_sphere_data(tar, theta, phi)
                
                prd = model(inp)
                for _ in range(nfuture):
                    prd = model(prd)

                
                if loss_fn == 'mse':
                    loss = mse_loss(prd, tar)
                elif loss_fn == 'l1':
                    loss = l1_loss(prd, tar)
                else:
                    raise NotImplementedError(f'Unknown loss function {loss_fn}')

                valid_loss += loss.item() * inp.size(0)
                errors[batch_size*index:batch_size*(index+1)] = l1_rel_error(tar, prd)

        valid_loss = valid_loss / len(dataloader.dataset)

        if scheduler is not None:
            scheduler.step(valid_loss)

        epoch_time = time.time() - epoch_start

        print(f'--------------------------------------------------------------------------------')
        print(f'Epoch {epoch} summary:')
        print(f'time taken: {epoch_time}')
        print(f'accumulated training loss: {acc_loss}')
        print(f'relative validation loss: {valid_loss}')
        print(f'median relative error: {torch.median(errors).item()}')



    train_time = time.time() - train_start

    print(f'--------------------------------------------------------------------------------')
    print(f'done. Training took {train_time}.')
    return valid_loss


def main(train=True, load_checkpoint=False, enable_amp=False, log_grads=0):

    # set seed
    torch.manual_seed(333)
    torch.cuda.manual_seed(333)

    # set device
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    if torch.cuda.is_available():
        torch.cuda.set_device(device.index)

    # 1 hour prediction steps
    dt = 1*3600
    dt_solver = 150
    nsteps = dt//dt_solver
    dataset = PdeDataset(dt=dt, nsteps=nsteps, dims=(256, 512), device=device, normalize=True)
    dataloader = DataLoader(dataset, batch_size=16, shuffle=True, num_workers=0, persistent_workers=False)

    nlat = dataset.nlat
    nlon = dataset.nlon

    # For selecting a fixed/variable set of uniformly randomly distributed points on the sphere
    select_random_points = RandomSphericalSampling(nlon, nlat)

    def count_parameters(model):
        return sum(p.numel() for p in model.parameters() if p.requires_grad)


    # Define hyperparameters for the SFNO model
    degree = 22
    width = 128
    num_layers = 4
    in_channels = out_channels = 3

    train_fixed = True
    train_variable = False

    ######################################################################
    # Training a model where collocation points are fixed between samples
    ######################################################################
    if train_fixed:
        # select the set of points at the beginning
        num_points = 5000 # yields approximately 5000 valid points, this is not exact
        theta_index, phi_index, theta, phi = select_random_points.random_points_on_sphere(num_points)
        
        # Using Fixed Points: initialize the matrices for the SHT
        sht_transform = SHT_DSE(phi, theta, degree)
    
        # Initialize the SFNO using fixed, arbitrary points
        model = SFNO_dse_fp(in_channels, out_channels, degree, width, sht_transform, num_layers).to(device)
    
        # Count the number of parameters
        num_params = count_parameters(model)
        print(f'number of trainable params: {num_params}')
    
        optimizer = torch.optim.Adam(model.parameters(), lr=1E-3)
        # scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min')
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.9)
        gscaler = amp.GradScaler('cuda', enabled=enable_amp)
    
        # Training a model where collocation points are fixed between samples
        train_model_fp(model, dataloader, optimizer, gscaler, select_random_points, theta_index, phi_index, scheduler, nepochs=200, loss_fn='l1')

        
    ######################################################################
    # Training a model where collocation points vary between samples
    ######################################################################
    
    if train_variable:
        # Initialize the SFNO using variable, arbitrary points
        model = SFNO_dse_vp(in_channels, out_channels, degree, width, num_layers).to(device)
    
        # Count the number of parameters
        num_params = count_parameters(model)
        print(f'number of trainable params: {num_params}')
    
        optimizer = torch.optim.Adam(model.parameters(), lr=1E-3)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min')
        gscaler = amp.GradScaler('cuda', enabled=enable_amp)
    
        train_model_vp(model, dataloader, optimizer, gscaler, select_random_points, scheduler, nepochs=200, loss_fn='l1')
        

if __name__ == "__main__":
    import torch.multiprocessing as mp
    mp.set_start_method('forkserver', force=True)

    main(train=True, load_checkpoint=False, enable_amp=False, log_grads=0)
