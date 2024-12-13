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

import numpy as np
import pandas as pd

import matplotlib.pyplot as plt

from torch_harmonics.examples import PdeDataset
from torch_harmonics import RealSHT

# wandb logging
import wandb

wandb.login()


def l2loss_sphere(solver, prd, tar, relative=False, squared=True):
    loss = solver.integrate_grid((prd - tar) ** 2, dimensionless=True).sum(dim=-1)
    if relative:
        loss = loss / solver.integrate_grid(tar**2, dimensionless=True).sum(dim=-1)

    if not squared:
        loss = torch.sqrt(loss)
    loss = loss.mean()

    return loss


def spectral_l2loss_sphere(solver, prd, tar, relative=False, squared=True):
    # compute coefficients
    coeffs = torch.view_as_real(solver.sht(prd - tar))
    coeffs = coeffs[..., 0] ** 2 + coeffs[..., 1] ** 2
    norm2 = coeffs[..., :, 0] + 2 * torch.sum(coeffs[..., :, 1:], dim=-1)
    loss = torch.sum(norm2, dim=(-1, -2))

    if relative:
        tar_coeffs = torch.view_as_real(solver.sht(tar))
        tar_coeffs = tar_coeffs[..., 0] ** 2 + tar_coeffs[..., 1] ** 2
        tar_norm2 = tar_coeffs[..., :, 0] + 2 * torch.sum(tar_coeffs[..., :, 1:], dim=-1)
        tar_norm2 = torch.sum(tar_norm2, dim=(-1, -2))
        loss = loss / tar_norm2

    if not squared:
        loss = torch.sqrt(loss)
    loss = loss.mean()

    return loss


def spectral_loss_sphere(solver, prd, tar, relative=False, squared=True):
    # gradient weighting factors
    lmax = solver.sht.lmax
    ls = torch.arange(lmax).float()
    spectral_weights = (ls * (ls + 1)).reshape(1, 1, -1, 1).to(prd.device)

    # compute coefficients
    coeffs = torch.view_as_real(solver.sht(prd - tar))
    coeffs = coeffs[..., 0] ** 2 + coeffs[..., 1] ** 2
    coeffs = spectral_weights * coeffs
    norm2 = coeffs[..., :, 0] + 2 * torch.sum(coeffs[..., :, 1:], dim=-1)
    loss = torch.sum(norm2, dim=(-1, -2))

    if relative:
        tar_coeffs = torch.view_as_real(solver.sht(tar))
        tar_coeffs = tar_coeffs[..., 0] ** 2 + tar_coeffs[..., 1] ** 2
        tar_coeffs = spectral_weights * tar_coeffs
        tar_norm2 = tar_coeffs[..., :, 0] + 2 * torch.sum(tar_coeffs[..., :, 1:], dim=-1)
        tar_norm2 = torch.sum(tar_norm2, dim=(-1, -2))
        loss = loss / tar_norm2

    if not squared:
        loss = torch.sqrt(loss)
    loss = loss.mean()

    return loss


def h1loss_sphere(solver, prd, tar, relative=False, squared=True):
    # gradient weighting factors
    lmax = solver.sht.lmax
    ls = torch.arange(lmax).float()
    spectral_weights = (ls * (ls + 1)).reshape(1, 1, -1, 1).to(prd.device)

    # compute coefficients
    coeffs = torch.view_as_real(solver.sht(prd - tar))
    coeffs = coeffs[..., 0] ** 2 + coeffs[..., 1] ** 2
    h1_coeffs = spectral_weights * coeffs
    h1_norm2 = h1_coeffs[..., :, 0] + 2 * torch.sum(h1_coeffs[..., :, 1:], dim=-1)
    l2_norm2 = coeffs[..., :, 0] + 2 * torch.sum(coeffs[..., :, 1:], dim=-1)
    h1_loss = torch.sum(h1_norm2, dim=(-1, -2))
    l2_loss = torch.sum(l2_norm2, dim=(-1, -2))

    # strictly speaking this is not exactly h1 loss
    if not squared:
        loss = torch.sqrt(h1_loss) + torch.sqrt(l2_loss)
    else:
        loss = h1_loss + l2_loss

    if relative:
        raise NotImplementedError("Relative H1 loss not implemented")

    loss = loss.mean()

    return loss


def fluct_l2loss_sphere(solver, prd, tar, inp, relative=False, polar_opt=0):
    # compute the weighting factor first
    fluct = solver.integrate_grid((tar - inp) ** 2, dimensionless=True, polar_opt=polar_opt)
    weight = fluct / torch.sum(fluct, dim=-1, keepdim=True)
    # weight = weight.reshape(*weight.shape, 1, 1)

    loss = weight * solver.integrate_grid((prd - tar) ** 2, dimensionless=True, polar_opt=polar_opt)
    if relative:
        loss = loss / (weight * solver.integrate_grid(tar**2, dimensionless=True, polar_opt=polar_opt))
    loss = torch.mean(loss)
    return loss


# rolls out the FNO and compares to the classical solver
def autoregressive_inference(model, dataset, path_root, nsteps, autoreg_steps=10, nskip=1, plot_channel=0, nics=50):

    model.eval()

    # make output
    if not os.path.isdir(path_root):
        os.makedirs(path_root, exist_ok=True)

    losses = np.zeros(nics)
    fno_times = np.zeros(nics)
    nwp_times = np.zeros(nics)

    # accumulation buffers for the power spectrum
    prd_mean_coeffs = []
    ref_mean_coeffs = []

    for iic in range(nics):
        ic = dataset.solver.random_initial_condition(mach=0.2)
        inp_mean = dataset.inp_mean
        inp_var = dataset.inp_var

        prd = (dataset.solver.spec2grid(ic) - inp_mean) / torch.sqrt(inp_var)
        prd = prd.unsqueeze(0)
        uspec = ic.clone()

        # add IC to power spectrum series
        prd_coeffs = [dataset.sht(prd[0, plot_channel]).detach().cpu().clone()]
        ref_coeffs = [prd_coeffs[0].clone()]

        # ML model
        start_time = time.time()
        for i in range(1, autoreg_steps + 1):
            # evaluate the ML model
            prd = model(prd)

            prd_coeffs.append(dataset.sht(prd[0, plot_channel]).detach().cpu().clone())

            if iic == nics - 1 and nskip > 0 and i % nskip == 0:

                # do plotting
                fig = plt.figure(figsize=(7.5, 6))
                dataset.solver.plot_griddata(prd[0, plot_channel], fig, vmax=4, vmin=-4, projection="robinson")
                plt.savefig(os.path.join(path_root,'pred_'+str(i//nskip)+'.png'))
                plt.close()

        fno_times[iic] = time.time() - start_time

        # classical model
        start_time = time.time()
        for i in range(1, autoreg_steps + 1):

            # advance classical model
            uspec = dataset.solver.timestep(uspec, nsteps)
            ref = (dataset.solver.spec2grid(uspec) - inp_mean) / torch.sqrt(inp_var)
            ref_coeffs.append(dataset.sht(ref[plot_channel]).detach().cpu().clone())

            if iic == nics - 1 and i % nskip == 0 and nskip > 0:

                fig = plt.figure(figsize=(7.5, 6))
                dataset.solver.plot_griddata(ref[plot_channel], fig, vmax=4, vmin=-4, projection="robinson")
                plt.savefig(os.path.join(path_root,'truth_'+str(i//nskip)+'.png'))
                plt.close()

        nwp_times[iic] = time.time() - start_time

        # compute power spectrum and add it to the buffers
        prd_mean_coeffs.append(torch.stack(prd_coeffs, 0))
        ref_mean_coeffs.append(torch.stack(ref_coeffs, 0))

        # ref = (dataset.solver.spec2grid(uspec) - inp_mean) / torch.sqrt(inp_var)
        ref = dataset.solver.spec2grid(uspec)
        prd = prd * torch.sqrt(inp_var) + inp_mean
        losses[iic] = l2loss_sphere(dataset.solver, prd, ref, relative=True).item()

    # compute the averaged powerspectra of prediction and reference
    with torch.no_grad():
        prd_mean_coeffs = torch.stack(prd_mean_coeffs, dim=0).abs().pow(2).mean(dim=0)
        ref_mean_coeffs = torch.stack(ref_mean_coeffs, dim=0).abs().pow(2).mean(dim=0)

        prd_mean_coeffs[..., 1:] *= 2.0
        ref_mean_coeffs[..., 1:] *= 2.0
        prd_mean_ps = prd_mean_coeffs.sum(dim=-1).contiguous()
        ref_mean_ps = ref_mean_coeffs.sum(dim=-1).contiguous()

        # split the stuff
        prd_mean_ps = [x.squeeze() for x in list(torch.split(prd_mean_ps, 1, dim=0))]
        ref_mean_ps = [x.squeeze() for x in list(torch.split(ref_mean_ps, 1, dim=0))]

    # compute the averaged powerspectrum
    for step, (pps, rps) in enumerate(zip(prd_mean_ps, ref_mean_ps)):
        fig = plt.figure(figsize=(7.5, 6))
        plt.semilogy(pps, label="prediction")
        plt.semilogy(rps, label="reference")
        plt.xlabel("$l$")
        plt.ylabel("powerspectrum")
        plt.legend()
        plt.savefig(os.path.join(path_root,f'powerspectrum_{step}.png'))
        fig.clf()
        plt.close()

    return losses, fno_times, nwp_times


# convenience function for logging weights and gradients
def log_weights_and_grads(model, iters=1):
    """
    Helper routine intended for debugging purposes
    """
    root_path = os.path.join(os.path.dirname(__file__), "weights_and_grads")

    weights_and_grads_fname = os.path.join(root_path, f"weights_and_grads_step{iters:03d}.tar")
    print(weights_and_grads_fname)

    weights_dict = {k: v for k, v in model.named_parameters()}
    grad_dict = {k: v.grad for k, v in model.named_parameters()}

    store_dict = {"iteration": iters, "grads": grad_dict, "weights": weights_dict}
    torch.save(store_dict, weights_and_grads_fname)


# training function
def train_model(model, dataloader, optimizer, gscaler, scheduler=None, nepochs=20, nfuture=0, num_examples=256, num_valid=8, loss_fn="l2", enable_amp=False, log_grads=0):

    train_start = time.time()

    # count iterations
    iters = 0

    for epoch in range(nepochs):

        # time each epoch
        epoch_start = time.time()

        dataloader.dataset.set_initial_condition("random")
        dataloader.dataset.set_num_examples(num_examples)

        # get the solver for its convenience functions
        solver = dataloader.dataset.solver

        # do the training
        acc_loss = 0
        model.train()

        for inp, tar in dataloader:

            with torch.autocast(device_type="cuda", enabled=enable_amp):

                prd = model(inp)
                for _ in range(nfuture):
                    prd = model(prd)

                if loss_fn == "l2":
                    loss = l2loss_sphere(solver, prd, tar, relative=False)
                elif loss_fn == "spectral l2":
                    loss = spectral_l2loss_sphere(solver, prd, tar, relative=False)
                elif loss_fn == "h1":
                    loss = h1loss_sphere(solver, prd, tar, relative=False)
                elif loss_fn == "spectral":
                    loss = spectral_loss_sphere(solver, prd, tar, relative=False)
                elif loss_fn == "fluct":
                    loss = fluct_l2loss_sphere(solver, prd, tar, inp, relative=True)
                else:
                    raise NotImplementedError(f"Unknown loss function {loss_fn}")

            acc_loss += loss.item() * inp.size(0)

            optimizer.zero_grad(set_to_none=True)
            gscaler.scale(loss).backward()

            if log_grads and iters % log_grads == 0:
                log_weights_and_grads(model, iters=iters)

            gscaler.step(optimizer)
            gscaler.update()

            iters += 1

        acc_loss = acc_loss / len(dataloader.dataset)

        dataloader.dataset.set_initial_condition("random")
        dataloader.dataset.set_num_examples(num_valid)

        # perform validation
        valid_loss = 0
        model.eval()
        with torch.no_grad():
            for inp, tar in dataloader:
                prd = model(inp)
                for _ in range(nfuture):
                    prd = model(prd)
                loss = l2loss_sphere(solver, prd, tar, relative=True)

                valid_loss += loss.item() * inp.size(0)

        valid_loss = valid_loss / len(dataloader.dataset)

        if scheduler is not None:
            scheduler.step(valid_loss)

        epoch_time = time.time() - epoch_start

        print(f"--------------------------------------------------------------------------------")
        print(f"Epoch {epoch} summary:")
        print(f"time taken: {epoch_time}")
        print(f"accumulated training loss: {acc_loss}")
        print(f"relative validation loss: {valid_loss}")

        if wandb.run is not None:
            current_lr = optimizer.param_groups[0]["lr"]
            wandb.log({"loss": acc_loss, "validation loss": valid_loss, "learning rate": current_lr})

    train_time = time.time() - train_start

    print(f"--------------------------------------------------------------------------------")
    print(f"done. Training took {train_time}.")
    return valid_loss


def main(train=True, load_checkpoint=False, enable_amp=False, log_grads=0):

    # set seed
    torch.manual_seed(333)
    torch.cuda.manual_seed(333)

    # set parameters
    nfuture=0

    # set device
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    if torch.cuda.is_available():
        torch.cuda.set_device(device.index)

    # 1 hour prediction steps
    dt = 1 * 3600
    dt_solver = 150
    nsteps = dt // dt_solver
    dataset = PdeDataset(dt=dt, nsteps=nsteps, dims=(257, 512), device=device, grid="legendre-gauss", normalize=True)
    dataset.sht = RealSHT(nlat=257, nlon=512, grid= "equiangular").to(device=device)
    # There is still an issue with parallel dataloading. Do NOT use it at the moment
    # dataloader = DataLoader(dataset, batch_size=4, shuffle=True, num_workers=4, persistent_workers=True)
    dataloader = DataLoader(dataset, batch_size=4, shuffle=True, num_workers=0, persistent_workers=False)

    nlat = dataset.nlat
    nlon = dataset.nlon

    def count_parameters(model):
        return sum(p.numel() for p in model.parameters() if p.requires_grad)

    # prepare dicts containing models and corresponding metrics
    models = {}
    metrics = {}

    from torch_harmonics.examples.models import SphericalFourierNeuralOperatorNet as SFNO
    from torch_harmonics.examples.models import LocalSphericalNeuralOperatorNet as LSNO

    models[f"sfno_sc2_layers4_e32_nomlp_leggauss"] = partial(
        SFNO,
        img_size=(nlat, nlon),
        grid="legendre-gauss",
        # hard_thresholding_fraction=0.8,
        num_layers=4,
        scale_factor=2,
        embed_dim=32,
        operator_type="driscoll-healy",
        activation_function="gelu",
        big_skip=False,
        pos_embed=False,
        use_mlp=False,
        normalization_layer="none",
    )

    models[f"lsno_sc1_layers4_e32_nomlp"] = partial(
        LSNO,
        spectral_transform="sht",
        img_size=(nlat, nlon),
        grid="equiangular",
        num_layers=4,
        scale_factor=2,
        embed_dim=32,
        operator_type="driscoll-healy",
        activation_function="gelu",
        big_skip=True,
        pos_embed=False,
        use_mlp=False,
        normalization_layer="none",
    )

    # iterate over models and train each model
    root_path = os.path.dirname(__file__)
    for model_name, model_handle in models.items():

        model = model_handle().to(device)

        print(model)

        metrics[model_name] = {}

        num_params = count_parameters(model)
        print(f"number of trainable params: {num_params}")
        metrics[model_name]["num_params"] = num_params

        exp_dir = os.path.join(root_path, 'checkpoints', model_name)
        if not os.path.isdir(exp_dir):
            os.makedirs(exp_dir, exist_ok=True)

        if load_checkpoint:
            model.load_state_dict(torch.load(os.path.join(exp_dir, "checkpoint.pt")))

        # run the training
        if train:
            run = wandb.init(project="sfno ablations spherical swe", group=model_name, name=model_name + "_" + str(time.time()), config=model_handle.keywords)

            # optimizer:
            optimizer = torch.optim.Adam(model.parameters(), lr=5e-4)
            scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, "min")
            gscaler = torch.GradScaler("cuda", enabled=enable_amp)

            start_time = time.time()

            print(f"Training {model_name}, single step")
            train_model(model, dataloader, optimizer, gscaler, scheduler, nepochs=20, loss_fn="l2", enable_amp=enable_amp, log_grads=log_grads)

            if nfuture > 0:
                print(f'Training {model_name}, {nfuture} step')
                optimizer = torch.optim.Adam(model.parameters(), lr=5E-5)
                scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min')
                gscaler = amp.GradScaler(enabled=enable_amp)
                dataloader.dataset.nsteps = 2 * dt//dt_solver
                train_model(model, dataloader, optimizer, gscaler, scheduler, nepochs=20, loss_fn="l2", nfuture=nfuture, enable_amp=enable_amp, log_grads=log_grads)
                dataloader.dataset.nsteps = 1 * dt//dt_solver

            training_time = time.time() - start_time

            run.finish()

            torch.save(model.state_dict(), os.path.join(exp_dir, 'checkpoint.pt'))

        # set seed
        torch.manual_seed(333)
        torch.cuda.manual_seed(333)

        with torch.inference_mode():
            losses, fno_times, nwp_times = autoregressive_inference(model, dataset, os.path.join(exp_dir,'figures'), nsteps=nsteps, autoreg_steps=30, nics=50)
            metrics[model_name]["loss_mean"] = np.mean(losses)
            metrics[model_name]["loss_std"] = np.std(losses)
            metrics[model_name]["fno_time_mean"] = np.mean(fno_times)
            metrics[model_name]["fno_time_std"] = np.std(fno_times)
            metrics[model_name]["nwp_time_mean"] = np.mean(nwp_times)
            metrics[model_name]["nwp_time_std"] = np.std(nwp_times)
            if train:
                metrics[model_name]["training_time"] = training_time

    df = pd.DataFrame(metrics)
    if not os.path.isdir(os.path.join(exp_dir, 'output_data',)):
        os.makedirs(os.path.join(exp_dir, 'output_data'), exist_ok=True)
    df.to_pickle(os.path.join(exp_dir, 'output_data', 'metrics.pkl'))


if __name__ == "__main__":
    import torch.multiprocessing as mp

    mp.set_start_method("forkserver", force=True)

    # main(train=False, load_checkpoint=True, enable_amp=False, log_grads=0)
    main(train=True, load_checkpoint=False, enable_amp=False, log_grads=0)
