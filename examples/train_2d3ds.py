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

from torch_harmonics.examples import SphericalSegmentationDataset, TarDownloader
from torch_harmonics import RealSHT

# wandb logging
import wandb


# helper routine for counting number of paramerters in model
def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


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
                plt.savefig(os.path.join(path_root, "pred_" + str(i // nskip) + ".png"))
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
                plt.savefig(os.path.join(path_root, "truth_" + str(i // nskip) + ".png"))
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
        plt.savefig(os.path.join(path_root, f"powerspectrum_{step}.png"))
        fig.clf()
        plt.close()

    return losses, fno_times, nwp_times


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


def main(train=True, load_checkpoint=False, enable_amp=False, log_grads=0, _data_dir="data"):

    # set seed
    torch.manual_seed(333)
    torch.cuda.manual_seed(333)

    # set parameters
    nfuture = 0

    # set device
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    if torch.cuda.is_available():
        torch.cuda.set_device(device.index)

    data_dir = os.path.join(_data_dir, "2D3DS")
    os.makedirs(data_dir, exist_ok=True)

    # 2D3DS download & dataset initialization
    tardl = TarDownloader(base_url="https://cvg-data.inf.ethz.ch/2d3ds/no_xyz/", local_dir=str(data_dir))
    _, classes = tardl.download_dataset([("area_3_no_xyz.tar", "area_3"), ("area_5a_no_xyz.tar", "area_5a")])
    dataset = SphericalSegmentationDataset([f"{data_dir}/area_3", f"{data_dir}/area_5a"], classes)

    dataloader = DataLoader(dataset, batch_size=4, shuffle=True, num_workers=0, persistent_workers=False)

    img_size = dataset.input_shape[1:]

    # prepare dicts containing models and corresponding metrics
    models = {}
    metrics = {}

    from torch_harmonics.examples.models import SphericalTransformerForSegmentation as S2T

    models[f"s2t_sc2_layers4_e32"] = partial(
        S2T,
        dataset.num_classes,
        img_size=img_size,
        grid="equiangular",
        in_chans=3,
        num_layers=4,
        scale_factor=2,
        embed_dim=128,
        activation_function="gelu",
        residual_prediction=False,
        pos_embed="spectral",
        use_mlp=True,
        normalization_layer="instance_norm",
        encoder_kernel_shape=(4, 4),
        filter_basis_type="morlet",
        upsample_sht=True,
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

        exp_dir = os.path.join(root_path, "checkpoints_2d3ds", model_name)
        if not os.path.isdir(exp_dir):
            os.makedirs(exp_dir, exist_ok=True)

        exp_dir = os.path.join(root_path, "checkpoints_2d3ds", model_name)
        if not os.path.isdir(exp_dir):
            os.makedirs(exp_dir, exist_ok=True)

        if load_checkpoint:
            model.load_state_dict(torch.load(os.path.join(exp_dir, "checkpoint.pt")))

        # run the training
        if train:
            run = wandb.init(project="local sno spherical swe", group=model_name, name=model_name + "_" + str(time.time()), config=model_handle.keywords)

            # optimizer:
            optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
            scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, "min")
            gscaler = torch.GradScaler("cuda", enabled=enable_amp)

            start_time = time.time()

            print(f"Training {model_name}, single step")
            train_model(model, dataloader, optimizer, gscaler, scheduler, nepochs=20, loss_fn="l2", enable_amp=enable_amp, log_grads=log_grads)

            training_time = time.time() - start_time

            run.finish()

            torch.save(model.state_dict(), os.path.join(exp_dir, "checkpoint.pt"))

        # set seed
        torch.manual_seed(333)
        torch.cuda.manual_seed(333)

        with torch.inference_mode():
            losses, fno_times, nwp_times = autoregressive_inference(model, dataset, os.path.join(exp_dir, "figures"), nsteps=nsteps, autoreg_steps=30, nics=50)
            metrics[model_name]["loss_mean"] = np.mean(losses)
            metrics[model_name]["loss_std"] = np.std(losses)
            metrics[model_name]["fno_time_mean"] = np.mean(fno_times)
            metrics[model_name]["fno_time_std"] = np.std(fno_times)
            metrics[model_name]["nwp_time_mean"] = np.mean(nwp_times)
            metrics[model_name]["nwp_time_std"] = np.std(nwp_times)
            if train:
                metrics[model_name]["training_time"] = training_time

    df = pd.DataFrame(metrics)
    if not os.path.isdir(
        os.path.join(
            exp_dir,
            "output_data",
        )
    ):
        os.makedirs(os.path.join(exp_dir, "output_data"), exist_ok=True)
    df.to_pickle(os.path.join(exp_dir, "output_data", "metrics.pkl"))


if __name__ == "__main__":
    import torch.multiprocessing as mp

    mp.set_start_method("forkserver", force=True)

    wandb.login()

    # main(train=False, load_checkpoint=True, enable_amp=False, log_grads=0)
    main(train=True, load_checkpoint=False, enable_amp=False, log_grads=0)
