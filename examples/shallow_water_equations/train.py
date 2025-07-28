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

import os, sys
import time
import argparse
from functools import partial

from tqdm import tqdm

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

import numpy as np
import pandas as pd

import matplotlib.pyplot as plt

from torch_harmonics.examples import PdeDataset
from torch_harmonics.examples.losses import L1LossS2, SquaredL2LossS2, L2LossS2, W11LossS2
from torch_harmonics import RealSHT
from torch_harmonics.plotting import plot_sphere

# import baseline models
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from model_registry import get_baseline_models

# wandb logging
try:
    import wandb
except:
    wandb = None


# helper routine for counting number of paramerters in model
def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


# convenience function for logging weights and gradients
def log_weights_and_grads(model, iters=1):
    root_path = os.path.join(os.path.dirname(__file__), "weights_and_grads")

    weights_and_grads_fname = os.path.join(root_path, f"weights_and_grads_step{iters:03d}.tar")
    print(weights_and_grads_fname)

    weights_dict = {k: v for k, v in model.named_parameters()}
    grad_dict = {k: v.grad for k, v in model.named_parameters()}

    store_dict = {"iteration": iters, "grads": grad_dict, "weights": weights_dict}
    torch.save(store_dict, weights_and_grads_fname)


# rolls out the FNO and compares to the classical solver
def autoregressive_inference(
    model,
    dataset,
    loss_fn,
    metrics_fns,
    path_root,
    nsteps,
    autoreg_steps=10,
    nskip=1,
    plot_channel=0,
    nics=50,
    device=torch.device("cpu"),
):

    model.eval()

    # make output
    if not os.path.isdir(path_root):
        os.makedirs(path_root, exist_ok=True)

    # accumulation buffers for losses, metrics and runtimes
    losses = torch.zeros(nics, dtype=torch.float32, device=device)
    metrics = {}
    for metric in metrics_fns:
        metrics[metric] = torch.zeros(nics, dtype=torch.float32, device=device)
    model_times = torch.zeros(nics, dtype=torch.float32, device=device)
    solver_times = torch.zeros(nics, dtype=torch.float32, device=device)

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

        # plot the initial condition
        if iic == nics - 1 and nskip > 0 and i % nskip == 0:

            # do plotting
            fig = plt.figure(figsize=(6, 6))
            plot_sphere(prd[0, plot_channel].cpu(), fig, vmax=4, vmin=-4, central_latitude=30, gridlines=True, projection="orthographic")
            fig.tight_layout()
            plt.savefig(os.path.join(path_root, "truth_" + str(0) + ".png"))
            plt.close()

        # ML model
        start_time = time.time()
        for i in range(1, autoreg_steps + 1):
            # evaluate the ML model
            prd = model(prd)

            prd_coeffs.append(dataset.sht(prd[0, plot_channel]).detach().cpu().clone())

            if iic == nics - 1 and nskip > 0 and i % nskip == 0:

                # do plotting
                fig = plt.figure(figsize=(6, 6))
                plot_sphere(prd[0, plot_channel].cpu(), fig, vmax=4, vmin=-4, central_latitude=30, gridlines=True, projection="orthographic")
                fig.tight_layout()
                plt.savefig(os.path.join(path_root, "pred_" + str(i // nskip) + ".png"))
                plt.close()

        model_times[iic] = time.time() - start_time

        # classical model
        start_time = time.time()
        for i in range(1, autoreg_steps + 1):

            # advance classical model
            uspec = dataset.solver.timestep(uspec, nsteps)
            ref = (dataset.solver.spec2grid(uspec) - inp_mean) / torch.sqrt(inp_var)
            ref_coeffs.append(dataset.sht(ref[plot_channel]).detach().cpu().clone())

            if iic == nics - 1 and i % nskip == 0 and nskip > 0:

                fig = plt.figure(figsize=(6, 6))
                plot_sphere(ref[plot_channel].cpu(), fig, vmax=4, vmin=-4, central_latitude=30, gridlines=True, projection="orthographic")
                fig.tight_layout()
                plt.savefig(os.path.join(path_root, "truth_" + str(i // nskip) + ".png"))
                plt.close()

        solver_times[iic] = time.time() - start_time

        # compute power spectrum and add it to the buffers
        prd_mean_coeffs.append(torch.stack(prd_coeffs, 0))
        ref_mean_coeffs.append(torch.stack(ref_coeffs, 0))

        ref = (dataset.solver.spec2grid(uspec) - inp_mean) / torch.sqrt(inp_var)
        # ref = dataset.solver.spec2grid(uspec)
        losses[iic] = loss_fn(prd, ref)
        # prd = prd * torch.sqrt(inp_var) + inp_mean
        for metric in metrics_fns:
            metric_buff = metrics[metric]
            metric_fn = metrics_fns[metric]
            metric_buff[iic] = metric_fn(prd, ref)

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
        fig.tight_layout()
        plt.savefig(os.path.join(path_root, f"powerspectrum_{step}.png"))
        fig.clf()
        plt.close()

    return losses, metrics, model_times, solver_times


# training function
def train_model(
    model,
    dataloader,
    loss_fn,
    metrics_fns,
    optimizer,
    gscaler,
    scheduler=None,
    nepochs=20,
    nfuture=0,
    num_examples=256,
    num_valid=8,
    amp_mode="none",
    log_grads=0,
    logging=True,
    device=torch.device("cpu"),
):
   
    train_start = time.time()

    # set AMP type
    amp_dtype = torch.float32
    if amp_mode == "fp16":
        amp_dtype = torch.float16
    elif amp_mode == "bf16":
        amp_dtype = torch.bfloat16

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
        accumulated_loss = 0
        model.train()

        for inp, tar in dataloader:

            with torch.autocast(device_type="cuda", dtype=amp_dtype, enabled=(amp_mode != "none")):

                prd = model(inp)
                for _ in range(nfuture):
                    prd = model(prd)

                loss = loss_fn(prd, tar)

            accumulated_loss += loss.item() * inp.size(0)

            optimizer.zero_grad(set_to_none=True)
            gscaler.scale(loss).backward()

            if log_grads and iters % log_grads == 0:
                log_weights_and_grads(model, iters=iters)

            gscaler.step(optimizer)
            gscaler.update()

            iters += 1

        accumulated_loss = accumulated_loss / len(dataloader.dataset)

        dataloader.dataset.set_initial_condition("random")
        dataloader.dataset.set_num_examples(num_valid)

        # eval mode
        model.eval()

        # prepare loss buffer for validation loss
        valid_loss = torch.zeros(2, dtype=torch.float32, device=device)

        # prepare metrics buffer for accumulation of validation metrics
        valid_metrics = {}
        for metric in metrics_fns:
            valid_metrics[metric] = torch.zeros(2, dtype=torch.float32, device=device)

        # perform validation
        with torch.no_grad():
            for inp, tar in dataloader:
                prd = model(inp)
                for _ in range(nfuture):
                    prd = model(prd)
                loss = loss_fn(prd, tar).item()

                valid_loss[0] += loss * inp.size(0)
                valid_loss[1] += inp.size(0)

                for metric in metrics_fns:
                    metric_buff = valid_metrics[metric]
                    metric_fn = metrics_fns[metric]
                    metric_buff[0] += metric_fn(prd, tar) * inp.size(0)
                    metric_buff[1] += inp.size(0)

        valid_loss = (valid_loss[0] / valid_loss[1]).item()
        for metric in valid_metrics:
            valid_metrics[metric] = (valid_metrics[metric][0] / valid_metrics[metric][1]).item()

        if scheduler is not None:
            scheduler.step(valid_loss)

        epoch_time = time.time() - epoch_start

        if logging:
            print(f"--------------------------------------------------------------------------------")
            print(f"Epoch {epoch} summary:")
            print(f"time taken: {epoch_time:.2f}")
            print(f"accumulated training loss: {accumulated_loss}")
            print(f"validation loss: {valid_loss}")
            for metric in valid_metrics:
                print(f"{metric}: {valid_metrics[metric]}")

            if wandb.run is not None:
                current_lr = optimizer.param_groups[0]["lr"]
                log_dict = {"loss": accumulated_loss, "validation loss": valid_loss, "learning rate": current_lr}
                for metric in valid_metrics:
                    log_dict[metric] = valid_metrics[metric]
                wandb.log(log_dict)

    train_time = time.time() - train_start

    print(f"--------------------------------------------------------------------------------")
    print(f"done. Training took {train_time}.")
    return valid_loss


def main(root_path, pretrain_epochs=100, finetune_epochs=10, batch_size=1, learning_rate=1e-3, train=True, load_checkpoint=False, amp_mode="none", log_grads=0):

    # enable logging by default
    logging = True

    # set seed
    torch.manual_seed(333)
    torch.cuda.manual_seed(333)

    # set device
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    if torch.cuda.is_available():
        torch.cuda.set_device(device.index)

    # 1 hour prediction steps
    dt = 1 * 3600
    dt_solver = 150
    nsteps = dt // dt_solver
    grid = "legendre-gauss"
    nlat, nlon = (128, 256)
    dataset = PdeDataset(dt=dt, nsteps=nsteps, dims=(nlat, nlon), device=device, grid=grid, normalize=True)
    dataset.sht = RealSHT(nlat=nlat, nlon=nlon, grid=grid).to(device=device)
    # There is still an issue with parallel dataloading. Do NOT use it at the moment
    # dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=4, persistent_workers=True)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=0, persistent_workers=False)

    nlat = dataset.nlat
    nlon = dataset.nlon

    # prepare dicts containing models and corresponding metrics
    models = {}

    # get baseline model registry
    baseline_models = get_baseline_models(img_size=(nlat, nlon), in_chans=3, out_chans=3, residual_prediction=True, grid=grid)

    # specify which models to train here
    models = [
        # "transformer_sc2_layers4_e128",
        # "s2transformer_sc2_layers4_e128",
        # "ntransformer_sc2_layers4_e128",
        # "s2ntransformer_sc2_layers4_e128",
        # "segformer_sc2_layers4_e128",
        # "s2segformer_sc2_layers4_e128",
        # "nsegformer_sc2_layers4_e128",
        # "s2nsegformer_sc2_layers4_e128",
        "egformer",
        # "sfno_sc2_layers4_e32",
        # "lsno_sc2_layers4_e32",
    ]
    models = {k: baseline_models[k] for k in models}

    # loss function
    loss_fn = SquaredL2LossS2(nlat=nlat, nlon=nlon, grid=grid).to(device)

    # dictionary for logging the metrics
    metrics = {}
    metrics_fns = {
        "L2 error": L2LossS2(nlat=nlat, nlon=nlon, grid=grid).to(device=device),
        "L1 error": L1LossS2(nlat=nlat, nlon=nlon, grid=grid).to(device=device),
        "W11 error": W11LossS2(nlat=nlat, nlon=nlon, grid=grid).to(device=device),
    }

    # iterate over models and train each model
    for model_name, model_handle in models.items():

        model = model_handle().to(device)

        print(model)

        metrics[model_name] = {}

        num_params = count_parameters(model)
        print(f"number of trainable params: {num_params}")
        metrics[model_name]["num_params"] = num_params

        exp_dir = os.path.join(root_path, model_name)
        if not os.path.isdir(exp_dir):
            os.makedirs(exp_dir, exist_ok=True)

        if load_checkpoint:
            model.load_state_dict(torch.load(os.path.join(exp_dir, "checkpoint.pt")))

        # run the training
        if train:
            if logging and wandb is not None:
                run = wandb.init(project="spherical shallow water equations", group=model_name, name=model_name + "_" + str(time.time()), config=model_handle.keywords)
            else:
                run = None

            # optimizer:
            optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
            scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, "min")
            gscaler = torch.GradScaler("cuda", enabled=(amp_mode == "fp16"))

            start_time = time.time()

            if logging:
                print(f"Training {model_name}, single step")

            train_model(
                model,
                dataloader,
                loss_fn,
                metrics_fns,
                optimizer,
                gscaler,
                scheduler,
                nepochs=pretrain_epochs,
                amp_mode=amp_mode,
                log_grads=log_grads,
                logging=logging,
                device=device,
            )

            if finetune_epochs > 0:
                nfuture = 1

                if logging:
                    print(f"Finetuning {model_name}, {nfuture} step")

                optimizer = torch.optim.Adam(model.parameters(), lr=0.1 * learning_rate)
                scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, "min")
                gscaler = torch.GradScaler(enabled=(amp_mode != "none"))
                dataloader.dataset.nsteps = 2 * dt // dt_solver
                train_model(
                    model,
                    dataloader,
                    loss_fn,
                    metrics_fns,
                    optimizer,
                    gscaler,
                    scheduler,
                    nepochs=finetune_epochs,
                    nfuture=nfuture,
                    amp_mode=amp_mode,
                    log_grads=log_grads,
                    logging=logging,
                    device=device,
                )
                dataloader.dataset.nsteps = 1 * dt // dt_solver

            training_time = time.time() - start_time

            if logging and run is not None:
                run.finish()

            torch.save(model.state_dict(), os.path.join(exp_dir, "checkpoint.pt"))

        # set seed
        torch.manual_seed(333)
        torch.cuda.manual_seed(333)

        # run validation
        print(f"Validating {model_name}")
        with torch.inference_mode():
            losses, metric_results, model_times, solver_times = autoregressive_inference(
                model, dataset, loss_fn, metrics_fns, os.path.join(exp_dir, "figures"), nsteps=nsteps, autoreg_steps=1, nics=50, device=device
            )

            # compute statistics
            metrics[model_name]["loss mean"] = torch.mean(losses).item()
            metrics[model_name]["loss std"] = torch.std(losses).item()
            metrics[model_name]["model time mean"] = torch.mean(model_times).item()
            metrics[model_name]["model time std"] = torch.std(model_times).item()
            metrics[model_name]["solver time mean"] = torch.mean(solver_times).item()
            metrics[model_name]["solver time std"] = torch.std(solver_times).item()
            for metric in metric_results:
                metrics[model_name][metric + " mean"] = torch.mean(metric_results[metric]).item()
                metrics[model_name][metric + " std"] = torch.std(metric_results[metric]).item()

            if train:
                metrics[model_name]["training_time"] = training_time

    # output metrics to data frame
    df = pd.DataFrame(metrics)
    if not os.path.isdir(os.path.join(root_path, "output_data")):
        os.makedirs(os.path.join(root_path, "output_data"), exist_ok=True)
    df.to_pickle(os.path.join(root_path, "output_data", "metrics.pkl"))


if __name__ == "__main__":
    import torch.multiprocessing as mp

    mp.set_start_method("forkserver", force=True)
    if wandb is not None:
        wandb.login()

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--root_path", default=os.path.join(os.path.dirname(__file__), "checkpoints"), type=str, help="Override the path where checkpoints and run information are stored"
    )
    parser.add_argument("--pretrain_epochs", default=100, type=int, help="Number of pretraining epochs.")
    parser.add_argument("--finetune_epochs", default=0, type=int, help="Number of fine-tuning epochs.")
    parser.add_argument("--batch_size", default=4, type=int, help="Switch for overriding batch size in the configuration file.")
    parser.add_argument("--learning_rate", default=1e-4, type=float, help="Switch to override learning rate.")
    parser.add_argument("--resume", action="store_true", help="Reload checkpoints.")
    parser.add_argument("--amp_mode", default="none", type=str, choices=["none", "bf16", "fp16"], help="Switch to enable AMP.")
    args = parser.parse_args()

    # main(train=False, load_checkpoint=True, enable_amp=False, log_grads=0)
    main(
        root_path=args.root_path,
        pretrain_epochs=args.pretrain_epochs,
        finetune_epochs=args.finetune_epochs,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        train=(args.pretrain_epochs > 0 or args.finetune_epochs > 0),
        load_checkpoint=args.resume,
        amp_mode=args.amp_mode,
        log_grads=0,
    )
