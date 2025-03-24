# coding=utf-8

# SPDX-FileCopyrightText: Copyright (c) 2025 The torch-harmonics Authors. All rights reserved.
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
import sys
import random

import time

import argparse

from tqdm import tqdm
from functools import partial

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data.distributed import DistributedSampler

import numpy as np
import pandas as pd

import matplotlib.pyplot as plt

from torch_harmonics.examples import StanfordDepthDataset, Stanford2D3DSDownloader
from torch_harmonics.quadrature import _precompute_latitudes
from torch_harmonics.examples.losses import L2LossS2
from torch_harmonics.examples.metrics import RmseS2
from torch_harmonics.plotting import plot_sphere, imshow_sphere

# wandb logging
import wandb


# helper routine for counting number of paramerters in model
def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


# convenience function for logging weights and gradients
def log_weights_and_grads(exp_dir, model, iters=1):
    """
    Helper routine intended for debugging purposes
    """
    log_path = os.path.join(exp_dir, "weights_and_grads")
    if not os.path.isdir(log_path):
        os.makedirs(log_path, exist_ok=True)

    weights_and_grads_fname = os.path.join(log_path, f"weights_and_grads_step{iters:03d}.tar")
    print(weights_and_grads_fname)

    weights_dict = {k: v for k, v in model.named_parameters()}
    grad_dict = {k: v.grad for k, v in model.named_parameters()}

    store_dict = {"iteration": iters, "grads": grad_dict, "weights": weights_dict}
    torch.save(store_dict, weights_and_grads_fname)


# rolls out the FNO and compares to the classical solver
def validate_model(model, dataloader, loss_fn, metrics_fns, path_root, normalization=None, logging=True, device=torch.device("cpu")):

    model.eval()

    num_examples = len(dataloader)

    # make output
    if logging and not os.path.isdir(path_root):
        os.makedirs(path_root, exist_ok=True)

    if dist.is_initialized():
        dist.barrier(device_ids=[device.index])

    losses = torch.zeros(num_examples, dtype=torch.float32, device=device)
    # fno_times = np.zeros(num_samples)

    metrics = {}
    for metric in metrics_fns:
        metrics[metric] = torch.zeros(num_examples, dtype=torch.float32, device=device)

    glob_off = 0
    if dist.is_initialized():
        glob_off = num_examples * dist.get_rank()

    with torch.no_grad():
        for idx, (inp, tar) in enumerate(dataloader):
            inp = inp.to(device)
            tar = tar.to(device)

            if normalization is not None:
                inp = normalization(inp)

            prd = model(inp)

            losses[idx] = loss_fn(prd, tar)

            for metric in metrics_fns:
                metric_buff = metrics[metric]
                metric_fn = metrics_fns[metric]
                metric_buff[idx] = metric_fn(prd, tar)

            prd = nn.functional.softmax(prd, dim=-3)
            prd = torch.argmax(prd, dim=-3).squeeze(0)

            # do plotting
            glob_idx = idx + glob_off
            fig = plt.figure(figsize=(7.5, 6))
            plot_sphere(prd.cpu(), fig=fig, vmax=1.0, vmin=0.0, cmap="rainbow")
            plt.savefig(os.path.join(path_root, "pred_" + str(glob_idx) + ".png"))
            plt.close()

            fig = plt.figure(figsize=(7.5, 6))
            plot_sphere(tar.cpu().squeeze(0), fig=fig, vmax=1.0, vmin=0.0, cmap="rainbow")
            plt.savefig(os.path.join(path_root, "truth_" + str(glob_idx) + ".png"))
            plt.close()

    return losses, metrics


# training function
def train_model(
    model,
    train_dataloader,
    train_sampler,
    test_dataloader,
    test_sampler,
    loss_fn,
    metrics_fns,
    optimizer,
    gscaler,
    scheduler=None,
    normalization=None,
    augmentation=None,
    nepochs=20,
    amp_mode="none",
    log_grads=0,
    exp_dir=None,
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

        # do the training
        accumulated_loss = torch.zeros(2, dtype=torch.float32, device=device)

        model.train()

        if dist.is_initialized():
            train_sampler.set_epoch(epoch)

        for step, (inp, tar) in enumerate(train_dataloader):
            inp = inp.to(device)
            tar = tar.to(device)

            if normalization is not None:
                inp = normalization(inp)

            if augmentation is not None:
                inp = augmentation(inp)
                
                # flip randomly horizontally
                if random.random() < 0.5:
                    inp = torch.flip(inp, dims=(-1,))
                    tar = torch.flip(tar, dims=(-1,))

            with torch.autocast(device_type="cuda", dtype=amp_dtype, enabled=(amp_mode != "none")):
                prd = model(inp)
                loss = loss_fn(prd, tar)

            optimizer.zero_grad(set_to_none=True)
            gscaler.scale(loss).backward()

            if log_grads and (iters % log_grads == 0) and (exp_dir is not None):
                log_weights_and_grads(exp_dir, model, iters=iters)

            gscaler.step(optimizer)
            gscaler.update()

            # accumulate loss
            accumulated_loss[0] += loss.detach() * inp.size(0)
            accumulated_loss[1] += inp.size(0)

            iters += 1

        if dist.is_initialized():
            dist.all_reduce(accumulated_loss)

        accumulated_loss = (accumulated_loss[0] / accumulated_loss[1]).item()

        # perform validation
        valid_loss = torch.zeros(2, dtype=torch.float32, device=device)

        valid_metrics = {}
        for metric in metrics_fns:
            valid_metrics[metric] = torch.zeros(2, dtype=torch.float32, device=device)

        model.eval()

        if dist.is_initialized():
            test_sampler.set_epoch(epoch)

        with torch.no_grad():
            for step, (inp, tar) in enumerate(test_dataloader):
                inp = inp.to(device)
                tar = tar.to(device)

                if normalization is not None:
                    inp = normalization(inp)

                with torch.autocast(device_type="cuda", dtype=amp_dtype, enabled=(amp_mode != "none")):
                    prd = model(inp)
                    loss = loss_fn(prd, tar)

                valid_loss[0] += loss * inp.size(0)
                valid_loss[1] += inp.size(0)

                for metric in metrics_fns:
                    metric_buff = valid_metrics[metric]
                    metric_fn = metrics_fns[metric]
                    metric_buff[0] += metric_fn(prd, tar) * inp.size(0)
                    metric_buff[1] += inp.size(0)

            if dist.is_initialized():
                dist.all_reduce(valid_loss)
                for metric in metrics_fns:
                    dist.all_reduce(valid_metrics[metric])

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
            print(f"relative validation loss: {valid_loss}")
            for metric in valid_metrics:
                print(f"{metric}: {valid_metrics[metric]}")

            if wandb.run is not None:
                current_lr = optimizer.param_groups[0]["lr"]
                wandb.log({"loss": accumulated_loss, "validation loss": valid_loss, "learning rate": current_lr})

    # wrapping up
    train_time = time.time() - train_start

    if logging:
        print(f"--------------------------------------------------------------------------------")
        print(f"done. Training took {train_time:.2f}.")

    return valid_loss


def main(
    root_path,
    num_epochs=100,
    batch_size=8,
    learning_rate=1e-4,
    train=True,
    load_checkpoint=False,
    amp_mode="none",
    ddp=False,
    enable_data_augmentation=False,
    ignore_alpha_channel=True,
    log_grads=0,
    data_path="data",
):

    # initialize distributed
    local_rank = 0
    logging = True
    if ddp:
        dist.init_process_group(backend="nccl")
        local_rank = dist.get_rank() % torch.cuda.device_count()
        logging = dist.get_rank() == 0

    # set seed
    torch.manual_seed(333)
    torch.cuda.manual_seed(333)

    # set device
    device = torch.device(f"cuda:{local_rank}" if torch.cuda.is_available() else "cpu")
    if torch.cuda.is_available():
        torch.cuda.set_device(device.index)

    # create dataset directory if it doesn't exist
    if logging:
        os.makedirs(data_path, exist_ok=True)

    # 2D3DS download & dataset initialization
    downloader = Stanford2D3DSDownloader(base_url="https://cvg-data.inf.ethz.ch/2d3ds/no_xyz/", local_dir=str(data_path))
    dataset_file = downloader.prepare_dataset(downsampling_factor=16)

    # intiialize distributed for ddp
    if dist.is_initialized():
        dist.barrier(device_ids=[device.index])

    # create the dataset and split it
    if logging:
        print(f"Initializing dataset...")

    # make sure splitting is consistent across ranks
    rng = torch.Generator().manual_seed(333)
    split_ratios = [0.95, 0.025, 0.025]
    dataset = StanfordDepthDataset(dataset_file=dataset_file, ignore_alpha_channel=ignore_alpha_channel)
    train_dataset, test_dataset, valid_dataset = torch.utils.data.random_split(dataset, split_ratios, generator=rng)

    # split dataset if distributed
    if dist.is_initialized():
        train_sampler = DistributedSampler(train_dataset, num_replicas=dist.get_world_size(), rank=dist.get_rank(), shuffle=True, drop_last=True)
        test_sampler = DistributedSampler(test_dataset, num_replicas=dist.get_world_size(), rank=dist.get_rank(), shuffle=False, drop_last=True)
        valid_sampler = DistributedSampler(valid_dataset, num_replicas=dist.get_world_size(), rank=dist.get_rank(), shuffle=False, drop_last=True)
    else:
        train_sampler = None
        test_sampler = None
        valid_sampler = None

    # create the dataloaders
    train_dataloader = DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True if train_sampler is None else False,
        sampler=train_sampler, num_workers=4, persistent_workers=True, pin_memory=True
    )
    test_dataloader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, sampler=test_sampler,
                                 num_workers=4, persistent_workers=True, pin_memory=True)
    valid_dataloader = DataLoader(valid_dataset, batch_size=1, shuffle=False, sampler=valid_sampler,
                                  num_workers=0, persistent_workers=False, pin_memory=True)

    # TODO: move augmentation into extra helper module
    if enable_data_augmentation:
        if not ignore_alpha_channel:
            raise NotImplementedError("You can only use data augmentation with RGB images, RGBA is not supported.")
        if logging:
            print("Using data augmentation")
        from torchvision.transforms import v2

        # get stas from file: WARNING! STATS HAVE BEEN COMPUTED OVER ALL SAMPLES
        # NEED TO DISENTANGLE THAT CORRECTLY WITH STATIC SPLITS!
        #mean = dataset.mean
        #std = dataset.std
        #print(f"Applying mean/variance normalization with mean={mean}, std={std}.")
        #normalization = v2.Normalize(mean=mean, std=std)

        # imagenet normalization
        normalization = v2.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        augmentation = v2.Compose(
            [
                v2.RandomAutocontrast(p=0.5),
                v2.GaussianNoise(mean=0.0, sigma=0.1, clip=True),
                v2.ColorJitter(),
            ]
        )
    else:
        normalization = None
        augmentation = None

    in_channels = 3 if ignore_alpha_channel else 4

    # print dataset info
    img_size = dataset.input_shape[1:]
    
    if logging:
        print(f"Train dataset initialized with {len(train_dataset)} samples of resolution {img_size}")
        print(f"Test dataset initialized with {len(test_dataset)} samples of resolution {img_size}")
        print(f"Validation dataset initialized with {len(valid_dataset)} samples of resolution {img_size}")

    # prepare dicts containing models and corresponding metrics
    models = {}
    metrics = {}

    from torch_harmonics.examples.models import SphericalFourierNeuralOperator as SFNO
    from torch_harmonics.examples.models import LocalSphericalNeuralOperator as LSNO
    from torch_harmonics.examples.models import SphericalTransformer as S2T

    nlat = 128
    nlon = 256
    grid = "equiangular"

    # pip install segmentation-models-pytorch
    try:
        from segmentation_models_pytorch import UnetPlusPlus
    except:
        print("Segmentation Models package not found, some models are not available")

    models[f"sfno_sc2_layers4_e32"] = partial(
        SFNO,
        out_chans=1,
        img_size=(nlat, nlon),
        grid=grid,
        num_layers=4,
        scale_factor=2,
        embed_dim=32,
        activation_function="gelu",
        residual_prediction=False,
        use_mlp=True,
        normalization_layer="none",
    )

    models[f"lsno_sc2_layers4_e32_morlet"] = partial(
        LSNO,
        out_chans=1,
        img_size=(nlat, nlon),
        grid=grid,
        num_layers=4,
        scale_factor=2,
        embed_dim=32,
        activation_function="gelu",
        residual_prediction=False,
        use_mlp=True,
        normalization_layer="none",
        kernel_shape=(4, 4),
        encoder_kernel_shape=(4, 4),
        filter_basis_type="morlet",
        upsample_sht = True,
    )

    models[f"s2t_sc2_layers4_e32_h1"] = partial(
        S2T,
        out_chans=1,
        img_size=(nlat, nlon),
        grid=grid,
        num_layers=4,
        scale_factor=2,
        embed_dim=32,
        activation_function="gelu",
        residual_prediction=False,
        pos_embed="spectral",
        num_heads=1,
        use_mlp=True,
        normalization_layer="instance_norm",
        encoder_kernel_shape=(4, 4),
        filter_basis_type="morlet",
        upsample_sht=True,
    )

    # create the loss object
    # loss_fn = CrossEntropyLossS2(nlat=img_size[0], nlon=img_size[1], grid="equiangular", weight=class_weights, smooth=0.).to(device=device)
    # loss_fn = DiceLossS2(nlat=img_size[0], nlon=img_size[1], grid="equiangular",  weight=class_weights, smooth=0.).to(device=device)
    # loss_fn = FocalLossS2(nlat=img_size[0], nlon=img_size[1], grid="equiangular").to(device=device)
    loss_fn = L2LossS2(nlat=img_size[0], nlon=img_size[1], grid="equiangular").to(device=device)
    # compile loss function
    # loss_fn = torch.compile(loss_fn, mode="max-autotune")

    # metrics
    metrics_fns = {
        "rmse": RmseS2(nlat=img_size[0], 
                        nlon=img_size[1], 
                        grid="equiangular").to(device=device)
    }

    # iterate over models and train each model
    for model_name, model_handle in models.items():

        model = model_handle().to(device)

        if logging:
            print(model)

        if dist.is_initialized():
            model = DDP(model, device_ids=[device.index])

        metrics[model_name] = {}

        num_params = count_parameters(model)
        if logging:
            print(f"number of trainable params: {num_params}")

        metrics[model_name]["num_params"] = num_params

        exp_dir = os.path.join(root_path, model_name)
        if not os.path.isdir(exp_dir):
            os.makedirs(exp_dir, exist_ok=True)

        if load_checkpoint:
            model.load_state_dict(torch.load(os.path.join(exp_dir, "checkpoint.pt")))

        # run the training
        if train:
            if logging:
                run = wandb.init(project="spherical segmentation 2d3ds", group=model_name, name=model_name + "_" + str(time.time()), config=model_handle.keywords)
            else:
                run = None

            # optimizer:
            optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=0.01,
                                          foreach=torch.cuda.is_available())
            scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, "min")
            gscaler = torch.GradScaler("cuda", enabled=(amp_mode == "fp16"))

            start_time = time.time()

            if logging:
                print(f"Training {model_name}")
            train_model(
                model,
                train_dataloader,
                train_sampler,
                test_dataloader,
                test_sampler,
                loss_fn,
                metrics_fns,
                optimizer,
                gscaler,
                scheduler,
                normalization=normalization,
                augmentation=augmentation,
                nepochs=num_epochs,
                amp_mode=amp_mode,
                log_grads=log_grads,
                exp_dir=exp_dir,
                logging=logging,
                device=device,
            )

            training_time = time.time() - start_time

            if logging:
                run.finish()
                torch.save(model.state_dict(), os.path.join(exp_dir, "checkpoint.pt"))

        # set seed
        torch.manual_seed(333)
        torch.cuda.manual_seed(333)

        with torch.inference_mode():
            losses, metric_results = validate_model(
                model, valid_dataloader, loss_fn, metrics_fns, os.path.join(exp_dir, "figures"), normalization=normalization, logging=logging, device=device
            )
            metrics[model_name]["loss_mean"] = torch.mean(losses)
            for metric in metric_results:
                metrics[model_name][metric + " mean"] = torch.mean(metric_results[metric])

            if dist.is_initialized():
                dist.all_reduce(metrics[model_name]["loss_mean"], dist.ReduceOp.AVG)
                for metric in metric_results:
                    dist.all_reduce(metrics[model_name][metric + " mean"], dist.ReduceOp.AVG)

            metrics[model_name]["loss_mean"] = metrics[model_name]["loss_mean"].item()
            for metric in metric_results:
                metrics[model_name][metric + " mean"] = metrics[model_name][metric + " mean"].item()
            # metrics[model_name]["loss_mean"] = np.mean(losses)
            # metrics[model_name]["loss_std"] = np.std(losses)
            # metrics[model_name]["fno_time_mean"] = np.mean(fno_times)
            # metrics[model_name]["fno_time_std"] = np.std(fno_times)
            if train:
                metrics[model_name]["training_time"] = training_time

    if logging:
        df = pd.DataFrame(metrics)
        if not os.path.isdir(os.path.join(exp_dir, "output_data")):
            os.makedirs(os.path.join(exp_dir, "output_data"), exist_ok=True)
        df.to_pickle(os.path.join(exp_dir, "output_data", "metrics.pkl"))

    if dist.is_initialized():
        dist.barrier(device_ids=[device.index])


if __name__ == "__main__":
    import torch.multiprocessing as mp

    mp.set_start_method("forkserver", force=True)

    wandb.login()

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--output_path", default=os.path.join(os.path.dirname(__file__), "checkpoints"), type=str, help="Override the path where checkpoints and run information are stored"
    )
    parser.add_argument(
        "--data_path",
        default=os.path.join(os.path.dirname(os.path.dirname(__file__)), "data"),
        type=str,
        help="Directory to where the dataset is stored. If the dataset is not found in that location, it will be downloaded automatically.",
    )
    parser.add_argument("--num_epochs", default=100, type=int, help="Switch for overriding batch size in the configuration file.")
    parser.add_argument("--batch_size", default=8, type=int, help="Switch for overriding batch size in the configuration file.")
    parser.add_argument("--learning_rate", default=1e-4, type=float, help="Switch to override learning rate.")
    parser.add_argument("--resume", action="store_true", help="Reload checkpoints.")
    parser.add_argument("--amp_mode", default="none", type=str, choices=["none", "bf16", "fp16"], help="Switch to enable AMP.")
    parser.add_argument("--enable_ddp", action="store_true", help="Switch to enable distributed data parallel.")
    parser.add_argument("--enable_data_augmentation", action="store_true", help="Switch to enable data augmentation.")
    args = parser.parse_args()

    main(
        root_path=args.output_path,
        num_epochs=args.num_epochs,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        train=True,
        load_checkpoint=args.resume,
        amp_mode=args.amp_mode,
        ddp=args.enable_ddp,
        enable_data_augmentation=args.enable_data_augmentation,
        ignore_alpha_channel=True,
        log_grads=0,
        data_path=args.data_path,
    )