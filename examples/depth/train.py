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

import os, sys
import time
import argparse

from functools import partial

import torch
from torch.utils.data import DataLoader
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data.distributed import DistributedSampler
from torchvision.transforms import v2

import pandas as pd
import matplotlib.pyplot as plt

from torch_harmonics.examples import (
    StanfordDepthDataset,
    Stanford2D3DSDownloader,
    compute_stats_s2,
)
from torch_harmonics.examples.losses import W11LossS2, L1LossS2, L2LossS2, NormalLossS2
from torch_harmonics.plotting import plot_sphere, imshow_sphere

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
def log_weights_and_grads(exp_dir, model, iters=1):
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
def validate_model(
    model,
    dataloader,
    loss_fn,
    metrics_fns,
    path_root,
    normalization_in=None,
    normalization_out=None,
    logging=True,
    device=torch.device("cpu"),
):

    model.eval()

    num_examples = len(dataloader)

    # make output
    if logging and not os.path.isdir(path_root):
        os.makedirs(path_root, exist_ok=True)

    if dist.is_initialized():
        dist.barrier(device_ids=[device.index])

    losses = torch.zeros(num_examples, dtype=torch.float32, device=device)

    metrics = {}
    for metric in metrics_fns:
        metrics[metric] = torch.zeros(num_examples, dtype=torch.float32, device=device)

    glob_off = 0
    if dist.is_initialized():
        glob_off = num_examples * dist.get_rank()

    with torch.no_grad():
        for idx, (inp, tar) in enumerate(dataloader):
            inpd = inp.to(device)
            tar = tar.to(device)
            mask = torch.where(tar == 0, 0.0, 1.0)

            if normalization_in is not None:
                inpd = normalization_in(inpd)

            if normalization_out is not None:
                tar = normalization_out(tar)

            prd = model(inpd)

            losses[idx] = loss_fn(prd, tar.unsqueeze(-3), mask)

            for metric in metrics_fns:
                metric_buff = metrics[metric]
                metric_fn = metrics_fns[metric]
                metric_buff[idx] = metric_fn(prd, tar, mask)

            tar = (tar * mask).squeeze()
            prd = (prd * mask).squeeze()

            # get the minimum
            vmin = min(tar.min(), prd.min())
            vmax = min(tar.max(), prd.max())

            # do plotting
            glob_idx = idx + glob_off
            fig = plt.figure(figsize=(7.5, 6))
            plot_sphere(prd.cpu(), fig=fig, vmax=vmax, vmin=vmin, cmap="plasma")
            plt.savefig(os.path.join(path_root, "pred_" + str(glob_idx) + ".png"))
            plt.close()

            fig = plt.figure(figsize=(7.5, 6))

            plot_sphere(tar.cpu(), fig=fig, vmax=vmax, vmin=vmin, cmap="plasma")
            plt.savefig(os.path.join(path_root, "truth_" + str(glob_idx) + ".png"))
            plt.close()

            fig = plt.figure(figsize=(7.5, 6))
            imshow_sphere(inp.cpu().squeeze(0).permute(1, 2, 0), fig=fig)
            plt.savefig(os.path.join(path_root, "input_" + str(glob_idx) + ".png"))
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
    normalization_in=None,
    normalization_out=None,
    augmentation=False,
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

        for inp, tar in train_dataloader:

            inp = inp.to(device)
            tar = tar.to(device)
            mask = torch.where(tar == 0, 0.0, 1.0)
            if normalization_in is not None:
                inp = normalization_in(inp)

            if normalization_out is not None:
                tar = normalization_out(tar)

            with torch.autocast(device_type="cuda", dtype=amp_dtype, enabled=(amp_mode != "none")):
                prd = model(inp)
                loss = loss_fn(prd, tar.unsqueeze(-3), mask)

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

        # prepare metrics buffer for accumulation of validation metrics
        valid_metrics = {}
        for metric in metrics_fns:
            valid_metrics[metric] = torch.zeros(2, dtype=torch.float32, device=device)

        model.eval()

        if dist.is_initialized():
            test_sampler.set_epoch(epoch)

        with torch.no_grad():
            for inp, tar in test_dataloader:
                inp = inp.to(device)
                tar = tar.to(device)
                mask = torch.where(tar == 0, 0.0, 1.0)

                if normalization_in is not None:
                    inp = normalization_in(inp)

                if normalization_out is not None:
                    tar = normalization_out(tar)

                with torch.autocast(device_type="cuda", dtype=amp_dtype, enabled=(amp_mode != "none")):
                    prd = model(inp)
                    loss = loss_fn(prd, tar.unsqueeze(-3), mask)

                valid_loss[0] += loss * inp.size(0)
                valid_loss[1] += inp.size(0)

                for metric in metrics_fns:
                    metric_buff = valid_metrics[metric]
                    metric_fn = metrics_fns[metric]
                    metric_buff[0] += metric_fn(prd, tar, mask) * inp.size(0)
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

            if wandb is not None and wandb.run is not None:
                current_lr = optimizer.param_groups[0]["lr"]
                log_dict = {"loss": accumulated_loss, "validation loss": valid_loss, "learning rate": current_lr}
                for metric in valid_metrics:
                    log_dict[metric] = valid_metrics[metric]
                wandb.log(log_dict)

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
    data_downsampling_factor=16,
    exclude_polar_fraction=0.15,
):

    # initialize distributed
    local_rank = 0
    logging = True
    if ddp:
        dist.init_process_group(backend="nccl")
        world_size = dist.get_world_size()
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
    dataset_file = downloader.prepare_dataset(dataset_file=f"stanford_2d3ds_dataset_ds{data_downsampling_factor}.h5", downsampling_factor=data_downsampling_factor)

    # intiialize distributed for ddp
    if dist.is_initialized():
        dist.barrier(device_ids=[device.index])

    # create the dataset and split it
    if logging:
        print(f"Initializing dataset...")

    # make sure splitting is consistent across ranks
    rng = torch.Generator().manual_seed(333)
    split_ratios = [0.95, 0.025, 0.025]
    dataset = StanfordDepthDataset(dataset_file=dataset_file, ignore_alpha_channel=ignore_alpha_channel, log_depth=False, exclude_polar_fraction=exclude_polar_fraction)
    train_dataset, test_dataset, valid_dataset = torch.utils.data.random_split(dataset, split_ratios, generator=rng)

    # stats computation
    means_in, stds_in, means_out, stds_out = compute_stats_s2(train_dataset.dataset, normalize_target=True)

    train_dataset.dataset.reset()
    if logging:
        print(f"Computed stats:")
        print(f"means_in={means_in}")
        print(f"stds_in={stds_in}")
        print(f"means_out={means_out}")
        print(f"stds_out={stds_out}")

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
        train_dataset, batch_size=batch_size, shuffle=True if train_sampler is None else False, sampler=train_sampler, num_workers=4, persistent_workers=True, pin_memory=True
    )

    test_dataloader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, sampler=test_sampler, num_workers=4, persistent_workers=True, pin_memory=True)
    valid_dataloader = DataLoader(valid_dataset, batch_size=1, shuffle=False, sampler=valid_sampler, num_workers=0, persistent_workers=False, pin_memory=True)

    # TODO: move augmentation into extra helper module
    normalization_in = v2.Normalize(mean=means_in.tolist(), std=stds_in.tolist())
    normalization_out = v2.Normalize(mean=means_out.tolist(), std=stds_out.tolist())
    augmentation = enable_data_augmentation

    in_channels = 3 if ignore_alpha_channel else 4
    out_channels = 1

    # print dataset info
    img_size = dataset.input_shape[1:]

    if logging:
        print(f"Train dataset initialized with {len(train_dataset)} samples of resolution {img_size}")
        print(f"Test dataset initialized with {len(test_dataset)} samples of resolution {img_size}")
        print(f"Validation dataset initialized with {len(valid_dataset)} samples of resolution {img_size}")

    # get baseline model registry
    baseline_models = get_baseline_models(img_size=img_size, in_chans=in_channels, out_chans=out_channels)

    # specify which models to train here
    models = [
        "sunet_depth3_e64_k5_pf4",
        # "transformer_sc2_layers4_e128",
        # "s2transformer_sc2_layers4_e128",
        # "ntransformer_sc2_layers4_e128",
        # "s2ntransformer_sc2_layers4_e128",
        # "segformer_sc2_layers4_e128",
        # "s2segformer_sc2_layers4_e128",
        # "nsegformer_sc2_layers4_e128",
        # "s2nsegformer_sc2_layers4_e128",
        # "sfno_sc2_layers4_e32",
        # "lsno_sc2_layers4_e32",
    ]
    models = {k: baseline_models[k] for k in models}

    # initialize Sobolev W11 loss function
    loss_w11 = W11LossS2(nlat=img_size[0], nlon=img_size[1], grid="equiangular").to(device=device)
    loss_l1 = L1LossS2(nlat=img_size[0], nlon=img_size[1], grid="equiangular").to(device=device)
    loss_fn = lambda prd, tar, mask: 0.1 * loss_w11(prd, tar, mask) + loss_l1(prd, tar, mask)

    # metrics
    metrics = {}
    metrics_fns = {
        "L2 error": L2LossS2(nlat=img_size[0], nlon=img_size[1], grid="equiangular").to(device=device),
        "L1 error": L1LossS2(nlat=img_size[0], nlon=img_size[1], grid="equiangular").to(device=device),
        "W11 error": W11LossS2(nlat=img_size[0], nlon=img_size[1], grid="equiangular").to(device=device),
        "Normals error": NormalLossS2(nlat=img_size[0], nlon=img_size[1], grid="equiangular").to(device=device),
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
            if logging and wandb is not None:
                run = wandb.init(project="depth estimation 2d3ds", group=model_name, name=model_name + "_" + str(time.time()), config=model_handle.keywords)
            else:
                run = None

            # optimizer:
            optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=0.01, foreach=torch.cuda.is_available())
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
                normalization_in=normalization_in,
                normalization_out=normalization_out,
                augmentation=None,
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

            # run the validation
            losses, metric_results = validate_model(
                model,
                valid_dataloader,
                loss_fn,
                metrics_fns,
                os.path.join(exp_dir, "figures"),
                normalization_in=normalization_in,
                normalization_out=normalization_out,
                logging=logging,
                device=device,
            )

            # gather losses and metrics into a single tensor
            if dist.is_initialized():
                losses_dist = torch.zeros(world_size * losses.shape[0], dtype=losses.dtype, device=device)
                dist.all_gather_into_tensor(losses_dist, losses)
                losses = losses_dist
                for metric_name, metric in metric_results.items():
                    metric_dist = torch.zeros(world_size * metric.shape[0], dtype=metric.dtype, device=device)
                    dist.all_gather_into_tensor(metric_dist, metric)
                    metric_results[metric_name] = metric_dist

            # compute statistics
            metrics[model_name]["loss mean"] = torch.mean(losses).item()
            metrics[model_name]["loss std"] = torch.std(losses).item()
            for metric in metric_results:
                metrics[model_name][metric + " mean"] = torch.mean(metric_results[metric]).item()
                metrics[model_name][metric + " std"] = torch.std(metric_results[metric]).item()

            if train:
                metrics[model_name]["training_time"] = training_time

    if logging:
        df = pd.DataFrame(metrics)
        if not os.path.isdir(os.path.join(root_path, "output_data")):
            os.makedirs(os.path.join(root_path, "output_data"), exist_ok=True)
        df.to_pickle(os.path.join(root_path, "output_data", "metrics.pkl"))

    if dist.is_initialized():
        dist.barrier(device_ids=[device.index])


if __name__ == "__main__":
    import torch.multiprocessing as mp

    mp.set_start_method("forkserver", force=True)

    if wandb is not None:
        wandb.login()

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--output_path", default=os.path.join(os.path.dirname(__file__), "checkpoints"), type=str, help="Override the path where checkpoints and run information are stored"
    )
    parser.add_argument(
        "--data_path",
        default=os.path.join(os.path.dirname(os.path.dirname(__file__)), "data", "2D3DS"),
        type=str,
        help="Directory to where the dataset is stored. If the dataset is not found in that location, it will be downloaded automatically.",
    )
    parser.add_argument("--num_epochs", default=100, type=int, help="Switch for overriding batch size in the configuration file.")
    parser.add_argument("--batch_size", default=8, type=int, help="Switch for overriding batch size in the configuration file.")
    parser.add_argument("--data_downsampling_factor", default=16, type=int, help="Switch for overriding the downsampling factor of the data.")
    parser.add_argument("--learning_rate", default=1e-3, type=float, help="Switch to override learning rate.")
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
        train=args.num_epochs > 0,
        load_checkpoint=args.resume,
        amp_mode=args.amp_mode,
        ddp=args.enable_ddp,
        enable_data_augmentation=args.enable_data_augmentation,
        ignore_alpha_channel=True,
        log_grads=0,
        data_path=args.data_path,
        data_downsampling_factor=args.data_downsampling_factor,
    )
