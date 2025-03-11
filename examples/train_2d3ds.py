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
sys.path.append("../notebooks")
import time

from tqdm import tqdm
from functools import partial

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

import numpy as np
import pandas as pd

import matplotlib.pyplot as plt

from torch_harmonics.examples import SphericalSegmentationDataset, SphericalSegmendationDatasetDownloader
from torch_harmonics.quadrature import _precompute_latitudes

from plotting import plot_sphere, plot_data, imshow_sphere

# wandb logging
import wandb


class CrossEntropyLossS2(nn.Module):

    def __init__(self, nlat: int, nlon: int, grid: str = "equiangular"):

        super().__init__()

        _, q = _precompute_latitudes(nlat=nlat, grid=grid)

        q = q.reshape(-1, 1) * 2 * torch.pi / nlon

        self.register_buffer("quad_weights", q)

    def forward(self, inp: torch.Tensor, tar: torch.Tensor):

        ce = nn.functional.cross_entropy(inp, tar, reduction="none")
        ce = (ce * self.quad_weights).sum(dim=(-1, -2)) / 4 / torch.pi
        ce = ce.mean()

        return ce


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
def validate_model(model, dataset, loss_fn, path_root, num_examples=10, device=torch.device("cpu")):

    model.eval()

    num_examples = min(len(dataset), num_examples)

    # make output
    if not os.path.isdir(path_root):
        os.makedirs(path_root, exist_ok=True)

    losses = np.zeros(num_examples)
    # fno_times = np.zeros(num_samples)


    for idx in range(num_examples):
        with torch.no_grad():
            inp, tar = dataset[idx]
            prd = model(inp.unsqueeze(0).to(device))
            num_classes = prd.shape[-3]

            losses[idx] = loss_fn(prd, tar.unsqueeze(0).to(device)).item()

            prd = nn.functional.softmax(prd, dim=-3)
            prd = torch.argmax(prd, dim=-3).squeeze()

            # do plotting
            fig = plt.figure(figsize=(7.5, 6))
            plot_data(prd.cpu() / num_classes, fig=fig, vmax=1.0, vmin=0.0, cmap="rainbow")
            plt.savefig(os.path.join(path_root, "pred_" + str(idx) + ".png"))
            plt.close()

            fig = plt.figure(figsize=(7.5, 6))
            plot_data(tar.cpu() / num_classes, fig=fig, vmax=1.0, vmin=0.0, cmap="rainbow")
            plt.savefig(os.path.join(path_root, "truth_" + str(idx) + ".png"))
            plt.close()

    return losses


# training function
def train_model(model, train_dataloader, test_dataloader, loss_fn, optimizer, gscaler, scheduler=None, nepochs=20, enable_amp=False, log_grads=0, device=torch.device("cpu")):


    train_start = time.time()

    # count iterations
    iters = 0

    for epoch in range(nepochs):

        # time each epoch
        epoch_start = time.time()

        # do the training
        accumulated_loss = 0
        model.train()

        for inp, tar in train_dataloader:

            with torch.autocast(device_type="cuda", enabled=enable_amp):

                inp = inp.to(device)
                tar = tar.to(device)

                prd = model(inp)

                loss = loss_fn(prd, tar)

            accumulated_loss += loss.item() * inp.size(0)

            optimizer.zero_grad(set_to_none=True)
            gscaler.scale(loss).backward()

            if log_grads and iters % log_grads == 0:
                log_weights_and_grads(model, iters=iters)

            gscaler.step(optimizer)
            gscaler.update()

            iters += 1

        accumulated_loss = accumulated_loss / len(train_dataloader.dataset)

        # perform validation
        valid_loss = 0
        model.eval()
        with torch.no_grad():
            for inp, tar in test_dataloader:
                inp = inp.to(device)
                tar = tar.to(device)

                prd = model(inp)

                loss = loss_fn(prd, tar)

                valid_loss += loss.item() * inp.size(0)

        valid_loss = valid_loss / len(test_dataloader.dataset)

        if scheduler is not None:
            scheduler.step(valid_loss)

        epoch_time = time.time() - epoch_start

        print(f"--------------------------------------------------------------------------------")
        print(f"Epoch {epoch} summary:")
        print(f"time taken: {epoch_time}")
        print(f"accumulated training loss: {accumulated_loss}")
        print(f"relative validation loss: {valid_loss}")

        if wandb.run is not None:
            current_lr = optimizer.param_groups[0]["lr"]
            wandb.log({"loss": accumulated_loss, "validation loss": valid_loss, "learning rate": current_lr})

    train_time = time.time() - train_start

    print(f"--------------------------------------------------------------------------------")
    print(f"done. Training took {train_time}.")
    return valid_loss


def main(train=True, load_checkpoint=False, enable_amp=False, log_grads=0, _data_dir="data"):

    # directory for outputs
    root_path = os.path.dirname(__file__)

    # set seed
    torch.manual_seed(333)
    torch.cuda.manual_seed(333)

    # set parameters
    nfuture = 0

    # set device
    # device=torch.device("cpu")
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    if torch.cuda.is_available():
        torch.cuda.set_device(device.index)

    data_dir = os.path.join(_data_dir, "2D3DS")
    os.makedirs(data_dir, exist_ok=True)

    # 2D3DS download & dataset initialization
    downloader = SphericalSegmendationDatasetDownloader(base_url="https://cvg-data.inf.ethz.ch/2d3ds/no_xyz/", local_dir=str(data_dir))
    dataset_file = downloader.prepare_dataset()

    # create the dataset and split it
    print(f"Initializing dataset...")
    dataset = SphericalSegmentationDataset(dataset_file=dataset_file)
    train_dataset, test_dataset = torch.utils.data.random_split(dataset, [0.9, 0.1])

    train_dataloader = DataLoader(train_dataset, batch_size=8, shuffle=True, num_workers=0, persistent_workers=False, pin_memory=True)
    test_dataloader = DataLoader(test_dataset, batch_size=8, shuffle=True, num_workers=0, persistent_workers=False, pin_memory=True)

    print(dataset.input_shape)

    img_size = dataset.input_shape[1:]
    print(f"Train dataset initialized with {len(train_dataset)} samples of resolution {img_size}")
    print(f"Test dataset initialized with {len(test_dataset)} samples of resolution {img_size}")

    # prepare dicts containing models and corresponding metrics
    models = {}
    metrics = {}

    from torch_harmonics.examples.models import SphericalFourierNeuralOperatorForSegmentation as SFNO
    from torch_harmonics.examples.models import LocalSphericalNeuralOperatorForSegmentation as LSNO
    from torch_harmonics.examples.models import SphericalTransformerForSegmentation as S2T


    # models[f"sfno_sc4_layers4_e32"] = partial(
    #     SFNO,
    #     dataset.num_classes,
    #     img_size=img_size,
    #     grid="equiangular",
    #     in_chans=4,
    #     num_layers=4,
    #     scale_factor=4,
    #     embed_dim=32,
    #     activation_function="gelu",
    #     residual_prediction=False,
    #     use_mlp=True,
    #     normalization_layer="instance_norm",
    # )

    models[f"lsno_sc4_layers4_e32_morlet"] = partial(
        LSNO,
        dataset.num_classes,
        img_size=img_size,
        grid="equiangular",
        in_chans=4,
        num_layers=4,
        scale_factor=4,
        embed_dim=32,
        activation_function="gelu",
        residual_prediction=False,
        use_mlp=True,
        normalization_layer="instance_norm",
        kernel_shape=(4, 4),
        encoder_kernel_shape=(4, 4),
        filter_basis_type="morlet",
        upsample_sht = True,
    )

    models[f"s2t_sc4_layers4_e32"] = partial(
        S2T,
        dataset.num_classes,
        img_size=img_size,
        grid="equiangular",
        in_chans=4,
        num_layers=4,
        scale_factor=4,
        embed_dim=32,
        activation_function="gelu",
        residual_prediction=False,
        pos_embed="spectral",
        use_mlp=True,
        normalization_layer="instance_norm",
        encoder_kernel_shape=(4, 4),
        filter_basis_type="morlet",
        upsample_sht=True,
    )

    # create the loss object
    loss_fn = CrossEntropyLossS2(nlat=img_size[0], nlon=img_size[1], grid="equiangular").to(device=device)

    # iterate over models and train each model
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

        if load_checkpoint:
            model.load_state_dict(torch.load(os.path.join(exp_dir, "checkpoint.pt")))

        # run the training
        if train:
            run = wandb.init(project="spherical segmentation 2d3ds", group=model_name, name=model_name + "_" + str(time.time()), config=model_handle.keywords)

            # optimizer:
            optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
            scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, "min")
            gscaler = torch.GradScaler("cuda", enabled=enable_amp)

            start_time = time.time()

            print(f"Training {model_name}")
            train_model(model, train_dataloader, test_dataloader, loss_fn, optimizer, gscaler, scheduler, nepochs=20, enable_amp=enable_amp, log_grads=log_grads, device=device)

            training_time = time.time() - start_time

            run.finish()

            torch.save(model.state_dict(), os.path.join(exp_dir, "checkpoint.pt"))

        # set seed
        torch.manual_seed(333)
        torch.cuda.manual_seed(333)

        with torch.inference_mode():
            losses = validate_model(model, test_dataset, loss_fn, os.path.join(exp_dir, "figures"), num_examples=50)
            # metrics[model_name]["loss_mean"] = np.mean(losses)
            # metrics[model_name]["loss_std"] = np.std(losses)
            # metrics[model_name]["fno_time_mean"] = np.mean(fno_times)
            # metrics[model_name]["fno_time_std"] = np.std(fno_times)
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
