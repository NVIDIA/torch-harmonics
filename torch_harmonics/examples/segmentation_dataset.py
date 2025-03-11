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
from torch.utils.data import Dataset
import torch
import numpy as np

# some specifiers where to find the dataset
DEFAULT_BASE_URL = "https://cvg-data.inf.ethz.ch/2d3ds/no_xyz/"
DEFAULT_TAR_FILE_PAIRS = [("area_1_no_xyz.tar", "area_1"), ("area_2_no_xyz.tar", "area_2"), ("area_3_no_xyz.tar", "area_3"), ("area_4_no_xyz.tar", "area_4"), ("area_5a_no_xyz.tar", "area_5a"), ("area_5b_no_xyz.tar", "area_5b"), ("area_6_no_xyz.tar", "area_6"),]
DEFAULT_LABELS_URL = "https://raw.githubusercontent.com/alexsax/2D-3D-Semantics/refs/heads/master/assets/semantic_labels.json"


class SphericalSegmendationDatasetDownloader:
    """
    Convenience class for downloading the 2d3ds dataset [1].

    References
    -----------
    .. [1] Armeni, I.,  Sax, S.,  Zamir, A. R.,  Savarese, S.;
        "Joint 2D-3D-Semantic Data for Indoor Scene Understanding" (2017).
        https://arxiv.org/abs/1702.01105.
    """

    def __init__(self, base_url: str = DEFAULT_BASE_URL, local_dir: str = "data"):

        self.base_url = base_url
        self.local_dir = local_dir
        os.makedirs(self.local_dir, exist_ok=True)

    def _download_file(self, filename):

        import requests
        from tqdm import tqdm

        url = f"{self.base_url}/{filename}"
        local_path = os.path.join(self.local_dir, filename)
        if os.path.exists(local_path):
            print(f"Note: Skipping download for {filename}, because it already exists")
            return local_path

        print(f"Downloading {filename}...")
        temp_path = local_path.split(".")[0] + ".part"

        # Resume logic
        headers = {}
        if os.path.exists(temp_path):
            headers = {"Range": f"bytes={os.stat(temp_path).st_size}-"}

        response = requests.get(url, headers=headers, stream=True, timeout=30)
        if os.path.exists(temp_path):
            total_size = int(response.headers.get("content-length", 0)) + os.stat(temp_path).st_size
        else:
            total_size = int(response.headers.get("content-length", 0))

        with open(temp_path, "ab") as f, tqdm(desc=filename, total=total_size, unit="B", unit_scale=True, unit_divisor=1024, initial=os.stat(temp_path).st_size) as pbar:
            for chunk in response.iter_content(chunk_size=1024):
                if chunk:
                    f.write(chunk)
                    pbar.update(len(chunk))

        os.rename(temp_path, local_path)
        return local_path

    def _extract_tar(self, tar_path):
        import tarfile

        with tarfile.open(tar_path) as tar:
            tar.extractall(path=self.local_dir)
            tar_filenames = tar.getnames()
            extracted_dir = tar_filenames[0]
            os.remove(tar_path)
            return extracted_dir

    def download_dataset(self, file_extracted_directory_pairs=DEFAULT_TAR_FILE_PAIRS):

        import requests

        data_folders = []
        for file, extracted_folder_name in file_extracted_directory_pairs:
            if not os.path.exists(os.path.join(self.local_dir, extracted_folder_name)):
                downloaded_file = self._download_file(file)
                data_folders.append(self._extract_tar(downloaded_file))
            else:
                print(f"Warning: Skipping D/L for '{file}' because folder '{extracted_folder_name}' already exists")
                data_folders.append(extracted_folder_name)

        labels_json_url = DEFAULT_LABELS_URL
        class_labels = requests.get(labels_json_url).json()
        return data_folders, class_labels

    def _rgb_to_id(self, img):
        return img[..., 0] * 256 * 256 + img[..., 1] * 256 + img[..., 2]

    def convert_dataset(
        self,
        data_folders,
        class_labels,
        input_path: str = "pano/rgb",
        output_path: str = "pano/semantic",
        output_filename="semantic",
        dataset_file: str = "segmentation_dataset.h5",
        downsampling_factor: int = 16,
    ):

        converted_dataset_path = os.path.join(self.local_dir, dataset_file)

        from PIL import Image
        from tqdm import tqdm
        import h5py as h5

        file_paths = []

        min_vals = None
        max_vals = None

        # get all the file path input, output pairs
        for base_path in data_folders:

            input_dir = os.path.join(self.local_dir, base_path, input_path)
            output_dir = os.path.join(self.local_dir, base_path, output_path)

            if os.path.exists(input_dir) and os.path.exists(output_dir):
                for file_input in os.listdir(input_dir):
                    if not file_input.endswith(".png"):
                        continue
                    input_filepath = os.path.join(input_dir, file_input)
                    output_filepath = "_".join(os.path.splitext(os.path.basename(input_filepath))[0].split("_")[:-1]) + f"_{output_filename}.png"
                    output_filepath = os.path.join(output_dir, output_filepath)
                    if not os.path.exists(output_filepath):
                        print(f"Warning: Couldn't find output file in pair: ({input_filepath},{output_filepath})")
                        continue
                    file_paths.append((input_filepath, output_filepath))
            elif not os.path.exists(input_dir):
                print("Warning: Input dir doesn't exist: ", input_dir)
                continue
            elif not os.path.exists(output_dir):
                print("Warning: Output dir doesn't exist: ", output_dir)
                continue

        num_samples = len(file_paths)

        if num_samples > 0:
            first_inp, first_tar = file_paths[0]
            first_inp = np.array(Image.open(first_inp))
            first_tar = np.array(Image.open(first_tar))

            inp_shape = first_inp.shape
            img_shape = (inp_shape[0] // downsampling_factor, inp_shape[1] // downsampling_factor)
            inp_channels = inp_shape[2]

        else:
            raise ValueError(f"No samples found")

        # create the dataset file
        with h5.File(converted_dataset_path, "w") as h5file:
            input_data = h5file.create_dataset("inputs", (num_samples, inp_channels, *img_shape), "f4")
            target_data = h5file.create_dataset("targets", (num_samples, *img_shape), "i8")
            classes = h5file.create_dataset("class_labels", data=class_labels)

            for i in tqdm(range(num_samples), desc="preparing dataset"):
                inp_data = np.array(Image.open(file_paths[i][0]))[::downsampling_factor, ::downsampling_factor] / 255.0
                inp_data = np.transpose(inp_data, axes=(2,0,1))
                input_data[i, ...] = inp_data[...]

                # compute minimum and maximum over the dataset
                tmp_min = np.min(inp_data, axis=(1,2))
                tmp_max = np.max(inp_data, axis=(1,2))
                if i == 0:
                    min_vals = tmp_min
                    max_vals = tmp_max
                else:
                    min_vals = np.minimum(min_vals, tmp_min)
                    max_vals = np.minimum(max_vals, tmp_max)

                tar_data = np.array(Image.open(file_paths[i][1]))[::downsampling_factor, ::downsampling_factor]
                target_data[i, ...] = self._rgb_to_id(tar_data)[...]

            h5file.create_dataset("min", data=min_vals.astype(np.float32))
            h5file.create_dataset("max", data=max_vals.astype(np.float32))

        return converted_dataset_path

    def prepare_dataset(self, file_extracted_directory_pairs=DEFAULT_TAR_FILE_PAIRS, dataset_file: str = "segmentation_dataset.h5"):

        converted_dataset_path = os.path.join(self.local_dir, dataset_file)
        if os.path.exists(converted_dataset_path):
            print(
                f"Dataset file at {converted_dataset_path} already exists. Skipping download and conversion. If you want to create a new dataset file, delete or rename the existing file."
            )
            return converted_dataset_path

        data_folders, class_labels = self.download_dataset(file_extracted_directory_pairs=file_extracted_directory_pairs)
        converted_dataset_path = self.convert_dataset(data_folders=data_folders, class_labels=class_labels, dataset_file=dataset_file)

        return converted_dataset_path


class SphericalSegmentationDataset(Dataset):
    """
    Spherical segmentation dataset from [1].

    References
    -----------
    .. [1] Armeni, I.,  Sax, S.,  Zamir, A. R.,  Savarese, S.;
        "Joint 2D-3D-Semantic Data for Indoor Scene Understanding" (2017).
        https://arxiv.org/abs/1702.01105.
    """

    def __init__(
        self,
        dataset_file,
    ):

        self.dataset_file = dataset_file

        # open file and check for
        self.h5file = None
        self.class_labels = None
        self.inputs = None
        self.targets = None
        self.current_buffer = 0
        self.inp_buffers = None
        self.tar_buffers = None


    @property
    def target_shape(self):
        if self.h5file is None:
            self._init_files()

        return self.img_seg

    @property
    def input_shape(self):
        if self.h5file is None:
            self._init_files()

        return self.img_rgb

    @property
    def num_classes(self):
        return len(self.class_labels)

    def _id_to_class(self, class_id):
        if class_id > self.num_classes:
            print("WARNING: ID > number of classes!")
            return None
        return self.segmentation_classes[class_id]

    def _mask_invalid(self, tar):
        return np.where(tar >= self.num_classes, -100, tar)

    def __len__(self):
        if self.h5file is None:
            self._init_files()

        return self.inputs.shape[0]

    def _init_files(self):
        import h5py as h5

        self.h5file = h5.File(self.dataset_file, "r")
        self.class_labels = self.h5file["class_labels"]
        self.inputs = self.h5file["inputs"]
        self.targets = self.h5file["targets"]

        self.img_rgb = self.inputs[0].shape
        self.img_seg = self.targets[0].shape

    def __getitem__(self, idx, mask_invalid=True):

        if self.h5file is None:
            # init files
            self._init_files()

        # init buffers
        if self.inp_buffers is None:
            self.inp_buffers = [np.empty(self.img_rgb, dtype=np.float32), np.empty(self.img_rgb, dtype=np.float32)]
        if self.tar_buffers is None:
            self.tar_buffers = [np.empty(self.img_seg, dtype=np.int64), np.empty(self.img_seg, dtype=np.int64)]

        # double buffering
        inp = self.inp_buffers[self.current_buffer]
        tar = self.tar_buffers[self.current_buffer]

        self.current_buffer = (self.current_buffer + 1) % 2

        self.inputs.read_direct(inp, np.s_[idx:idx+1, 0:self.input_shape[0], 0:self.input_shape[1], 0:self.input_shape[2]])
        self.targets.read_direct(tar, np.s_[idx:idx+1, 0:self.target_shape[0], 0:self.target_shape[1]])

        if mask_invalid:
            tar = self._mask_invalid(tar)

        return inp, tar
