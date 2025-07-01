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
import math

import torch
from torch.utils.data import Dataset, DataLoader, Subset

import numpy as np

from torch_harmonics.quadrature import _precompute_latitudes
from torch_harmonics.examples.losses import get_quadrature_weights

# some specifiers where to find the dataset
DEFAULT_BASE_URL = "https://cvg-data.inf.ethz.ch/2d3ds/no_xyz/"
DEFAULT_TAR_FILE_PAIRS = [
    ("area_1_no_xyz.tar", "area_1"),
    ("area_2_no_xyz.tar", "area_2"),
    ("area_3_no_xyz.tar", "area_3"),
    ("area_4_no_xyz.tar", "area_4"),
    ("area_5a_no_xyz.tar", "area_5a"),
    ("area_5b_no_xyz.tar", "area_5b"),
    ("area_6_no_xyz.tar", "area_6"),
]
DEFAULT_LABELS_URL = "https://raw.githubusercontent.com/alexsax/2D-3D-Semantics/refs/heads/master/assets/semantic_labels.json"


class Stanford2D3DSDownloader:
    """
    Convenience class for downloading the 2d3ds dataset [1].

    Parameters
    ----------
    base_url : str, optional
        Base URL for downloading the dataset, by default DEFAULT_BASE_URL
    local_dir : str, optional
        Local directory to store downloaded files, by default "data"

    Returns
    -------
    data_folders : list
        List of extracted directory names
    class_labels : list
        List of semantic class labels

    References
    ----------
    .. [1] Armeni, I.,  Sax, S.,  Zamir, A. R.,  Savarese, S.;
        "Joint 2D-3D-Semantic Data for Indoor Scene Understanding" (2017).
        https://arxiv.org/abs/1702.01105.
    """

    def __init__(self, base_url: str = DEFAULT_BASE_URL, local_dir: str = "data"):
        """
        Initialize the Stanford 2D3DS dataset downloader.
        
        Parameters
        -----------
        base_url : str, optional
            Base URL for downloading the dataset, by default DEFAULT_BASE_URL
        local_dir : str, optional
            Local directory to store downloaded files, by default "data"
        """
        self.base_url = base_url
        self.local_dir = local_dir
        os.makedirs(self.local_dir, exist_ok=True)

    def _download_file(self, filename):
        """
        Download a single file with progress bar and resume capability.
        
        Parameters
        -----------
        filename : str
            Name of the file to download
            
        Returns
        -------
        str
            Local path to the downloaded file
        """
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
        """
        Extract a tar file and return the extracted directory name.
        
        Parameters
        -----------
        tar_path : str
            Path to the tar file to extract
            
        Returns
        -------
        str
            Name of the extracted directory
        """
        import tarfile

        with tarfile.open(tar_path) as tar:
            tar.extractall(path=self.local_dir)
            tar_filenames = tar.getnames()
            extracted_dir = tar_filenames[0]
            os.remove(tar_path)
            return extracted_dir

    def download_dataset(self, file_extracted_directory_pairs=DEFAULT_TAR_FILE_PAIRS):
        """
        Download and extract the complete dataset.
        
        Parameters
        -----------
        file_extracted_directory_pairs : list, optional
            List of (filename, extracted_folder_name) pairs, by default DEFAULT_TAR_FILE_PAIRS
            
        Returns
        -------
        tuple
            (data_folders, class_labels) where data_folders is a list of extracted directory names
            and class_labels is the semantic label mapping
        """
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

    def _rgb_to_id(self, img, class_labels_map, class_labels_indices):
        """
        Convert RGB image to class ID using color mapping.
        
        Parameters
        -----------
        img : numpy.ndarray
            RGB image array
        class_labels_map : list
            Mapping from color values to class labels
        class_labels_indices : list
            List of class label indices
            
        Returns
        -------
        numpy.ndarray
            Class ID array
        """
        # Convert to int32 first to avoid overflow
        r = img[..., 0].astype(np.int32)
        g = img[..., 1].astype(np.int32)
        b = img[..., 2].astype(np.int32)
        lookup_indices = r * 256 * 256 + g * 256 + b

        def _convert(lookup: int) -> int:
            # the dataset has a bad label for clutter, so we need to fix it
            # clutter is 855309, but the labels file has it as 3341
            # The original conversion used uint8, which overflowed the clutter label to 3341
            # this is a fix to handle that accidental usage of undefined overflow behavior
            if lookup == 855309:
                label = class_labels_map[3341]  # clutter
            else:
                label = class_labels_map[lookup]
            class_index = class_labels_indices.index(label)
            return class_index

        lookup_fn = np.vectorize(_convert)

        return lookup_fn(lookup_indices)

    def convert_dataset(
        self,
        data_folders,
        class_labels,
        rgb_path: str = "pano/rgb",
        semantic_path: str = "pano/semantic",
        depth_path: str = "pano/depth",
        output_filename="semantic",
        dataset_file: str = "stanford_2d3ds_dataset.h5",
        downsampling_factor: int = 16,
        remove_alpha_channel: bool = True,
    ):
        """
        Convert the downloaded dataset to HDF5 format for efficient loading.
        
        Parameters
        -----------
        data_folders : list
            List of extracted data folder names
        class_labels : list
            List of semantic class labels
        rgb_path : str, optional
            Relative path to RGB images within each data folder, by default "pano/rgb"
        semantic_path : str, optional
            Relative path to semantic labels within each data folder, by default "pano/semantic"
        depth_path : str, optional
            Relative path to depth images within each data folder, by default "pano/depth"
        output_filename : str, optional
            Suffix for semantic label files, by default "semantic"
        dataset_file : str, optional
            Output HDF5 filename, by default "stanford_2d3ds_dataset.h5"
        downsampling_factor : int, optional
            Factor by which to downsample images, by default 16
        remove_alpha_channel : bool, optional
            Whether to remove alpha channel from RGB images, by default True
            
        Returns
        -------
        str
            Path to the created HDF5 dataset file
        """
        converted_dataset_path = os.path.join(self.local_dir, dataset_file)

        from PIL import Image
        from tqdm import tqdm
        import h5py as h5

        file_paths = []

        min_vals = None
        max_vals = None

        # condition class labels first:
        class_labels_map = [label.split("_")[0] for label in class_labels]
        class_labels_indices = sorted(list(set(class_labels_map)))

        # get all the file path input, output pairs
        for base_path in data_folders:

            rgb_dir = os.path.join(self.local_dir, base_path, rgb_path)
            semantic_dir = os.path.join(self.local_dir, base_path, semantic_path)
            depth_dir = os.path.join(self.local_dir, base_path, depth_path)

            if os.path.exists(rgb_dir) and os.path.exists(semantic_dir) and os.path.exists(depth_dir):
                for file_input in os.listdir(rgb_dir):
                    if not file_input.endswith(".png"):
                        continue
                    rgb_filepath = os.path.join(rgb_dir, file_input)
                    semantic_filepath = "_".join(os.path.splitext(os.path.basename(rgb_filepath))[0].split("_")[:-1]) + f"_{output_filename}.png"
                    semantic_filepath = os.path.join(semantic_dir, semantic_filepath)
                    depth_filepath = "_".join(os.path.splitext(os.path.basename(rgb_filepath))[0].split("_")[:-1]) + f"_depth.png"
                    depth_filepath = os.path.join(depth_dir, depth_filepath)
                    if not os.path.exists(semantic_filepath):
                        print(f"Warning: Couldn't find output file in pair: ({rgb_filepath},{semantic_filepath})")
                        continue

                    if not os.path.exists(depth_filepath):
                        print(f"Warning: Couldn't find depth file in pair: ({rgb_filepath},{depth_filepath})")
                        continue

                    file_paths.append((rgb_filepath, semantic_filepath, depth_filepath))
            elif not os.path.exists(rgb_dir):
                print("Warning: RGB dir doesn't exist: ", rgb_dir)
                continue
            elif not os.path.exists(semantic_dir):
                print("Warning: Semantic dir doesn't exist: ", semantic_dir)
                continue
            elif not os.path.exists(depth_dir):
                print("Warning: Depth dir doesn't exist: ", depth_dir)
                continue

        num_samples = len(file_paths)

        if num_samples > 0:
            first_rgb, first_semantic, first_depth = file_paths[0]
            first_rgb = np.array(Image.open(first_rgb))
            # first_semantic = np.array(Image.open(first_semantic))
            # first_depth = np.array(Image.open(first_depth))

            rgb_shape = first_rgb.shape
            img_shape = (rgb_shape[0] // downsampling_factor, rgb_shape[1] // downsampling_factor)
            rgb_channels = rgb_shape[2]

            if remove_alpha_channel:
                rgb_channels = 3
        else:
            raise ValueError(f"No samples found")

        # create the dataset file
        with h5.File(converted_dataset_path, "w") as h5file:
            rgb_data = h5file.create_dataset("rgb", (num_samples, rgb_channels, *img_shape), "f4")
            semantic_data = h5file.create_dataset("semantic", (num_samples, *img_shape), "i8")
            depth_data = h5file.create_dataset("depth", (num_samples, *img_shape), "f4")
            classes = h5file.create_dataset("class_labels", data=class_labels_indices)
            num_classes = len(set(class_labels_indices))
            data_source_path = h5file.create_dataset("data_source_path", (num_samples,), dtype=h5.string_dtype(encoding="utf-8"))
            data_target_path = h5file.create_dataset("data_target_path", (num_samples,), dtype=h5.string_dtype(encoding="utf-8"))

            # prepare computation of the class histogram
            class_histogram = np.zeros(num_classes)
            _, quad_weights = _precompute_latitudes(nlat=img_shape[0], grid="equiangular")
            quad_weights = quad_weights.reshape(-1, 1) * 2 * torch.pi / float(img_shape[1])
            quad_weights = quad_weights.tile(1, img_shape[1])
            quad_weights /= torch.sum(quad_weights)
            quad_weights = quad_weights.numpy()

            for count in tqdm(range(num_samples), desc="preparing dataset"):
                # open image
                img = Image.open(file_paths[count][0])

                # downsampling
                if downsampling_factor != 1:
                    # first width, then weight, weird
                    img = img.resize(size=(img_shape[1], img_shape[0]), resample=Image.BILINEAR)

                # remove alpha channel if requested
                if remove_alpha_channel:
                    img = img.convert("RGBA")
                    background = Image.new("RGBA", img.size, (255, 255, 255))
                    # compoe foreground and background and remove alpha channel
                    img = np.array(Image.alpha_composite(background, img))
                    r_data = img[:, :, :3]
                else:
                    r_data = np.array(img)

                # transpose to channels first
                r_data = np.transpose(r_data / 255.0, axes=(2, 0, 1))

                # write to disk
                rgb_data[count, ...] = r_data[...]
                data_source_path[count] = file_paths[count][0]

                # compute stats -> segmentation
                # min/max
                tmp_min = np.min(r_data, axis=(1, 2))
                tmp_max = np.max(r_data, axis=(1, 2))
                # mean/var
                tmp_mean = np.sum(r_data * quad_weights[np.newaxis, :, :], axis=(1, 2))
                tmp_m2 = np.sum(np.square(r_data - tmp_mean[:, np.newaxis, np.newaxis]) * quad_weights[np.newaxis, :, :])
                if count == 0:
                    # min/max
                    min_vals = tmp_min
                    max_vals = tmp_max
                    # mean/var
                    mean_vals = tmp_mean
                    m2_vals = tmp_m2
                else:
                    # min/max
                    min_vals = np.minimum(min_vals, tmp_min)
                    max_vals = np.minimum(max_vals, tmp_max)
                    # mean/var
                    delta = tmp_mean - mean_vals
                    mean_vals += delta / float(count + 1)
                    m2_vals += tmp_m2 + delta * delta * float(count / (count + 1))

                # get the target
                sem = Image.open(file_paths[count][1])

                # downsampling
                if downsampling_factor != 1:
                    sem = sem.resize(size=(img_shape[1], img_shape[0]), resample=Image.NEAREST)

                sem_data = np.array(sem, dtype=np.uint32)

                # map to classes
                sem_data = self._rgb_to_id(sem_data, class_labels_map, class_labels_indices)

                # write to file
                semantic_data[count, ...] = sem_data[...]
                data_target_path[count] = file_paths[count][1]

                # Here we want depth
                dep = Image.open(file_paths[count][2])

                if downsampling_factor != 1:
                    dep = dep.resize(size=(img_shape[1], img_shape[0]), resample=Image.NEAREST)
                dep_data = np.array(dep)

                depth_data[count, ...] = dep_data[...] / 65536.0

                # compute stats -> depth
                # min/max
                tmp_min_depth = np.min(dep_data, axis=(0, 1))
                tmp_max_depth = np.max(dep_data, axis=(0, 1))
                # mean/var
                tmp_mean_depth = np.sum(dep_data * quad_weights[:, :])
                tmp_m2_depth = np.sum(np.square(dep_data - tmp_mean_depth) * quad_weights[:, :])
                if count == 0:
                    min_vals_depth = tmp_min_depth
                    max_vals_depth = tmp_max_depth
                    mean_vals_depth = tmp_mean_depth
                    m2_vals_depth = tmp_m2_depth
                else:
                    min_vals_depth = np.minimum(min_vals_depth, tmp_min_depth)
                    max_vals_depth = np.minimum(max_vals_depth, tmp_max_depth)
                    delta = tmp_mean_depth - mean_vals_depth
                    mean_vals_depth += delta / float(count + 1)
                    m2_vals_depth += tmp_m2_depth + delta * delta * float(count / (count + 1))

                # update the class histogram
                for c in range(num_classes):
                    class_histogram[c] += quad_weights[sem_data == c].sum()

            # record min/max
            h5file.create_dataset("min_rgb", data=min_vals.astype(np.float32))
            h5file.create_dataset("max_rgb", data=max_vals.astype(np.float32))
            h5file.create_dataset("mean_rgb", data=mean_vals.astype(np.float32))
            std_vals = np.sqrt(m2_vals / float(num_samples - 1))
            h5file.create_dataset("std_rgb", data=std_vals.astype(np.float32))

            # record min/max
            h5file.create_dataset("min_depth", data=min_vals_depth.astype(np.float32))
            h5file.create_dataset("max_depth", data=max_vals_depth.astype(np.float32))
            h5file.create_dataset("mean_depth", data=mean_vals_depth.astype(np.float32))
            std_vals_depth = np.sqrt(m2_vals_depth / float(num_samples - 1))
            h5file.create_dataset("std_depth", data=std_vals_depth.astype(np.float32))

            # record class histogram
            class_histogram = class_histogram / num_samples
            h5file.create_dataset("class_histogram", data=class_histogram.astype(np.float32))

        return converted_dataset_path

    def prepare_dataset(self, file_extracted_directory_pairs=DEFAULT_TAR_FILE_PAIRS, dataset_file: str = "stanford_2d3ds_dataset.h5", downsampling_factor: int = 16):

        converted_dataset_path = os.path.join(self.local_dir, dataset_file)
        if os.path.exists(converted_dataset_path):
            print(
                f"Dataset file at {converted_dataset_path} already exists. Skipping download and conversion. If you want to create a new dataset file, delete or rename the existing file."
            )
            return converted_dataset_path

        data_folders, class_labels = self.download_dataset(file_extracted_directory_pairs=file_extracted_directory_pairs)
        converted_dataset_path = self.convert_dataset(data_folders=data_folders, class_labels=class_labels, dataset_file=dataset_file, downsampling_factor=downsampling_factor)

        self.converted_dataset_path = converted_dataset_path

        return self.converted_dataset_path


class StanfordSegmentationDataset(Dataset):
    """
    Spherical segmentation dataset from [1].

    Parameters
    ----------
    dataset_file : str
        Path to the HDF5 dataset file
    ignore_alpha_channel : bool, optional
        Whether to ignore the alpha channel in the RGB images, by default True
    log_depth : bool, optional
        Whether to log the depth values, by default False
    exclude_polar_fraction : float, optional
        Fraction of polar points to exclude, by default 0.0

    Returns
    -------
    StanfordSegmentationDataset
        Dataset object

    References
    ----------
    .. [1] Armeni, I.,  Sax, S.,  Zamir, A. R.,  Savarese, S.;
        "Joint 2D-3D-Semantic Data for Indoor Scene Understanding" (2017).
        https://arxiv.org/abs/1702.01105.
    """

    def __init__(
        self,
        dataset_file,
        ignore_alpha_channel=True,
        exclude_polar_fraction=0,
    ):

        import h5py as h5

        self.dataset_file = dataset_file
        self.exclude_polar_fraction = exclude_polar_fraction

        with h5.File(self.dataset_file, "r") as h5file:
            self.img_rgb = h5file["rgb"][0].shape
            self.img_seg = h5file["semantic"][0].shape
            self.num_samples = h5file["rgb"].shape[0]
            self.num_classes = h5file["class_labels"].shape[0]

            self.class_labels = [class_name.decode("utf-8") for class_name in h5file["class_labels"][...].tolist()]
            self.class_histogram = np.array(h5file["class_histogram"][...])
            self.class_histogram = self.class_histogram / self.class_histogram.sum()

            self.mean = h5file["mean_rgb"][...]
            self.std = h5file["std_rgb"][...]
            self.min = h5file["min_rgb"][...]
            self.max = h5file["max_rgb"][...]

            self.img_filepath = h5file["data_source_path"][...]
            self.tar_filepath = h5file["data_target_path"][...]

        if ignore_alpha_channel:
            self.img_rgb = (3, self.img_rgb[1], self.img_rgb[2])

        # open file and check for
        self.h5file = None
        self.rgb = None
        self.semantic = None

        # return index set to false by default
        # when true, the __getitem__ method will return the index of the input,target pair
        self.return_index = False

    @property
    def target_shape(self):
        return self.img_seg

    @property
    def input_shape(self):
        return self.img_rgb

    def set_return_index(self, return_index: bool):
        self.return_index = return_index

    def get_img_filepath(self, idx: int):
        return self.img_filepath[idx]

    def get_tar_filepath(self, idx: int):
        return self.tar_filepath[idx]

    def _id_to_class(self, class_id):
        if class_id > self.num_classes:
            print("WARNING: ID > number of classes!")
            return None
        return self.segmentation_classes[class_id]

    def _mask_invalid(self, tar):
        return np.where(tar >= self.num_classes, -100, tar)

    def __len__(self):
        return self.num_samples

    def _init_files(self):
        import h5py as h5

        self.h5file = h5.File(self.dataset_file, "r")
        self.rgb = self.h5file["rgb"]
        self.semantic = self.h5file["semantic"]

    def reset(self):
        self.rgb = None
        self.semantic = None
        if self.h5file is not None:
            self.h5file.close()
            del self.h5file
        self.h5file = None

    def __getitem__(self, idx, mask_invalid=True):

        if self.h5file is None:
            # init files
            self._init_files()

        rgb = self.rgb[idx, 0 : self.img_rgb[0], 0 : self.img_rgb[1], 0 : self.img_rgb[2]]
        sem = self.semantic[idx, 0 : self.img_seg[0], 0 : self.img_seg[1]]
        if mask_invalid:
            sem = self._mask_invalid(sem)

        if self.exclude_polar_fraction > 0:
            hcut = int(self.exclude_polar_fraction * sem.shape[0])
            if hcut > 0:
                sem[0:hcut, :] = -100
                sem[-hcut:, :] = -100

        return rgb, sem


class StanfordDatasetSubset(Subset):
    def __init__(self, dataset, indices, return_index=False):
        super().__init__(dataset, indices)
        self.return_index = return_index
        self.dataset = dataset

    def set_return_index(self, value):
        self.return_index = value

    def __getitem__(self, index):
        real_index = self.indices[index]
        data = self.dataset[real_index]

        if self.return_index:
            return data[0], data[1], real_index
        else:
            # Otherwise, return only (data, target)
            return data[0], data[1]


class StanfordDepthDataset(Dataset):
    """
    Spherical segmentation dataset from [1].

    Parameters
    ----------
    dataset_file : str
        Path to the HDF5 dataset file
    ignore_alpha_channel : bool, optional
        Whether to ignore the alpha channel in the RGB images, by default True
    log_depth : bool, optional
        Whether to log the depth values, by default False
    exclude_polar_fraction : float, optional
        Fraction of polar points to exclude, by default 0.0

    References
    ----------
    .. [1] Armeni, I.,  Sax, S.,  Zamir, A. R.,  Savarese, S.;
        "Joint 2D-3D-Semantic Data for Indoor Scene Understanding" (2017).
        https://arxiv.org/abs/1702.01105.
    """

    def __init__(self, dataset_file, ignore_alpha_channel=True, log_depth=False, exclude_polar_fraction=0.0):

        import h5py as h5

        self.dataset_file = dataset_file
        self.log_depth = log_depth
        self.exclude_polar_fraction = exclude_polar_fraction
        with h5.File(self.dataset_file, "r") as h5file:
            self.img_rgb = h5file["rgb"][0].shape
            self.img_depth = h5file["depth"][0].shape
            self.num_samples = h5file["rgb"].shape[0]

            self.mean_in = h5file["mean_rgb"][...]
            self.std_in = h5file["std_rgb"][...]
            self.min_in = h5file["min_rgb"][...]
            self.max_in = h5file["max_rgb"][...]

            self.mean_out = h5file["mean_depth"][...]
            self.std_out = h5file["std_depth"][...]
            self.min_out = h5file["min_depth"][...]
            self.max_out = h5file["max_depth"][...]

        if ignore_alpha_channel:
            self.img_rgb = (3, self.img_rgb[1], self.img_rgb[2])

        # open file and check for
        self.h5file = None
        self.rgb = None
        self.depth = None

    @property
    def target_shape(self):
        return self.img_depth

    @property
    def input_shape(self):
        return self.img_rgb

    def __len__(self):
        return self.num_samples

    def _init_files(self):
        import h5py as h5

        self.h5file = h5.File(self.dataset_file, "r")
        self.rgb = self.h5file["rgb"]
        self.depth = self.h5file["depth"]

    def reset(self):
        self.rgb = None
        self.depth = None
        if self.h5file is not None:
            self.h5file.close()
            del self.h5file
        self.h5file = None

    def _mask_invalid(self, tar):
        return tar * np.where(tar == tar.max(), 0, 1)

    def __getitem__(self, idx, mask_invalid=True):

        if self.h5file is None:
            # init files
            self._init_files()

        rgb = self.rgb[idx, 0 : self.img_rgb[0], 0 : self.img_rgb[1], 0 : self.img_rgb[2]]

        depth = self.depth[idx, 0 : self.img_depth[0], 0 : self.img_depth[1]]
        if mask_invalid:
            depth = self._mask_invalid(depth)

        if self.exclude_polar_fraction > 0:
            hcut = int(self.exclude_polar_fraction * depth.shape[0])
            if hcut > 0:
                depth[0:hcut, :] = 0
                depth[-hcut:, :] = 0

        if self.log_depth:
            depth = np.log(1 + depth)

        return rgb, depth


def compute_stats_s2(dataset: Dataset, normalize_target: bool = False):
    """
    Compute stats using parallel welford reduction and quadrature on the sphere. The parallel welford reduction follows this article (parallel algorithm): https://en.wikipedia.org/wiki/Algorithms_for_calculating_variance
    """

    nexamples = len(dataset)
    count = 0
    for isample in range(nexamples):
        token = dataset[isample]

        # dimension of inp and tar are (3, nlat, nlon)
        inp, tar = token
        nlat = tar.shape[-2]
        nlon = tar.shape[-1]

        # pre-compute quadrature weights
        if isample == 0:
            quad_weights = get_quadrature_weights(nlat=inp.shape[1], nlon=inp.shape[2], grid="equiangular", tile=True).numpy().astype(np.float64)

        # this is a special case for the depth dataset
        # TODO: maybe make this an argument
        if normalize_target:
            mask = np.where(tar == 0, 0, 1)
            masked_area = np.sum(mask * quad_weights[np.newaxis, :, :], axis=(-2, -1))

        # get initial welford values
        if isample == 0:
            # input
            inp_means = np.sum(inp * quad_weights[np.newaxis, :, :], axis=(-2, -1))
            inp_m2s = np.sum(np.square(inp - inp_means[:, np.newaxis, np.newaxis]) * quad_weights[np.newaxis, :, :], axis=(-2, -1))

            # target
            if normalize_target:
                tar_means = np.sum(mask * tar * quad_weights[np.newaxis, :, :], axis=(-2, -1)) / masked_area
                tar_m2s = np.sum(mask * np.square(tar - tar_means[:, np.newaxis, np.newaxis]) * quad_weights[np.newaxis, :, :], axis=(-2, -1)) / masked_area

            # update count
            count = 1

        # do welford update
        else:
            # input
            # get new mean and m2
            inp_mean = np.sum(inp * quad_weights[np.newaxis, :, :], axis=(-2, -1))
            inp_m2 = np.sum(np.square(inp - inp_mean[:, np.newaxis, np.newaxis]) * quad_weights[np.newaxis, :, :], axis=(-2, -1))
            # update welford values
            inp_delta = inp_mean - inp_means
            inp_m2s = inp_m2s + inp_m2 + inp_delta**2 * count / float(count + 1)
            inp_means = inp_means + inp_delta / float(count + 1)

            # target
            if normalize_target:
                # get new mean and m2
                tar_mean = np.sum(mask * tar * quad_weights[np.newaxis, :, :], axis=(-2, -1)) / masked_area
                tar_m2 = np.sum(mask * np.square(tar - tar_mean[:, np.newaxis, np.newaxis]) * quad_weights[np.newaxis, :, :], axis=(-2, -1)) / masked_area
                # update welford values
                tar_delta = tar_mean - tar_means
                tar_m2s = tar_m2s + tar_m2 + tar_delta**2 * count / float(count + 1)
                tar_means = tar_means + tar_delta / float(count + 1)

            # update count
            count += 1

    # finalize
    inp_stds = np.sqrt(inp_m2s / float(count))
    result = (inp_means.astype(np.float32), inp_stds.astype(np.float32))

    if normalize_target:
        tar_stds = np.sqrt(tar_m2s / float(count))
        result += (tar_means.astype(np.float32), tar_stds.astype(np.float32))

    return result
