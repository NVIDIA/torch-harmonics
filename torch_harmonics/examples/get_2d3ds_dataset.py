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
import tarfile
import requests
from tqdm import tqdm
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import numpy as np
from pathlib import Path

class TarDownloader:
    def __init__(self, base_url, local_dir="data"):
        self.base_url = base_url
        self.local_dir = Path(local_dir)
        self.local_dir.mkdir(parents=True, exist_ok=True)

    # def _extract_tar_root(self, file_paths, root_path):
    #     # Convert root_path to a Path object
    #     root = Path(root_path)

    #     first_file = Path(file_paths[1]) # [0] is root directory which we are actually trying to extract...

    #     # Extract the common path by finding the relative path from root
    #     common_path = root / first_file.relative_to(root).parts[0]

    #     return str(common_path)

    def _download_file(self, filename):
        url = f"{self.base_url}/{filename}"
        local_path = self.local_dir / filename
        if local_path.exists():
            print(f"Note: Skipping download for {filename}, because it already exists")
            return local_path

        print(f"Downloading {filename}...")
        temp_path = local_path.with_suffix('.part')

        # Resume logic
        headers = {}
        if temp_path.exists():
            headers = {'Range': f'bytes={temp_path.stat().st_size}-'}

        response = requests.get(url, headers=headers, stream=True, timeout=30)
        if temp_path.exists():
            total_size = int(response.headers.get('content-length', 0)) + temp_path.stat().st_size
        else:
            total_size = int(response.headers.get('content-length', 0))

        with open(temp_path, 'ab') as f, tqdm(
            desc=filename,
            total=total_size,
            unit='B',
            unit_scale=True,
            unit_divisor=1024,
            initial=temp_path.stat().st_size
        ) as pbar:
            for chunk in response.iter_content(chunk_size=1024):
                if chunk:
                    f.write(chunk)
                    pbar.update(len(chunk))

        os.rename(temp_path, local_path)
        return local_path

    def _extract_tar(self, tar_path):
        with tarfile.open(tar_path) as tar:
            tar.extractall(path=self.local_dir)
            tar_filenames = tar.getnames()
            extracted_dir = tar_filenames[0]
            os.remove(tar_path)
            return extracted_dir

    def download_dataset(self, file_extracted_directory_pairs):
        results = []
        for (file, extracted_folder_name) in file_extracted_directory_pairs:
            if not (self.local_dir / extracted_folder_name).exists():
                downloaded_file = self._download_file(file)
                results.append(self._extract_tar(downloaded_file))
            else:
                print(f"Warning: Skipping D/L for '{file}' because folder '{extracted_folder_name}' already exists")
                results.append(extracted_folder_name)

        return results

class Spherical2D3DSDataset(Dataset):
    def __init__(self, root_dirs, input_folder='pano/rgb', output_folder='pano/semantic', output_name="semantic"):
        self.root_dirs = root_dirs
        self.input_folder = input_folder
        self.output_folder = output_folder

        # Collect all file paths in the directories
        self.file_paths = []

        for root_dir in self.root_dirs:

            if not os.path.exists(root_dir):
                print(f"Warning: Base directory doesn't exist '{root_dir}'")
                continue

            input_dir = os.path.join(root_dir, input_folder)
            output_dir = os.path.join(root_dir, output_folder)

            if os.path.exists(input_dir) and os.path.exists(output_dir):
                for file_input in os.listdir(input_dir):
                    if not file_input.endswith(".png"):
                        continue
                    input_path = os.path.join(input_dir, file_input)
                    file_output = "_".join(Path(input_path).stem.split("_")[:-1])+f"_{output_name}.png"
                    output_path = os.path.join(output_dir, file_output)
                    if not os.path.exists(output_path):
                        print(f"Warning: Couldn't find output file in pair: ({input_path},{output_path})")
                        continue
                    self.file_paths.append((input_path, output_path))
            elif not os.path.exists(input_dir):
                print("Warning: Input dir doesn't exist: ", input_dir)
                continue
            elif not os.path.exists(output_dir):
                print("Warning: Output dir doesn't exist: ", output_dir)
                continue

    def __len__(self):
        return len(self.file_paths)

    def __getitem__(self, idx):
        input_path, output_path = self.file_paths[idx]

        # Load and convert images to numpy arrays (or tensors if needed)
        input_img = np.array(Image.open(input_path))
        output_img = np.array(Image.open(output_path))

        return input_img, output_img
