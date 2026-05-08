#!/usr/bin/env python3

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

"""Check that Python files contain the required SPDX license header."""

import sys

REQUIRED_LINES = [
    "SPDX-FileCopyrightText:",
    "SPDX-License-Identifier: BSD-3-Clause",
]

# Maximum number of lines to scan at the top of each file
HEADER_SCAN_LINES = 10


def check_file(path):
    """Return True if the file contains the required header lines."""
    try:
        with open(path, "r", encoding="utf-8") as f:
            head = "".join(f.readline() for _ in range(HEADER_SCAN_LINES))
    except (OSError, UnicodeDecodeError):
        return True  # skip files we cannot read

    if not head.strip():
        return True  # skip empty files

    for marker in REQUIRED_LINES:
        if marker not in head:
            return False
    return True


def main():
    failed = []
    for path in sys.argv[1:]:
        if not path.endswith(".py"):
            continue
        if not check_file(path):
            failed.append(path)

    if failed:
        print("Missing SPDX license header in:")
        for path in failed:
            print(f"  {path}")
        print()
        print("Every .py file must contain these lines near the top:")
        for marker in REQUIRED_LINES:
            print(f"  # {marker}")
        print()
        print("See CONTRIBUTING.md for the full header template.")
        sys.exit(1)


if __name__ == "__main__":
    main()
