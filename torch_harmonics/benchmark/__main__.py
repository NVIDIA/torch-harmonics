import sys

import torch_harmonics.benchmark  # noqa: F401 — triggers benchmark registration
from torch_harmonics.benchmark.run import cli

sys.exit(cli())
