import argparse
import dataclasses

import functools
import gc
import math
import os
from typing import Tuple, Union, Optional, Any, Sequence, List
# fix recursive import
from bloombee.flexgen_utils.pytorch_backend import TorchDevice, TorchDisk, TorchMixedDevice

import numpy as np
import torch
@dataclasses.dataclass(frozen=True)
class ExecutionEnv:
    """Hardware environment."""
    gpu: Any = None
    cpu: Any = None
    disk: Any = None
    mixed: Any = None

    @classmethod
    def create(cls, offload_dir, device_type="cuda"):
        # For CPU-only mode, use CPU for all devices including 'gpu' slot
        if device_type == "cpu":
            gpu = TorchDevice("cpu")  # Use CPU for the 'gpu' slot
        else:
            gpu = TorchDevice("cuda:0")
        print('ExecutionEnv: gpu ', gpu)
        cpu = TorchDevice("cpu")
        disk = TorchDisk(offload_dir)
        return cls(gpu=gpu, cpu=cpu, disk=disk, mixed=TorchMixedDevice([gpu, cpu, disk]))

    def close_copy_threads(self):
        self.disk.close_copy_threads()
