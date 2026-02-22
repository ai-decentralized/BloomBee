import argparse
import dataclasses

import functools
import gc
import math
import os
from typing import Tuple, Union, Optional, Any, Sequence, List

import numpy as np
import torch
@dataclasses.dataclass(frozen=True)
class Task:
    """A generation task."""
    inputs: Union[np.array, List[List[int]]]
    prompt_len: int
    gen_len: int
    cut_gen_len: Optional[int]

    do_sample: bool
    temperature: float
    stop: Optional[int]
    top_p: Optional[float] = None
