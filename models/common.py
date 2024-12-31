import os
import sys
import torch
import torch.nn as nn
import torch.nn.functional as F

import numpy as np
from easydict    import EasyDict
from functools   import reduce, partial
from contextlib  import nullcontext
from collections import OrderedDict

ROOT = os.path.abspath(os.path.join(__file__, "..", ".."))

def exists(val):
    return val is not None

def default(val, d):
    if exists(val):
        return val
    return d() if callable(d) else d

def fix_param_as_buffer(model, name):
    p = getattr(model, name)
    delattr(model, name)
    model.register_buffer(name, p.data)

class Normalize(nn.Module):
    def __init__(self, mean=None, std=None, shape=None):
        super().__init__()

        mean = [0] if mean is None else mean
        std  = [1] if std  is None else std
        shape = [1,1,1,] if shape is None else shape

        mean = torch.as_tensor(mean).float().reshape(shape)
        std  = torch.as_tensor(std).float().reshape(shape)

        # self.register_buffer("mean", mean)
        # self.register_buffer("std",   std)
        self.register_buffer("mean", mean.clone().detach())
        self.register_buffer("std",   std.clone().detach())

        self.register_buffer("inv_std", 
            torch.where(self.std==0, torch.zeros_like(self.std), 1/self.std))
    
    def forward(self, x):
        return (x - self.mean) / self.std
        # return (x - self.mean) * self.inv_std

    @torch.jit.export
    def invert(self, x):
        return x * self.std + self.mean
    
    def __str__(self):
        return f"Normalize(mean_shape={tuple(self.mean.shape)}, std_shape={tuple(self.std.shape)})"
