import os
import torch
import torch.nn as nn
import torch.nn.functional as F

from torch.profiler import record_function

import numpy as np
import copy
from tqdm import tqdm

from . import networks_stylegan2 as stylegan

# from torch_utils import persistence

from easydict import EasyDict
from collections import defaultdict
from contextlib import nullcontext

from typing import Tuple
import functools

if not hasattr(torch, "compile"):
    print("[WARNING] 'torch.compile' is not supported")
    torch.compile = lambda x:x

ROOT = os.path.join(os.path.join(os.path.dirname(__file__), ".."))

class SuperresolutionHybrid8XDC(torch.nn.Module):
    def __init__(self, channels, img_resolution, sr_num_fp16_res, sr_antialias,
                num_fp16_res=4, conv_clamp=None, channel_base=None, channel_max=None,# IGNORE
                **block_kwargs):
        super().__init__()
        # assert img_resolution == 512

        use_fp16 = sr_num_fp16_res > 0
        self.input_resolution = 128
        self.sr_antialias = sr_antialias
        self.block0 = stylegan.SynthesisBlock(channels, 256, w_dim=512, resolution=256,
                img_channels=3, is_last=False, use_fp16=use_fp16, conv_clamp=(256 if use_fp16 else None), **block_kwargs)
        self.block1 = stylegan.SynthesisBlock(256, 128, w_dim=512, resolution=512,
                img_channels=3, is_last=True, use_fp16=use_fp16, conv_clamp=(256 if use_fp16 else None), **block_kwargs)

    def forward(self, rgb, x, ws, **block_kwargs):
        ws = ws[:, -1:, :].repeat(1, 3, 1)

        if x.shape[-1] != self.input_resolution:
            x = torch.nn.functional.interpolate(x, size=(self.input_resolution, self.input_resolution),
                                                  mode='bilinear', align_corners=False, antialias=self.sr_antialias)
            rgb = torch.nn.functional.interpolate(rgb, size=(self.input_resolution, self.input_resolution),
                                                  mode='bilinear', align_corners=False, antialias=self.sr_antialias)

        x, rgb = self.block0(x, rgb, ws, **block_kwargs)
        x, rgb = self.block1(x, rgb, ws, **block_kwargs)
        return rgb

class SuperresolutionHybridNXDC(torch.nn.Module):
    def __init__(self, channels, sr_factor, img_resolution, img_channels, sr_antialias,
                num_fp16_res=4, conv_clamp=None, channel_base=None, channel_max=None,
                **block_kwargs):
        super().__init__()
        # assert img_resolution == 512

        use_fp16 = num_fp16_res > 0
        self.input_resolution = img_resolution // sr_factor
        self.sr_antialias = sr_antialias

        block_kwargs = copy.deepcopy(block_kwargs)
        if "min_resolution" in block_kwargs:
            del block_kwargs["min_resolution"]
        
        block_list = []
        res = 2 * img_resolution // sr_factor
        ic, oc = channels, channel_max if channel_max is not None else 256
        while res <= img_resolution:
            is_last = res >= img_resolution
            block_list.append(stylegan.SynthesisBlock(
                        ic, oc, w_dim=512, resolution=res, 
                        img_channels=img_channels, is_last=is_last, use_fp16=use_fp16, conv_clamp=(256 if use_fp16 else None),
                         **block_kwargs))

            res *= 2
            ic, oc = oc, oc//2

        for i, blk in enumerate(block_list):
            setattr(self, f"block{i}", blk)
        
        self.num_blocks = len(block_list)
        
    def forward(self, rgb, x, ws, **block_kwargs):
        ws = ws[:, -1:, :].repeat(1, 3, 1)

        if x.shape[-1] != self.input_resolution:
            x = torch.nn.functional.interpolate(x, size=(self.input_resolution, self.input_resolution),
                                                  mode='bilinear', align_corners=False, antialias=self.sr_antialias)
            rgb = torch.nn.functional.interpolate(rgb, size=(self.input_resolution, self.input_resolution),
                                                  mode='bilinear', align_corners=False, antialias=self.sr_antialias)

        for i in range(self.num_blocks):
            blk = getattr(self, f"block{i}")
            # print(f"block{i}", rgb.shape, ws.shape)
            x, rgb = blk(x, rgb, ws, **block_kwargs)

        # x, rgb = self.block0(x, rgb, ws, **block_kwargs)
        # x, rgb = self.block1(x, rgb, ws, **block_kwargs)
        return rgb