import os
import sys
import torch
import torch.nn as nn
import torch.nn.functional as F

import numpy as np

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))

sys.path.insert(0, ROOT)
from utils import chunk_fn, device_control
sys.path.pop(0)

torch_compile = device_control.compile

class PositionEncoding(nn.Module):
    def __init__(self, dim_input, dim_band):
        super().__init__()

        self.dim_input = dim_input
        self.dim_band  = dim_band

        self.embed_fn      = []
        self.embed_grad_fn = []

        max_freq = dim_band - 1

        freq_bands = [2**i for i in np.linspace(0, max_freq, dim_band)]

        func_list = [torch.sin, torch.cos]
        # grad_list = [torch.cos, torch.sin]

        mul_act = lambda x, mul=1.0, fun=None: fun(mul*x) 
        mul_drv = lambda x, mul=1.0, fun=None: mul*fun(mul*x) 
        from functools import partial

        # include input
        self.embed_fn.append(
            partial(mul_act, mul=1, fun=lambda x:x)
        )

        self.embed_grad_fn.append(
            partial(mul_act, mul=1, fun=lambda x:torch.ones_like(x))
        )

        scale = torch.as_tensor(freq_bands)[:, None].expand(-1, 2).flatten() # N*2
        shift = torch.as_tensor([0, torch.pi/2])[None, :].expand(dim_band, -1).flatten() # N*2
        self.register_buffer("scale", scale.float())
        self.register_buffer("shift", shift.float())

        self.register_buffer( "weight", torch.kron(scale.reshape(-1, 1), torch.eye(dim_input)).float() ) # N*2*3
        self.register_buffer( "bias",   shift[:, None].expand(-1, dim_input).flatten().float() )         # N*2*3

        # print(self.weight)
        # print(self.bias)

        for freq in freq_bands:
            for fn in func_list:
                self.embed_fn.append(
                    partial(mul_act, mul=freq, fun=fn)
                )
                if fn == torch.sin:
                    d_fn = torch.cos
                elif fn == torch.cos:
                    d_fn = lambda x:-torch.sin(x)
                self.embed_grad_fn.append(
                    partial(mul_drv, mul=freq, fun=d_fn)
                )
        print(len(freq_bands), len(self.embed_fn))

        self.dim_output = len(self.embed_fn)*dim_input

        del self.scale
        del self.shift

    # @torch_compile(fullgraph=True, mode="reduce-overhead")
    def forward(self, data):
        # scaled  = torch.einsum("b...c,k->b...ck", data, self.scale)
        # shifted = scaled + self.shift
        # shifted = shifted.transpose(-2, -1).flatten(-2)

        shifted = torch.nn.functional.linear(data, self.weight, self.bias)
        # pe  = torch.sin(shifted)            # B,N*2*C
        pe  = shifted.sin_()
        ret = torch.cat([data, pe], dim=-1) # B,C+N*2*C

        # ret = []
        # for fn in self.embed_fn:
        #     ret.append(fn(data))
        
        # ret = torch.cat(ret, axis=-1)
        return ret

    def __repr__(self):
        return f"PositionEncoding({self.dim_input}, {self.dim_band}, dim_output={self.dim_output})"