import os
import sys
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms.functional as Ft
import torchvision.ops                   as Tops

import math
import copy

from easydict    import EasyDict
from contextlib  import nullcontext, contextmanager
from collections import OrderedDict, namedtuple

from torch.profiler import record_function
from typing import Dict, List, Tuple
import functools

from .gaussian_geometry.eg_head_v0 import EGHeadConfig, EGHead, FLAMEParam
from . import SRs

ROOT = os.path.abspath(os.path.join(__file__, "..", ".."))

sys.path.insert(0, ROOT)
import models.networks_stylegan2 as stylegan

from utils import chunk_fn, near_far_from_sphere, near_far_from_value, device_control
sys.path.pop(0)

torch_compile_or_jit = device_control.torch_compile_or_jit

@torch_compile_or_jit
def sigmoid_clamp(value, eps:float = 0.001):
    return torch.sigmoid(value)*(1 + 2*eps) - eps

@torch_compile_or_jit
def resize_2d(img_bhwc, H:int, W:int):
    b, h, w, c = img_bhwc.shape

    if (h, w) != (H, W):
        img_bhwc = F.interpolate(img_bhwc.permute(0, 3, 1, 2), (H, W), 
            mode="bilinear", align_corners=False, antialias=True).permute(0, 2, 3, 1)
    return img_bhwc

def get_densify_component(ver, tri, uvt, uvt_tri, densify_level):
    device = tri.device
    i_ver, i_uvt = [], []
    v_all = []
    f_ind = []
    NV = 0

    for ti in range(tri.size(0)):
        level = densify_level[ti]
        vid_3 = tri[ti].reshape(3)
        uvt_3 = uvt_tri[ti].reshape(3)

        if level == 1:
            i_ver.append(torch.stack([
                torch.full((3,), NV, device=device), vid_3
                ], dim=0))
            i_uvt.append(torch.stack([
                torch.full((3,), NV, device=device), uvt_3
                ], dim=0))
            v_all.append(
                torch.full((3,), 1/3, device=device)
            )
            f_ind.append(ti)
            NV += 1
        elif level == 2:
            v_all.append(torch.as_tensor((2/3, 1/6, 1/6), device=device))
            v_all.append(torch.as_tensor((1/6, 2/3, 1/6), device=device))
            v_all.append(torch.as_tensor((1/6, 1/6, 2/3), device=device))
            v_all.append(torch.as_tensor((1/3, 1/3, 1/3), device=device))
            i_ver.append(torch.stack([torch.full((3,), NV+0, device=device), vid_3], dim=0))
            i_ver.append(torch.stack([torch.full((3,), NV+1, device=device), vid_3], dim=0))
            i_ver.append(torch.stack([torch.full((3,), NV+2, device=device), vid_3], dim=0))
            i_ver.append(torch.stack([torch.full((3,), NV+3, device=device), vid_3], dim=0))
            i_uvt.append(torch.stack([torch.full((3,), NV+0, device=device), uvt_3], dim=0))
            i_uvt.append(torch.stack([torch.full((3,), NV+1, device=device), uvt_3], dim=0))
            i_uvt.append(torch.stack([torch.full((3,), NV+2, device=device), uvt_3], dim=0))
            i_uvt.append(torch.stack([torch.full((3,), NV+3, device=device), uvt_3], dim=0))
            f_ind.extend([ti]*4)
            NV += 4
        
    i_ver = torch.cat(i_ver, dim=1)
    i_uvt = torch.cat(i_uvt, dim=1)
    v_all = torch.cat(v_all, dim=0)
    f_ind = torch.as_tensor(f_ind)
    ver_matrix = torch.sparse_coo_tensor(i_ver, v_all, (NV, ver.size(0)))
    uvt_matrix = torch.sparse_coo_tensor(i_uvt, v_all, (NV, uvt.size(0)))

    densify_dict = {
        "ver_mat": ver_matrix,
        "uvt_mat": uvt_matrix,
        "face_id": f_ind,
    }
    state_dict = {
        "accgrad": torch.zeros((NV,), dtype=torch.float32, device=device),
        "counter": torch.zeros((NV,), dtype=torch.int32, device=device),
    }
    return densify_dict, state_dict

@torch_compile_or_jit
def get_ray(pix_x, pix_y, K, c2w):
    # ray dir
    fx = K[..., 0, 0]
    fy = K[..., 1, 1]
    cx = K[..., 0, 2]
    cy = K[..., 1, 2]
    sk = K[..., 0, 1]

    y, x = pix_y, pix_x  # x:(W,H), y:(W,H)
    # y, x = torch.meshgrid(pix_y, pix_x)  # x:(W,H), y:(W,H)
    dirs = torch.stack([(x-cx + cy*sk/fy - sk*y/fy)/fx, (y-cy)/fy, torch.ones_like(x)], -1) # (H,W,3)
    # Rotate ray directions from camera frame to the world frame
    rays_d = torch.einsum("...ij,...j->...i", c2w[..., :3,:3], dirs)
    rays_o = c2w[..., :3, -1]
    return rays_o, rays_d

@torch_compile_or_jit
def get_primary_ray(m2v, ndc, H:int=512, W:int=512):
    BS     = m2v.size(0)
    device = m2v.device

    # with torch.cuda.amp.autocast(enabled=False):
    yi, xi = torch.meshgrid(torch.arange(H, device=device), torch.arange(W, device=device))
    # yi, xi = map(lambda x:x.float().flatten(), [yi, xi])
    yi, xi = yi.float().flatten(), xi.float().flatten()
    xu, yv = (xi+0.5)/W, (yi+0.5)/H

    # cam_gl2cv = locals.get("cam_gl2cv", torch.diag(torch.as_tensor([1,-1,-1,1]).float()).to(device)) # opengl camera to opencv
    cam_gl2cv = torch.diag(torch.as_tensor([1,-1,-1,1], dtype=torch.float, device=device)) # opengl camera to opencv

    m2v_cv = cam_gl2cv.unsqueeze(0) @ m2v
    c2w    = torch.inverse(m2v_cv)

    fx, fy = ndc[:, 0, 0, None]/2, ndc[:, 1, 1, None]/2
    # cx = cy = 0.5
    cx, cy = 0.5-ndc[:, 0, 2, None]/2, ndc[:, 1, 2, None]/2+0.5
    sk = 0
    x, y = xu.unsqueeze(0).expand(BS,-1), yv.unsqueeze(0).expand(BS,-1)
    dirs = torch.stack([(x-cx + cy*sk/fy - sk*y/fy)/fx, (y-cy)/fy, torch.ones_like(x)], -1) # (BS,H*W,3)
    # Rotate ray directions from camera frame to the world frame
    rays_d = torch.einsum("bij,bnj->bni", c2w[..., :3,:3], dirs)
    rays_o = c2w[..., :3, -1]
    return rays_o.reshape(BS, 1, 1, 3), rays_d.reshape(BS, H, W, 3)

def erp1_xyz2uv(xyz):
    x,y,z = torch.tensor_split(xyz, 3, dim=-1)
    # y,z,x = torch.tensor_split(xyz, 3, dim=-1)

    # phi = torch.atan(-z/x)
    # phi = torch.atan(-z/x)
    the = torch.acos(z)

    sin_the = torch.sin(the)
    sin_phi =-y / sin_the
    cos_phi = x / sin_the
    # sin_phi = y / sin_the
    # cos_phi =-x / sin_the

    phi = torch.acos(cos_phi)
    phi = torch.where( sin_phi > 0, phi, -phi)
    the = torch.where( torch.isfinite(the), the, 0)
    phi = torch.where( torch.isfinite(phi), phi, 0)

    u = phi / (2*torch.pi) + 0.5
    v = the / torch.pi

    # u = u - 0.5
    # u = torch.where( u<0, 1+u, u)
    # u = torch.where( u>1, u-1, u)

    return torch.cat([u, v], dim=-1)

def erp2_xyz2uv(xyz, rot_90_x=True):
    '''
    theta [0, pi]
    phi   [0, 2*pi]
    '''
    x,y,z = torch.tensor_split(xyz, 3, dim=-1)
    if rot_90_x is True:
        y, z = -z, y

    the = torch.acos(z)

    phi = torch.atan2(-y, x)

    assert torch.isfinite(phi).all(), f"{sin_phi[torch.logical_not(torch.isfinite(phi))]} {cos_phi[torch.logical_not(torch.isfinite(phi))]}"

    u = phi / (2*torch.pi) + 0.5
    v = the / torch.pi

    return torch.cat([u, v], dim=-1)

@torch_compile_or_jit
def shading_SH2(normals, sh_coeff):
    #, locals:Dict[str, torch.Tensor]={}):
    '''
        normals:  [bs, N, 3]
        sh_coeff: [bs, 9, 3]
    '''

    pi = np.pi
    # np.sqrt will cause jit.script fail
    constant_factor = torch.as_tensor(
    [1 / math.sqrt(4 * pi), ((2 * pi) / 3) * (math.sqrt(3 / (4 * pi))), ((2 * pi) / 3) * (math.sqrt(3 / (4 * pi))), \
     ((2 * pi) / 3) * (math.sqrt(3 / (4 * pi))), (pi / 4) * (3) * (math.sqrt(5 / (12 * pi))),
     (pi / 4) * (3) * (math.sqrt(5 / (12 * pi))), \
     (pi / 4) * (3) * (math.sqrt(5 / (12 * pi))), (pi / 4) * (3 / 2) * (math.sqrt(5 / (12 * pi))),
     (pi / 4) * (1 / 2) * (math.sqrt(5 / (4 * pi)))], dtype=torch.float32, device=normals.device)
    # constant_factor = locals.get("constant_factor", constant_factor.to(normals.device))
    # locals["constant_factor"] = constant_factor
    # constant_factor = constant_factor.to(normals.device)

    x, y, z = normals.unbind(-1)
    sh = torch.stack([
        x * 0. + 1., x, y, \
        z,  x * y, x * z,
        y * z, x ** 2 - y ** 2, 3 * (z ** 2) - 1
        ],
        dim=1)  # [bs, 9, N]
    sh = sh * constant_factor[None, :, None]   # [bs, 9, N]
    # shading = torch.sum(sh_coeff[:, :, :, None, None] * sh[:, :, None, :, :], 1)  # [bz, 9, 3, h, w]
    shading = torch.einsum("bkc,bkn->bnc", sh_coeff, sh)
    return shading

def fix_param_as_buffer(model, name):
    p = getattr(model, name)
    delattr(model, name)
    model.register_buffer(name, p.data)

def unlock_buffer(model, name):
    b = getattr(model, name)
    delattr(model, name)
    model.register_parameter(name, nn.Parameters(b))

class FCDecoder(torch.nn.Module):
    def __init__(self, n_features, options):
        super().__init__()
        self.hidden_dim = options.get("decoder_hidden_dim", 64)
        # self.aggregate = options.get("fuse", "avg")

        act_fn = options.get("decoder_activation", "softplus")
        if act_fn == "softplus":
            act_factory = lambda :nn.Softplus()
        elif act_fn == "silu":
            act_factory = lambda :nn.SiLU()
        
        lr_mul = options['decoder_lr_mul']
        
        layers = []
        for i in range(options["decoder_num_layer"]):
            ic = oc = self.hidden_dim
            if i == 0:
                ic = n_features
            if i == options["decoder_num_layer"] - 1:
                oc = options["decoder_output_dim"]

            layers.append(stylegan.FullyConnectedLayer(ic, oc, lr_multiplier=lr_mul))
            layers.append(act_factory())
        
        del layers[-1]

        self.net = torch.nn.Sequential(*layers)

        # self.net = torch.nn.Sequential(
        #     stylegan.FullyConnectedLayer(n_features, self.hidden_dim, lr_multiplier=),
        #     torch.nn.Softplus(),
        #     stylegan.FullyConnectedLayer(self.hidden_dim, self.hidden_dim, lr_multiplier=options['decoder_lr_mul']),
        #     torch.nn.Softplus(),
        #     stylegan.FullyConnectedLayer(self.hidden_dim, options['decoder_output_dim'], lr_multiplier=options['decoder_lr_mul'])
        # )
        
    def forward(self, sampled_features, ray_directions):
        # Aggregate features
        # if self.aggregate == "avg":
        #     sampled_features = sampled_features.mean(1)
        # elif self.aggregate == "avg":
        #     sampled_features = sampled_features.prod(1)
        x = sampled_features

        N, M, C = x.shape
        x = x.view(N*M, C)

        x = self.net(x)
        x = x.view(N, M, -1)
        rgb = x
        # rgb = torch.sigmoid(x[..., 1:])*(1 + 2*0.001) - 0.001 # Uses sigmoid clamping from MipNeRF
        # sigma = x[..., 0:1]
        return {'rgb': rgb}

class SRWithCond(nn.Module):
    def __init__(self, final_resolution, input_channel, sr_factor, output_channel, c_dim=0, w_dim=512, z_dim=512, 
        num_fp16_res=4,
        mapping_kwargs={}, block_kwargs={}):
        super().__init__()

        self.c_dim          = c_dim
        self.output_channel = output_channel

        if "num_fp16_res" in block_kwargs:
            block_kwargs = {k:v for k, v in block_kwargs.items() if k != "num_fp16_res"}

        if self.c_dim > 0:
            self.cond_inject     = stylegan.MappingNetwork(
                    z_dim=z_dim, c_dim=c_dim, w_dim=w_dim, num_ws=1, **mapping_kwargs)
        
        if input_channel < output_channel:
            self.proj = nn.Conv2d(input_channel, output_channel, (1, 1), stride=1, padding=0)
        else:
            self.proj = nn.Identity()

        self.superresolution = SRs.SuperresolutionHybridNXDC(input_channel, sr_factor, final_resolution, output_channel, 
            sr_antialias=True, num_fp16_res=num_fp16_res, **block_kwargs)
    
    # @torch.compile
    def forward(self, image, ws, cond=None, truncation_psi=1, truncation_cutoff=None, update_emas=False):
        if ws.ndim == 2:
            ws = ws.unsqueeze(1)
        if self.c_dim > 0:
            sr_ws = self.cond_inject(ws[:, -1, :], cond, truncation_psi=truncation_psi, truncation_cutoff=truncation_cutoff, update_emas=update_emas)
        else:
            sr_ws = ws[:, -1:, :]
        
        image = self.superresolution(self.proj(image[:, :self.output_channel]), image, sr_ws, noise_mode="none")
        return image, sr_ws

class SRCond(nn.Module):
    def __init__(self, final_resolution, input_channel, sr_factor, output_channel, c_dim=0, w_dim=512, z_dim=512, 
        num_fp16_res=4,
        mapping_kwargs={}, block_kwargs={}, adain=False):
        super().__init__()

        self.c_dim          = c_dim
        self.output_channel = output_channel

        if "num_fp16_res" in block_kwargs:
            block_kwargs = {k:v for k, v in block_kwargs.items() if k != "num_fp16_res"}

        self.cond_inject     = stylegan.MappingNetwork(
                z_dim=z_dim, c_dim=c_dim, w_dim=w_dim, num_ws=1, **mapping_kwargs)

        if adain:
            self.adain_affine = nn.Linear(w_dim, input_channel*2)
        self.adain = adain

        if input_channel < output_channel:
            self.proj = nn.Conv2d(input_channel, output_channel, (1, 1), stride=1, padding=0)
        else:
            self.proj = nn.Identity()

        self.superresolution = SRs.SuperresolutionHybridNXDC(input_channel, sr_factor, final_resolution, output_channel, 
            sr_antialias=True, num_fp16_res=num_fp16_res, **block_kwargs)
    
    # @torch.compile
    def forward(self, image, noise, cond=None, truncation_psi=1, truncation_cutoff=None, update_emas=False):
        sr_ws = self.cond_inject(noise, cond, truncation_psi=truncation_psi, truncation_cutoff=truncation_cutoff, update_emas=update_emas)

        if self.adain:
            # x * torch.rsqrt(torch.mean(x ** 2, dim=1, keepdim=True) + 1e-5)
            w_b = self.adain_affine(sr_ws[:, 0])[:, :, None, None]  # BS, 2*C, 1, 1
            w,b = torch.split(w_b, (image.size(1), image.size(1)), dim=1)
            image = F.instance_norm(image)*w+b

        image = self.superresolution(self.proj(image[:, :self.output_channel]), image, sr_ws, noise_mode="none")
        return image, sr_ws

class ShadingNetBase(nn.Module):
    def __init__(self):
        super().__init__()

        pi = np.pi
        # np.sqrt will cause jit.script fail
        constant_factor = torch.as_tensor(
            [1 / math.sqrt(4 * pi), ((2 * pi) / 3) * (math.sqrt(3 / (4 * pi))), ((2 * pi) / 3) * (math.sqrt(3 / (4 * pi))), \
            ((2 * pi) / 3) * (math.sqrt(3 / (4 * pi))), (pi / 4) * (3) * (math.sqrt(5 / (12 * pi))),
            (pi / 4) * (3) * (math.sqrt(5 / (12 * pi))), \
            (pi / 4) * (3) * (math.sqrt(5 / (12 * pi))), (pi / 4) * (3 / 2) * (math.sqrt(5 / (12 * pi))),
            (pi / 4) * (1 / 2) * (math.sqrt(5 / (4 * pi)))], dtype=torch.float32)

        self.register_buffer("sh_constant_factor", constant_factor)
    
    def get_sh(self, normal, sh_coeff):
        #, locals:Dict[str, torch.Tensor]={}):
        '''
            normals:  [bs, N, 3]
            sh_coeff: [bs, 9, 3]
        '''

        # constant_factor = locals.get("constant_factor", constant_factor.to(normals.device))
        # locals["constant_factor"] = constant_factor
        # constant_factor = constant_factor.to(normals.device)

        x, y, z = normal.unbind(-1)
        sh = torch.stack([
            x * 0. + 1., x, y, \
            z,  x * y, x * z,
            y * z, x ** 2 - y ** 2, 3 * (z ** 2) - 1
            ],
            dim=1)  # [bs, 9, N]
        # sh = sh * constant_factor[None, :, None]   # [bs, 9, N]
        # shading = torch.sum(sh_coeff[:, :, :, None, None] * sh[:, :, None, :, :], 1)  # [bz, 9, 3, h, w]
        shading = torch.einsum("bkc,bkn,k->bnc", sh_coeff, sh, self.sh_constant_factor)
        return shading

class MagicScale(torch.autograd.Function):
    @staticmethod
    def forward(ctx, val, mul_fwd, mul_bwd):
        # ctx.save_for_backward(mul_bwd)
        ctx.mul_bwd = mul_bwd
        return val*mul_fwd
    
    @staticmethod
    def backward(ctx, grad):
        # mul, = ctx.saved_tensors
        mul = ctx.mul_bwd
        return grad*mul, None, None


    def forward(self, nrm_dir, depth, light_code, m2v, ndc):
        with record_function("shading_sh"):
            sh2 = self.get_sh(nrm_dir, light_code)
        with record_function("shading_ao"):
            ao  = self.get_ao(nrm_dir, depth, light_code, m2v, ndc)

class AO_SH2_MLP(ShadingNetBase):
    def __init__(self, light_channel, logit_scale=5.0):
        super().__init__()

        self.mlp = nn.Sequential(
            nn.Linear(1+3+light_channel, 32),   # in-efficient when shading object ~1M point
            nn.ELU(),
            nn.Linear(32, 32),
            nn.ELU(),
            nn.Linear(32, 3),
        )
        self.logit_scale = logit_scale

        self.ao_conv = nn.Sequential(
            nn.Conv2d(4, 16,  (3, 3), padding=1, stride=1),
            nn.ELU(),
            nn.Conv2d(16, 16, (3, 3), padding=1, stride=1),
            nn.ELU(),
            nn.Conv2d(16, 1,  (3, 3), padding=1, stride=1),
        )
    
    # def get_sh(self, nrm_dir, light_code):
    #     return shading_SH2(nrm_dir.flatten(1, 2), light_code.reshape(-1, 9, 3)).reshape(nrm_dir.shape) # b,h,w,c
    def get_sh(self, nrm_dir, light_code):
        # return shading_SH2(nrm_dir.flatten(1, 2), light_code.reshape(-1, 9, 3)).reshape(nrm_dir.shape) # b,h,w,c
        return super().get_sh(nrm_dir.flatten(1, 2), light_code.reshape(-1, 9, 3)).reshape(nrm_dir.shape)
    
    def get_ao(self, nrm_dir, depth, light_code, m2v, ndc):
        depth = depth[..., :1]
        ao = self.ao_conv(torch.cat([nrm_dir.permute(0, 3, 1, 2), depth.permute(0, 3, 1, 2)], dim=1)).permute(0, 2, 3, 1).expand_as(nrm_dir)
        ao = torch.sigmoid(ao + 3)
        return ao

    def forward(self, nrm_dir, depth, light_code, m2v, ndc):
        sh2= self.get_sh(nrm_dir, light_code)
        ao = self.get_ao(nrm_dir, depth, light_code, m2v, ndc)
        return sh2 * ao

with nullcontext("G"):
    class G(nn.Module):
        def __init__(self, noise_config=None, condition_config=None):
            super().__init__()

            def make_dim(val):
                if isinstance(val, (list, tuple)):
                    return val
                else:
                    return [val]

            if isinstance(noise_config, dict):
                k = list(noise_config.keys())
                v = [make_dim(noise_config[i]) for i in k]
                self.noise_names = k
                self.noise_dims  = v

                self.noise_split = [np.prod(d) for d in self.noise_dims]
            else:
                self.noise_split = []

            if isinstance(condition_config, dict):
                k = list(condition_config.keys())
                v = [make_dim(condition_config[i]) for i in k]
                self.condition_names = k
                self.condition_dims  = v

                self.condition_split = [np.prod(d) for d in self.condition_dims]
            elif condition_config is None:
                self.condition_names = []
                self.condition_dims  = []
                self.condition_split = []
            
            assert len(set(self.noise_names)&set(self.condition_names)) == 0, f"{self.noise_names}, {self.condition_names}"
        
        def forward(self, noise, condition=None):
            pass

        # @torch_compile()
        def unpack_condition(self, condition):
            # split_size = [np.prod(d) for d in self.condition_dims]
            split_size:List[int] = self.condition_split
            split_th   = torch.split(condition, split_size, dim=-1)

            unpack_d   = {}
            for i, d in enumerate(self.condition_dims):
                k = self.condition_names[i]
                v = split_th[i].unflatten(1, d)

                unpack_d[k] = v
            return unpack_d

        # @torch_compile()
        def pack_condition(self, condition_dict: Dict[str, torch.Tensor]):
            if len(self.condition_names) == 0:
                return None
            pack = []
            for k, d in zip(self.condition_names, self.condition_dims):
                # assert k in condition_dict, f"'{k}' is not in {condition_dict.keys()}"
                v = condition_dict[k]
                # assert tuple(v.shape[1:]) == tuple(d), f"pack error for {k}, {v.shape[1:]}->{d}"
                pack.append(v.flatten(1))
            return torch.cat(pack, dim=-1)

        # @torch_compile()
        def unpack_noise(self, noise):
            split_size:List[int] = self.noise_split
            split_th   = torch.split(noise, split_size, dim=-1)

            unpack_d   = {}
            for i, d in enumerate(self.noise_dims):
                k = self.noise_names[i]
                v = split_th[i].unflatten(1, d)

                unpack_d[k] = v
            return unpack_d

        # @torch_compile()
        def pack_noise(self, noise_dict: Dict[str, torch.Tensor]):
            if len(self.noise_names) == 0:
                return None
            pack = []
            for k, d in zip(self.noise_names, self.noise_dims):
                # assert k in noise_dict, f"'{k}' is not in {noise_dict.keys()}"
                v = noise_dict[k]
                # assert tuple(v.shape[1:]) == tuple(d), f"pack error for {k}, {v.shape[1:]}->{d}"
                pack.append(v.flatten(1))
            return torch.cat(pack, dim=-1)

    # @persistence.persistent_class
    class Gen2D(G):
        def __init__(self, 
            # z_dim, 
            # c_dim, 
            w_dim, 
            img_resolution,
            img_channels,
            mapping_kwargs    = {},
            synthesis_kwargs  = {},
            sr_factor           = 1,
            sr_synthesis_kwargs = {},
            noise_config      = None,
            condition_config  = None,
            cond_swap_config  = None,
            ):
            super().__init__(noise_config, condition_config)
            z_dim = np.sum(self.noise_split)
            c_dim = np.sum(self.condition_split)
            self.z_dim = z_dim
            self.c_dim = c_dim
            self.w_dim = w_dim
            self.img_resolution = img_resolution
            self.img_channels   = img_channels


            # block_keys = ["noise_mode", "fused_modconv", "gain", "force_fp32", "update_emas"]
            block_keys = ["noise_mode", "fused_modconv", "gain", "force_fp32"]

            self.block_kwargs = {k:v for k, v in synthesis_kwargs.items() if k in block_keys}
            synthesis_kwargs  = {k:v for k, v in synthesis_kwargs.items() if k not in block_keys}

            if sr_factor == 1:
                synthesis_net = stylegan.SynthesisNetwork(
                    w_dim=w_dim, img_resolution=img_resolution, img_channels=img_channels, **synthesis_kwargs)
                super_res_net = None
            else:
                synthesis_net = stylegan.SynthesisNetwork(
                    w_dim=w_dim, img_resolution=img_resolution//sr_factor, img_channels=32, **synthesis_kwargs)
                super_res_net = SRs.SuperresolutionHybridNXDC(32, sr_factor, img_resolution, img_channels, num_fp16_res=4, sr_antialias=True)

            self.num_ws    = synthesis_net.num_ws
            mapping_net   = stylegan.MappingNetwork(
                z_dim=z_dim, c_dim=c_dim, w_dim=w_dim, num_ws=self.num_ws, **mapping_kwargs)
                # num_layers=2)

            self.mapping         = mapping_net
            self.synthesis       = synthesis_net
            self.superresolution = super_res_net

            print("num_ws", self.num_ws)

        def forward(self, noise, condition, truncation_psi=1, truncation_cutoff=None, update_emas=False, **synthesis_kwargs):
            BS     = noise.size(0)
            device = noise.device

            kwargs = dict(synthesis_kwargs)
            kwargs.update(self.block_kwargs)
            if "noise_mode" in synthesis_kwargs:
                kwargs["noise_mode"] = synthesis_kwargs["noise_mode"]

            with torch.cuda.amp.autocast(enabled=False):
                ws    = self.mapping(noise, condition, 
                                        truncation_psi=truncation_psi, truncation_cutoff=truncation_cutoff, update_emas=update_emas)
                image = self.synthesis(ws, update_emas=update_emas, **kwargs)            # B,C,H,W

                image = image.reshape(BS, -1, image.size(-2), image.size(-1))

                if self.superresolution is not None:
                    sr_ws = ws[:, -1:, :].expand(-1, 3, -1)
                    image = self.superresolution(image[:, :self.img_channels], image, sr_ws, noise_mode="none")
            return image, ws
    
    class EGG3DGAN(G):
        def __init__(self, 
            z_dim, 
            w_dim                    = 512,
            noise_config             = None,
            condition_config         = None,
            disc_condition_config    = None,

            tex_noise_config         = None,
            tex_condition_config     = None,

            alb_noise_config         = None,
            alb_condition_config     = None,

            bmp_noise_config         = None,
            bmp_condition_config     = None,

            tri_noise_config         = None,
            tri_condition_config     = None,

            bcg_noise_config         = None,
            bcg_condition_config     = None,

            cond_swap_config         = {}, # tex, albd, bump, trif, back
            cond_swap_init_prob      = 0,
            cond_swap_decay          = 100,

            mapping_kwargs           = {},
            sr_mapping_kwargs        = {},

            field_config             = {},

            # common
            num_fp16_res             = 0,
            min_resolution           = 4,
            fp16_channels_last       = True,

            img_resolution           = 512, 
            img_channels             = 3,
            shd_resolution           = 64,

            # G_share
            shared_resolution        = 0,
            shared_channel_max       = 256,
            shared_sr_factor         = 1,
            feature_channels         = 32,

            # G_tex
            tex_resolution           = 512,
            # tex_min_resolution       = 4,
            tex_channel_max          = 512,
            tex_channel_base         = 32768,
            tex_sr_factor            = 1,

            # G_tri
            tri_resolution           = 512,
            tri_channel_max          = 512,
            tri_channel_base         = 32768,
            tri_sr_factor            = 1,
            tri_roll_out             = None,
            tri_noise_mode           = "random",

            # G_back
            back_resolution          = 512,
            back_min_resolution      = 4,
            back_num_fp16            = 0,
            back_up_factor           = 2,
            back_channel_max         = 512,
            back_channel_base        = 32768,
            back_sr_factor           = 1,
            background_activation    = "sigmoid",

            head_gs_spp              = 1,
            head_gs_init             = "loop_area",
            head_gs_normal           = "simple",
            head_gs_scale            = (0.0006, 0.0004),
            head_gs_bump_range       = 0.0025,
            free_gs_offset           = 1,
            free_gs_scale            = 1,
            free_gs_ratio            = {},
            free_gs_residual         = False,
            free_gs_mlp_key          = ["hair", "shape"],

            # global multiplier
            gs_scale_multiplier      = 1,
            gs_opacity_multiplier    = 1,

            # scale gradient of Gaussian position
            gaussian_position_grad_scale = 1,

            # flame_offset             = None,
            pose_multiplier          = None,

            # rendering
            render_method            = "3dgs",
            shading_type             = "fragment",
            light_method             = "Shading_SH2",

            # !!! think before change
            # using_bump               = True,
            using_streams            = False,
            render_streams           = False,
            fast_init                = False,
            ):

            super().__init__(noise_config, condition_config)

            # Config
            with nullcontext("Config"):
                code_config = {k: v for k, v in noise_config.items()}
                code_config.update(condition_config)

                self.z_dim    = np.sum(self.noise_split)
                self.c_dim    = np.sum(self.condition_split)

                self.cond_swap_config = cond_swap_config

                self.cond_swap_init_prob = cond_swap_init_prob
                self.cond_swap_decay     = cond_swap_decay

                disc_condition_config = disc_condition_config if disc_condition_config is not None else condition_config
                self.disc_G = G(noise_config, disc_condition_config)

                # flame config
                model_cfg = EasyDict({
                    "flame_model_path":        os.path.join(ROOT, "Data", "FLAME2020", "generic_model.pkl"),
                    "flame_lmk_embedding_path":os.path.join(ROOT, "Data", "FLAME2020", "landmark_embedding.npy"),
                    "flame_tex_path":          os.path.join(ROOT, "Data", "FLAME2020", "FLAME_texture.npz"),
                    "tex_path":                os.path.join(ROOT, "Data", "FLAME2020", "FLAME_albedo_from_BFM.npz"),
                    "mask_path":               os.path.join(ROOT, "Data", "FLAME_masks", "FLAME_masks.pkl"),
                    "n_shape": 100,
                    "n_exp":   50,
                    "n_tex":   50,
                    "tex_type": 'FLAME',
                })
                self.flame_cfg = model_cfg

                # flame_offset = torch.as_tensor(flame_offset) if flame_offset is not None else torch.zeros(3)
                # self.register_buffer("flame_offset", flame_offset.float().reshape(1, 1, 3))

                if pose_multiplier is None:
                    self.pose_multiplier = 1
                else:
                    self.register_buffer("pose_multiplier", torch.as_tensor(pose_multiplier).reshape(1, -1))

                self.img_resolution = img_resolution
                self.img_channels   = img_channels
                self.shd_resolution = shd_resolution
                self.fixed_intrinsic= True

                # self.feature_resolution = feature_resolution
                self.field_config = copy.deepcopy(field_config)
                # {
                #     "dec_type": "attribute",
                #     "box_warp":  0.5,
                #     "pe_scale":  1,
                #     "aggregate": "mul",
                #     "layout":    "flat"
                # }

                # self.bump_range        = 0.0025 # 2.5 mm
                self.bump_range        = head_gs_bump_range
                self.environ_generator = "synthesis"

                conv_clamp = 256 if num_fp16_res > 0 else None
                common_synthesis_kwargs = {
                    "channel_base": 32768, "channel_max": 512, 
                    "num_fp16_res": num_fp16_res, "fp16_channels_last": fp16_channels_last, "conv_clamp": conv_clamp, 
                    "fused_modconv_default": "inference_only",
                    }

                common_synthesis_kwargs["min_resolution"] = min_resolution

                common_mapping_kwargs = mapping_kwargs

                temp_mapping_kwargs = copy.deepcopy(common_mapping_kwargs)
                temp_mapping_kwargs.update(sr_mapping_kwargs)

                sr_mapping_kwargs = temp_mapping_kwargs
                # sr_mapping_kwargs["num_layers"] = 4

            with nullcontext("backbone config"):
                # Share
                if shared_resolution > 0:
                    self.shared_backbone = Gen2D(512, shared_resolution, feature_channels*3, 
                        mapping_kwargs=common_mapping_kwargs, synthesis_kwargs=common_synthesis_kwargs,
                        noise_config=noise_config, condition_config={})
                else:
                    self.shared_backbone = None

                alb_noise_config     = alb_noise_config if alb_noise_config is not None else tex_noise_config
                bmp_noise_config     = bmp_noise_config if bmp_noise_config is not None else tex_noise_config

                alb_condition_config = alb_condition_config if alb_condition_config is not None else tex_condition_config
                bmp_condition_config = bmp_condition_config if bmp_condition_config is not None else tex_condition_config

                self.tex_G = G(noise_config=tex_noise_config, condition_config=tex_condition_config)
                self.alb_G = G(noise_config=alb_noise_config, condition_config=alb_condition_config)
                self.bmp_G = G(noise_config=bmp_noise_config, condition_config=bmp_condition_config)
                self.tri_G = G(noise_config=tri_noise_config, condition_config=tri_condition_config)
                self.bcg_G = G(noise_config=bcg_noise_config, condition_config=bcg_condition_config)

                hair_cond_config = {k: code_config[k] for k in free_gs_mlp_key}
                self.hair_G = G(noise_config={}, condition_config=hair_cond_config)

            with nullcontext("eghead model"):
                eg_head_config = EGHeadConfig()
                eg_head_config.texture_res = tex_resolution 
                used_flame_cfg             = copy.deepcopy(model_cfg)
                used_flame_cfg.n_hair_code = np.sum(self.hair_G.condition_split)
                eg_head_config.flame_cfg   = used_flame_cfg 
                eg_head_config.dim_feature = self.field_config["plane_dim"]
                eg_head_config.field_kwargs= self.field_config
                eg_head_config.field_num_band = self.field_config["num_band"]

                eg_head_config.dec_hidden_dim = self.field_config["decoder_hidden_dim"]

                eg_head_config.head_gs_spp   = head_gs_spp
                eg_head_config.head_gs_init  = head_gs_init
                eg_head_config.head_gs_normal= head_gs_normal
                eg_head_config.head_gs_scale = head_gs_scale

                eg_head_config.free_gs_offset    = free_gs_offset
                eg_head_config.free_gs_scale     = (None, free_gs_scale)
                eg_head_config.free_gs_ratio     = free_gs_ratio
                eg_head_config.free_gs_lbs_order = "query_offset"

                eg_head_config.fast_init   = fast_init

                eg_head = EGHead(eg_head_config)
                eg_head.render_seg = True
                del eg_head.albedo
                del eg_head.bump
                del eg_head.field

                if free_gs_residual:
                    pass
                else:
                    del eg_head.free_prm
                    eg_head.free_prm = 0

                # eg_head.hair_prm  = 0
                # eg_head.hairg_prm = 0
                # eg_head.glass_prm = 0
                # eg_head.teeth_prm = 0
                # eg_head.cloth_prm = 0

                head_gs_color = eg_head.head_gs_color[..., :3]
                del eg_head.head_gs_color
                eg_head.register_buffer("head_gs_color", head_gs_color)

                self.register_buffer(f"free_xyz_fixed", eg_head.free_xyz.requires_grad_(False).clone().detach())
                    
                if render_streams is True:
                    eg_head.streams = []
                else:
                    eg_head.streams = None
                
                self.rgb_channels = eg_head.mat_channels

                self.feature_channels = feature_channels

            with nullcontext("backbone"):

                if self.shared_backbone is not None:
                    # sr common kwargs
                    srcond_common_kwargs = {"adain": True}

                    # Albedo/Bump
                    alb_synthesis_kwargs = copy.deepcopy(common_synthesis_kwargs)
                    alb_synthesis_kwargs["channel_max"] = tex_channel_max
                    bmp_synthesis_kwargs = copy.deepcopy(common_synthesis_kwargs)
                    bmp_synthesis_kwargs["channel_max"] = tex_channel_max

                    self.alb_sr = SRCond(tex_resolution, feature_channels, tex_resolution//shared_resolution, self.rgb_channels, 
                        z_dim=np.sum(self.alb_G.noise_split), c_dim=np.sum(self.alb_G.condition_split), 
                        mapping_kwargs=sr_mapping_kwargs, block_kwargs=alb_synthesis_kwargs, **srcond_common_kwargs)
                    self.bmp_sr = SRCond(tex_resolution, feature_channels, tex_resolution//shared_resolution, 1, 
                        z_dim=np.sum(self.bmp_G.noise_split), c_dim=np.sum(self.bmp_G.condition_split), 
                        mapping_kwargs=sr_mapping_kwargs, block_kwargs=bmp_synthesis_kwargs, **srcond_common_kwargs)

                    # Tri-plane
                    tri_synthesis_kwargs = copy.deepcopy(common_synthesis_kwargs)
                    tri_synthesis_kwargs["channel_max"] = tri_channel_max
                    self.tri_sr   = SRCond(tri_resolution, feature_channels, tri_resolution//shared_resolution, 3*feature_channels, 
                        z_dim=np.sum(self.tri_G.noise_split), c_dim=np.sum(self.tri_G.condition_split), 
                        mapping_kwargs=sr_mapping_kwargs, block_kwargs=tri_synthesis_kwargs, **srcond_common_kwargs)

                    # Background
                    bcg_synthesis_kwargs = copy.deepcopy(common_synthesis_kwargs)
                    bcg_synthesis_kwargs["min_resolution"] = back_min_resolution
                    bcg_synthesis_kwargs["num_fp16_res"]   = back_num_fp16
                    bcg_synthesis_kwargs["up_factor"]      = back_up_factor
                    bcg_synthesis_kwargs["channel_max"]    = back_channel_max
                    bcg_synthesis_kwargs["channel_base"]   = back_channel_base
                    if bcg_synthesis_kwargs["num_fp16_res"] > 0:
                        bcg_synthesis_kwargs["conv_clamp"] = 256
                    self.bcg_sr   = SRCond(img_resolution, feature_channels, img_resolution//shared_resolution, 3, 
                        z_dim=np.sum(self.bcg_G.noise_split), c_dim=np.sum(self.bcg_G.condition_split), 
                        mapping_kwargs=sr_mapping_kwargs, block_kwargs=bcg_synthesis_kwargs, **srcond_common_kwargs)
                
                else:
                    mapping_kwargs   = copy.deepcopy(common_mapping_kwargs)

                    # Albedo/Bump
                    tex_synthesis_kwargs = copy.deepcopy(common_synthesis_kwargs)
                    tex_synthesis_kwargs["channel_max"]    = tex_channel_max
                    tex_synthesis_kwargs["channel_base"]   = tex_channel_base
                    # tex_synthesis_kwargs["min_resolution"] = tex_min_resolution
                    self.tex_gen = Gen2D(512, tex_resolution, self.rgb_channels + 1, 
                        mapping_kwargs=mapping_kwargs, synthesis_kwargs=tex_synthesis_kwargs,
                        noise_config=tex_noise_config, condition_config=tex_condition_config)

                    # Tri-plane
                    tri_synthesis_kwargs = copy.deepcopy(common_synthesis_kwargs)
                    tri_synthesis_kwargs["channel_max"]  = tri_channel_max
                    tri_synthesis_kwargs["channel_base"] = tri_channel_base
                    tri_synthesis_kwargs["noise_mode"]   = tri_noise_mode
                    # tri_synthesis_kwargs["roll_out"]     = tri_roll_out

                    res = tri_resolution
                    self.tri_gen = Gen2D(512, res, 3*feature_channels, 
                        mapping_kwargs=mapping_kwargs, synthesis_kwargs=tri_synthesis_kwargs,
                        noise_config=tri_noise_config, condition_config=tri_condition_config)

                    # Background
                    bcg_synthesis_kwargs = copy.deepcopy(common_synthesis_kwargs)
                    bcg_synthesis_kwargs["min_resolution"] = back_min_resolution
                    bcg_synthesis_kwargs["num_fp16_res"]   = back_num_fp16
                    bcg_synthesis_kwargs["up_factor"]      = back_up_factor
                    bcg_synthesis_kwargs["channel_max"]    = back_channel_max
                    bcg_synthesis_kwargs["channel_base"]   = back_channel_base
                    if bcg_synthesis_kwargs["num_fp16_res"] > 0:
                        bcg_synthesis_kwargs["conv_clamp"] = 256

                    self.bcg_gen = Gen2D(512, back_resolution, self.rgb_channels, 
                        mapping_kwargs=mapping_kwargs, synthesis_kwargs=bcg_synthesis_kwargs,
                        sr_factor=back_sr_factor,
                        noise_config=bcg_noise_config, condition_config=bcg_condition_config)
                    
            self.background_activation = background_activation
            self.light_method = light_method

            # Shading
            with nullcontext("Shading"):
                cls_map = {key:kls for key, kls in zip(
                    ["Neural_AO_SH2",],
                    [      AO_SH2_MLP,],
                    )}
                light_code_sizee = np.prod(code_config["light"])

                if light_method in cls_map:
                    self.shading_net = cls_map[light_method](light_code_sizee)
                else:
                    self.shading_net = None

            
            self.train_step   = 0
            self.render_cache = {}

            self.gs_renderer  = eg_head

            self.streams      = [] if using_streams else None

            for k, v in eg_head.named_buffers():
                print(k, v.shape)

        @contextmanager
        def using_generated(self, decoded):
            if isinstance(self.gs_renderer, EGHead):
                try:
                    self.gs_renderer.albedo   = decoded["albedo"]
                    # self.renderer.bump     = torch.zeros_like(decoded["bump"])
                    self.gs_renderer.bump     = decoded["bump"]
                    self.gs_renderer.field    = decoded["field"]
                    yield decoded
                finally:
                    del self.gs_renderer.albedo
                    del self.gs_renderer.bump
                    del self.gs_renderer.field

        # @torch_compile()
        def gen_shared(self, b_z, b_c, alb_z, alb_c, bmp_z, bmp_c, tri_z, tri_c, bcg_z, bcg_c, **map_kwargs):

            bone, bone_ws = self.shared_backbone(b_z, b_c, **map_kwargs)
            tex_feat, tri_feat, bcg_feat = bone.split([self.feature_channels]*3, dim=1)

            device = b_z.device

            # texture
            with record_function("gen_texture"):
                texture, tex_ws = None, None

                if "tex" in self.render_cache:
                    albedo, bump = self.render_cache["tex"]
                    albedo, bump = albedo.expand(BS, -1, -1, -1), bump.expand(BS, -1, -1, -1)
                else:
                    alb_feat = tex_feat
                    bmp_feat = torch.cat([tex_feat[:, 3:4].expand(-1, 4, -1, -1), tex_feat[:, 4:]], dim=1)

                    albedo, albedo_ws = self.alb_sr(alb_feat, alb_z, alb_c, **map_kwargs)
                    bump,   bump_ws   = self.bmp_sr(bmp_feat, bmp_z, bmp_c, **map_kwargs)
                    
                    albedo = albedo.permute(0, 2, 3, 1)
                    bump   = bump.permute(0, 2, 3, 1)

                    albedo = sigmoid_clamp(albedo)
                    bump   = self.bump_range*torch.tanh(bump) # + resize_2d(self.fixed_bump, bump.size(1), bump.size(2))

                    # albedo, bump # [B,H,W,3] [B,H,W,1]

            # tri-plane
            with record_function("feature-plane"):
                if "tri" in self.render_cache:
                    planes_ws, planes = self.render_cache["tri"]
                    planes = planes.expand(BS, -1, -1, -1, -1)
                else:
                    planes, planes_ws = self.tri_sr(tri_feat, tri_z, tri_c, **map_kwargs)

                    planes = planes.view(len(planes), -1, self.feature_channels, planes.shape[-2], planes.shape[-1])

            # background
            with record_function("gen_background"):
                env_ws  = None
                if "back" in self.render_cache:
                    back_ws, back = self.render_cache["back"]
                    back = back.expand(BS, -1, -1, -1)
                else:
                    if self.environ_generator == "white":
                        background = torch.ones((1,1,1,3), device=device)
                    elif self.environ_generator == "black":
                        background = torch.zeros((1,1,1,3), device=device)
                    elif self.environ_generator == "synthesis":

                        back, back_ws = self.bcg_sr(bcg_feat, bcg_z, bcg_c, **map_kwargs)

                        back = back.permute(0, 2, 3, 1)                  # [B,H,W,C]
                        if self.background_activation == "sigmoid":
                            back = sigmoid_clamp(back)
                        else:
                            back = back.clamp(0, 1)

            # print(albedo.shape, bump.shape, planes.shape)
            return albedo, bump, planes, planes_ws, back, back_ws

        # @torch_compile()
        def gen_isolat(self, tex_z, tex_c, tri_z, tri_c, bcg_z, bcg_c, **synthesis_kwargs):

            device = tex_z.device

            default_stream = torch.cuda.default_stream(device)
            if self.streams is not None:
                if len(self.streams) == 0:
                    self.streams = [torch.cuda.Stream(device) for _ in range(8)]
                # sync   = lambda i : self.streams[i].wait_stream(default_stream)
                used_stream_index = set()
                def branch(i):
                    used_stream_index.add(i)
                    self.streams[i].wait_stream(default_stream)
                    return torch.cuda.stream(self.streams[i])
                def join(i):
                    default_stream.wait_stream(self.streams[i])
                def join_used():
                    for i in used_stream_index:
                        default_stream.wait_stream(self.streams[i])
            else:
                branch    = nullcontext
                join      = lambda i:i
                join_used = lambda : None

            BS = tex_z.size(0)
            # texture
            # self.streams[0].wait_stream(default_stream)
            # with record_function("gen_texture"),    torch.cuda.stream(self.streams[0]):
            with record_function("gen_texture"),    branch(0):
                texture, tex_ws = None, None

                if "tex" in self.render_cache:
                    albedo, bump = self.render_cache["tex"]
                    albedo, bump = albedo.expand(BS, -1, -1, -1), bump.expand(BS, -1, -1, -1)
                else:
                    texture, tex_ws = self.tex_gen(tex_z, tex_c, **synthesis_kwargs)

                    # print("non-share", texture.shape)

                    albedo, bump = torch.split(texture, [self.rgb_channels, 1], dim=1)
                    
                    albedo = albedo.permute(0, 2, 3, 1)
                    bump   = bump.permute(0, 2, 3, 1)

                    albedo = sigmoid_clamp(albedo)
                    bump   = self.bump_range*torch.tanh(bump) # + resize_2d(self.fixed_bump, bump.size(1), bump.size(2))

                    # albedo, bump # [B,H,W,3] [B,H,W,1]

            # tri-plane
            # self.streams[1].wait_stream(default_stream)
            # with record_function("gen_tri-plane"),  torch.cuda.stream(self.streams[1]):
            with record_function("gen_tri-plane"),  branch(1):
                if "tri" in self.render_cache:
                    planes_ws, planes = self.render_cache["tri"]
                    planes = planes.expand(BS, -1, -1, -1, -1)
                else:
                    planes, planes_ws = self.tri_gen(tri_z, tri_c, **synthesis_kwargs)

                    planes = planes.reshape(len(planes), -1, self.feature_channels, planes.shape[-2], planes.shape[-1])

            # background
            # self.streams[2].wait_stream(default_stream)
            # with record_function("gen_background"), torch.cuda.stream(self.streams[2]):
            with record_function("gen_background"), branch(2):
            # with record_function("gen_background"):
                back = back_ws = None

                if "back" in self.render_cache:
                    back_ws, back = self.render_cache["back"]
                    back = back.expand(BS, -1, -1, -1)
                else:
                    if self.environ_generator == "white":
                        back = torch.ones((1,1,1,3), device=device)
                    elif self.environ_generator == "black":
                        back = torch.zeros((1,1,1,3), device=device)
                    elif self.environ_generator == "synthesis":

                        back, back_ws = self.bcg_gen(bcg_z, bcg_c, **synthesis_kwargs)

                        if back.size(-1) != self.img_resolution:
                            back = F.interpolate(back, (self.img_resolution, self.img_resolution), 
                                mode="bilinear", align_corners=False, antialias=True)

                        back = back.permute(0, 2, 3, 1)                  # [B,H,W,C]

                        if self.background_activation == "sigmoid":
                            back = sigmoid_clamp(back)
                        else:
                            back = back.clamp(0, 1)

            join(0)
            join(1)
            join(2)
            return albedo, bump, tex_ws, planes, planes_ws, back, back_ws

        def pack_flame_param(self, data):

            zero = torch.zeros_like(data["pose"][..., :3])
            root_pose, neck_pose, jaw_pose = zero, zero, data["pose"][..., 3:]

            # if "hair_code" in data:
            #     hair_code = data["hair_code"]
            # else:
            #     hair_code = self.hair_G.pack_condition(data)

            flame_param = FLAMEParam(
                shape_params      = data["shape"],
                expression_params = data["exp"],
                pose_params       = torch.cat([root_pose, jaw_pose], dim=-1),
                neck_pose_params  = neck_pose,
                eye_pose_params   = data["eye_pose"],
                hair_code         = data["hair_code"],
            )

            return flame_param

        # @torch_compile()
        def render_3dgs(self, generated, c_dict, m2v, ndc, H=512, W=512):
            BS, device = m2v.size(0), m2v.device
            aux_d = {}

            with record_function("gen_ray"):
                ray_org, ray_dir = get_primary_ray(m2v, ndc, H=H, W=W)
                ray_dir = F.normalize(ray_dir, dim=-1)

                aux_d["ray_dir"] = ray_dir

            with self.using_generated(generated), record_function("render_image"), torch.cuda.amp.autocast(enabled=False):
                flame_params = self.pack_flame_param(c_dict)
                image, alpha, depth, normal, aux_d = self.gs_renderer.render_image(flame_params, m2v, ndc, H=H, W=W)

                # gs_img, gs_alp, gs_dep, gs_nrm, gs_aux = self.render_3dgs(fused_dict_h, verts, joint_transform, albedo, bump, planes=planes, H=H, W=W)
            
                # gs_img = gs_img + (1-gs_alp)*back

            if self.gs_renderer.render_seg:
                # segim, image = torch.split(image, [image.size(-1)-3, 3], dim=-1)
                num_seg = self.gs_renderer.head_gs_color.size(-1)
                segim, image = torch.split(image, [num_seg, image.size(-1)-num_seg], dim=-1)
                aux_d["segim"] = segim

            # shading
            with nullcontext("shading"):
                lgt_code = c_dict.get('light', None)
                shading  = None
                if lgt_code is not None:
                    
                    if self.light_method == "Shading_SH2":
                        shading = shading_SH2(normal.reshape(BS, -1, 3), lgt_code.reshape(BS, 9, 3)).reshape(normal.shape)
                    elif self.light_method == "Neural_AO_SH2":
                        shading = self.shading_net(normal, depth, lgt_code.reshape(BS, -1), m2v, ndc)
                        shading = shading.reshape(normal.shape)

                if shading is not None:
                    aux_d["unshade"] = image[..., :3]
                    image = image[..., :3] * shading
                    aux_d["shading"] = shading

                    aux_d["shading_sh"] = self.shading_net.get_sh(normal,  lgt_code)
                    aux_d["shading_ao"] = self.shading_net.get_ao(normal, depth, lgt_code, m2v, ndc)
                else:
                    image = image[..., :3]

            # background
            if "background" in generated:
                bg_im = generated["background"]
                image = image + (1-alpha)*bg_im

                aux_d["background"] = bg_im

            return image, alpha, depth, normal, aux_d

        # @torch_compile()
        def forward(self, noise, condition, truncation_psi=1, truncation_cutoff=None, update_emas=False, texture_modify=None, bump_modify=None, H=None, W=None, **synthesis_kwargs):
            synthesis_kwargs = dict(synthesis_kwargs)
            synthesis_kwargs["truncation_psi"]    = truncation_psi
            synthesis_kwargs["truncation_cutoff"] = truncation_cutoff
            synthesis_kwargs["update_emas"]       = update_emas

            BS = noise.size(0)
            H  = H if H is not None else self.img_resolution
            W  = W if W is not None else self.img_resolution
            device = noise.device

            if self.training:
                self.train_step += 1
            
            n_dict = self.unpack_noise(noise)
            c_dict = self.unpack_condition(condition)

            if "rot" not in c_dict:
                c_dict["rot"] = c_dict["m2v"][:, :3, :3].clone()

            fused_dict = n_dict.copy()
            fused_dict.update(c_dict)

            def swap_gen_cond(cond_swap_conf):
                fused_dict_r = {k:v for k, v in fused_dict.items()} # shallow copy
                if self.training:
                    for k, prob in cond_swap_conf.items():
                        if prob == 1:
                            v_cond = torch.zeros_like(fused_dict_r[k])
                        else:
                            # prob   = prob*min(1, self.train_step/(100*1000))

                            alpha = min(1, self.train_step/(self.cond_swap_decay*1000))
                            prob  = (1-alpha)*self.cond_swap_init_prob + alpha*prob   # decay from init_prob->target
                            v      = fused_dict_r[k].clone()
                            v_swap = torch.roll(v, 1, 0)
                            v_cond = torch.where(torch.rand((v.shape[0], *([1]*(v.ndim-1))), device=v.device) < prob, v_swap, v)
                        fused_dict_r[k] = v_cond
                else:
                    for k, prob in cond_swap_conf.items():
                        if prob == 1:
                            v_cond = torch.zeros_like(fused_dict_r[k])
                        else:
                            v_cond = fused_dict_r[k]
                        fused_dict_r[k] = v_cond
                    return fused_dict_r
                return fused_dict_r

            fused_dict_t = swap_gen_cond(self.cond_swap_config["tex"])  if "tex"  in self.cond_swap_config else fused_dict
            fused_dict_a = swap_gen_cond(self.cond_swap_config["albd"]) if "albd" in self.cond_swap_config else fused_dict
            fused_dict_b = swap_gen_cond(self.cond_swap_config["bump"]) if "bump" in self.cond_swap_config else fused_dict

            fused_dict_f = swap_gen_cond(self.cond_swap_config["trif"]) if "trif" in self.cond_swap_config else fused_dict
            fused_dict_h = swap_gen_cond(self.cond_swap_config["hair"]) if "hair" in self.cond_swap_config else fused_dict

            fused_dict_b = swap_gen_cond(self.cond_swap_config["back"]) if "back" in self.cond_swap_config else fused_dict

            tex_code = fused_dict['tex']
            
            with record_function("generate_maps"):
                SHARED_BACKBONE = self.shared_backbone is not None

                if SHARED_BACKBONE is True:
                    b_z, b_c     = self.shared_backbone.pack_noise(fused_dict), self.shared_backbone.pack_condition(fused_dict)

                    alb_z, alb_c = self.alb_G.pack_noise(fused_dict_a), self.alb_G.pack_condition(fused_dict_a)
                    bmp_z, bmp_c = self.bmp_G.pack_noise(fused_dict_b), self.bmp_G.pack_condition(fused_dict_b)

                    tri_z, tri_c = self.tri_G.pack_noise(fused_dict_f), self.tri_G.pack_condition(fused_dict_f)
                    bcg_z, bcg_c = self.bcg_G.pack_noise(fused_dict_b), self.bcg_G.pack_condition(fused_dict_b)
                

                    albedo, bump, planes, planes_ws, back, back_ws = self.gen_shared(b_z, b_c, alb_z, alb_c, bmp_z, bmp_c, tri_z, tri_c, bcg_z, bcg_c, **synthesis_kwargs)
                else:
                    tex_z, tex_c = self.tex_G.pack_noise(fused_dict_t), self.tex_G.pack_condition(fused_dict_t)

                    tri_z, tri_c = self.tri_G.pack_noise(fused_dict_f), self.tri_G.pack_condition(fused_dict_f)
                    bcg_z, bcg_c = self.bcg_G.pack_noise(fused_dict_b), self.bcg_G.pack_condition(fused_dict_b)

                    albedo, bump, tex_ws, planes, planes_ws, back, back_ws = self.gen_isolat(tex_z, tex_c, tri_z, tri_c, bcg_z, bcg_c, **synthesis_kwargs)

            if texture_modify is not None:
                texture_modify = resize_2d(texture_modify, albedo.size(1), albedo.size(2))
                albedo = torch.lerp(albedo, texture_modify[..., :albedo.size(-1)], texture_modify[..., -1:])

            if bump_modify is not None:
                bump_modify = resize_2d(bump_modify, bump.size(1), bump.size(2))
                bump = torch.lerp(bump, bump_modify[..., :bump.size(-1)], bump_modify[..., -1:])

            back = resize_2d(back, H, W)

            generated = {
                "albedo":     albedo,
                "bump":       bump,
                "field":      planes,
                "background": back,
            }

            c_dict["hair_code"] = self.hair_G.pack_condition(fused_dict_h)

            image, alpha, depth, normal, aux_d = self.render_3dgs(generated, c_dict, c_dict["m2v"], c_dict["ndc"], H=H, W=W)
            
            # 3DGS
            aux_dict = {
                "background_ws": back_ws,
                "background":    back,

                "texture_ws":    tex_ws,
                "albedo":        albedo,
                "bump":          bump,
                "field":         planes,
                "hair_code":     c_dict["hair_code"],

                "planes_ws":     planes_ws,
                "planes":        planes,
            }
            aux_dict.update(aux_d)
            aux_dict = {k:v for k, v in aux_dict.items() if v is not None}

            return image, alpha, depth, normal, aux_dict

with nullcontext("D"):

    class SegDiscriminator(torch.nn.Module):
        def __init__(self,
            c_dim,                          # Conditioning label (C) dimensionality.
            img_resolution,                 # Input resolution.
            img_channels,                   # Number of input color channels.
            seg_resolution,                 # Segment resolution
            seg_channels,                   # Number of input segment channels.
            seg_scale           = 1,
            seg_channels_scale  = 1,
            architecture        = 'resnet', # Architecture: 'orig', 'skip', 'resnet'.
            channel_base        = 32768,    # Overall multiplier for the number of channels.
            channel_max         = 512,      # Maximum number of channels in any layer.
            num_fp16_res        = 4,        # Use FP16 for the N highest resolutions.
            conv_clamp          = 256,      # Clamp the output of convolution layers to +-X, None = disable clamping.
            cmap_dim            = None,     # Dimensionality of mapped conditioning label, None = default.
            multi_res_indx      = [],
            block_kwargs        = {},       # Arguments for DiscriminatorBlock.
            mapping_kwargs      = {},       # Arguments for MappingNetwork.
            epilogue_kwargs     = {},       # Arguments for DiscriminatorEpilogue.
            fusion              = "mid",
            min_seg_dropout     = 0,
            seg_dropout_decay   = 1000,
            seg_dropout_initp   = 1,

            fp16_channels_last  = True,
            channels_last       = False, #True,

            normalize           = False,
            using_streams       = False,
        ):
            super().__init__()
            val_repr = []
            for k, v in zip(["c_dim", "img_resolution", "seg_channels", "img_channels", "architecture",
                            "channel_base", "channel_max", "num_fp16_res", "conv_clamp", "cmap_dim",
                            "block_kwargs", "mapping_kwargs", "epilogue_kwargs"],
                            [c_dim, img_resolution, seg_channels, img_channels, architecture, 
                             channel_base, channel_max, num_fp16_res, conv_clamp, cmap_dim,
                            block_kwargs, mapping_kwargs, epilogue_kwargs]):
                val_repr.append(f"{v}")
            self._ex_repr_str = ", ".join(val_repr)

            self.seg_scale = seg_scale

            self.channels_last = channels_last
            self.normalize     = normalize
            self.fusion = fusion
            self.min_seg_dropout   = min_seg_dropout
            self.seg_dropout_decay = seg_dropout_decay
            self.seg_dropout_init_prob   = seg_dropout_initp
            self.seg_dropout_target_prob = min_seg_dropout
            self.current_seg_dropout_p = 1
            self.c_dim = c_dim

            self.seg_resolution = seg_resolution
            self.seg_resolution_log2 = int(np.log2(seg_resolution))
            self.seg_channels = seg_channels
            self.seg_block_resolutions = [2 ** i for i in range(self.seg_resolution_log2, 2, -1)]

            self.img_resolution = img_resolution
            self.img_resolution_log2 = int(np.log2(img_resolution))
            self.img_channels = img_channels
            self.block_resolutions = [2 ** i for i in range(self.img_resolution_log2, 2, -1)]
            channels_dict = {res: min(channel_base // res, channel_max) for res in self.block_resolutions + [4]}
            seg_channels_dict = { k: int(v*seg_channels_scale) for k, v in channels_dict.items() }
            seg_channels_dict[4] = channels_dict[4]
            fp16_resolution = max(2 ** (self.img_resolution_log2 + 1 - num_fp16_res), 8)

            if cmap_dim is None:
                cmap_dim = channels_dict[4]
            if c_dim == 0:
                cmap_dim = 0
            if c_dim > 0:
                self.mapping = stylegan.MappingNetwork(z_dim=0, c_dim=c_dim, w_dim=cmap_dim, num_ws=None, w_avg_beta=None, **mapping_kwargs)

            common_kwargs     = dict(img_channels=img_channels, architecture=architecture, conv_clamp=conv_clamp)
            common_kwargs_seg = dict(img_channels=seg_channels, architecture=architecture, conv_clamp=conv_clamp)
            cur_layer_idx = 0
            for res in self.block_resolutions:
                in_channels = channels_dict[res] if res < img_resolution else 0
                tmp_channels = channels_dict[res]
                out_channels = channels_dict[res // 2]
                use_fp16 = (res >= fp16_resolution)
                block = stylegan.DiscriminatorBlock(in_channels, tmp_channels, out_channels, resolution=res,
                    first_layer_idx=cur_layer_idx, use_fp16=use_fp16, **block_kwargs, **common_kwargs, fp16_channels_last=fp16_channels_last)
                setattr(self, f'b{res}', block)
                cur_layer_idx += block.num_layers
            self.b4 = stylegan.DiscriminatorEpilogue(channels_dict[4], cmap_dim=cmap_dim, resolution=4, **epilogue_kwargs, **common_kwargs)

            cur_layer_idx = 0
            for res in self.seg_block_resolutions:
                if res <= self.seg_resolution:
                    in_channels =  seg_channels_dict[res] if res < self.seg_resolution else 0
                    tmp_channels = seg_channels_dict[res]
                    out_channels = seg_channels_dict[res // 2]
                    use_fp16 = (res >= fp16_resolution)
                    sblock = stylegan.DiscriminatorBlock(in_channels, tmp_channels, out_channels, resolution=res,
                        first_layer_idx=cur_layer_idx, use_fp16=use_fp16, **block_kwargs, **common_kwargs_seg, fp16_channels_last=fp16_channels_last)
                    setattr(self, f'sb{res}', sblock)

            if self.fusion == "late":
                self.sb4 = stylegan.DiscriminatorEpilogue(seg_channels_dict[4], cmap_dim=0, resolution=4, **epilogue_kwargs, **common_kwargs)

            # self.register_buffer('resample_filter', upfirdn2d.setup_filter([1,3,3,1]))
            # self.register_buffer("mean", torch.as_tensor((0.485, 0.456, 0.406)).reshape(1, 3, 1, 1))
            # self.register_buffer("std",  torch.as_tensor((0.229, 0.224, 0.225)).reshape(1, 3, 1, 1))

            if len(multi_res_indx) > 0:
                from .pg_modules.discriminator import SingleDiscCond, MultiScaleD
                
                channels    = [ channels_dict[res//2] for res in self.block_resolutions ]
                resolutions = [ res//2 for res in self.block_resolutions ]

                self.multi_res_disc = MultiScaleD(
                    channels=[ channels[i] for i in multi_res_indx ],
                    resolutions=[ resolutions[i] for i in multi_res_indx ],
                    num_discs=len(multi_res_indx)
                )
            else:
                self.multi_res_disc = None

            self.multi_res_indx = multi_res_indx

            self.train_steps = 0
            
            self.streams     = [] if using_streams else None

        # @torch_compile()
        # def forward(self, img, condition, update_emas=False, **block_kwargs):
        def forward(self, img, seg, condition, update_emas=False, **block_kwargs):
            if self.training:
                self.train_steps += 1
            # seg_dropout_p = max(self.min_seg_dropout, 1 - self.train_steps/(self.seg_dropout_decay * 1e3))

            alpha = min(self.train_steps / (self.seg_dropout_decay * 1e3), 1)
            seg_dropout_p = (1-alpha)*self.seg_dropout_init_prob + alpha*self.seg_dropout_target_prob

            self.current_seg_dropout_p = seg_dropout_p

            def unscaled_dropout(value, prob, training):
                if training:
                    return torch.where(torch.rand_like(value) < prob, 0, value)
                else:
                    return value

            img = img.permute(0, 3, 1, 2)
            seg = seg.permute(0, 3, 1, 2)
            # img = (img-self.mean) / self.std

            # seg, img = torch.split(img, (self.seg_channels, self.img_channels), dim=1)

            if self.channels_last:
                seg = seg.to(memory_format=torch.channels_last) 
                img = img.to(memory_format=torch.channels_last) 

            if self.normalize:
                seg = seg * 2 - 1
                img = img * 2 - 1

            device = img.device
            default_stream = torch.cuda.default_stream(device)
            if self.streams is not None:
                if len(self.streams) == 0:
                    self.streams = [torch.cuda.Stream(device) for _ in range(8)]
                # sync   = lambda i : self.streams[i].wait_stream(default_stream)
                def branch(i):
                    self.streams[i].wait_stream(default_stream)
                    return torch.cuda.stream(self.streams[i])
                def join(i):
                    default_stream.wait_stream(self.streams[i])
            else:
                branch = nullcontext
                join   = lambda i:i

            _ = update_emas # unused

            # segment, condition
            cmap = None
            # with record_function("Dseg_blocks"):
            with record_function("Dseg_blocks"), branch(0):
                if seg.shape[2:] != (self.seg_resolution, self.seg_resolution):
                    seg = F.interpolate(seg, (self.seg_resolution, self.seg_resolution), mode="bilinear", align_corners=False, antialias=True)
                x_ = None
                for res in self.seg_block_resolutions:
                    block = getattr(self, f'sb{res}')
                    x_, seg = block(x_, seg, **block_kwargs)
                    # print(f"return {res} seg", type(seg))

                with torch.cuda.amp.autocast(enabled=False):
                    if self.c_dim > 0:
                        cmap = self.mapping(None, condition)

            logits = None
            with record_function("Dimg_blocks"):
            # with record_function("Dimg_blocks"), branch(0):
                features = {}
                x = None
                for i, res in enumerate(self.block_resolutions):
                    block = getattr(self, f'b{res}')
                    x, img = block(x, img, **block_kwargs)

                    if i in self.multi_res_indx:
                        features[str(len(features))] = x

                if self.multi_res_disc is not None:
                    logits = self.multi_res_disc(features, None)

            join(0)
            # join(1)
            
            if self.fusion == "mid":
                x = x + unscaled_dropout(x_, seg_dropout_p, training=self.training)*self.seg_scale
                # x  = x + F.dropout(x_, seg_dropout_p, training=self.training)

            x = self.b4(x, img, cmap)

            if self.fusion == "late":
                x_ = self.sb4(x_, seg, None)
                x = x + unscaled_dropout(x_, seg_dropout_p, training=self.training)*self.seg_scale
                # x  = x + F.dropout(x_, seg_dropout_p, training=self.training)

            if logits is not None:
                x = x + torch.mean(logits, dim=-1, keepdims=True)
            return x

        def extra_repr(self):
            return self._ex_repr_str