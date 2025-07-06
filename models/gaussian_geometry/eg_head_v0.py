import os
import sys
import torch
import torch.nn as nn
import torch.nn.functional as F

import dataclasses
# from dataclasses import dataclass, field
from collections import namedtuple, OrderedDict

import copy
import math
import numpy as np
import pickle
from typing import Tuple, List, Dict

from contextlib     import nullcontext
from torch.profiler import record_function

ROOT = os.path.abspath(os.path.join(__file__, "..", "..", ".."))

from models                            import neural_represent
from models.FLAME                      import FLAME, FLAMETex
from utils                             import device_control
from utils.kplane_mlp                  import kplane_mlp
from utils.diff_gaussian_rasterization import make_diff_gaussian_rasterization

from models.networks_stylegan2 import FullyConnectedLayer

FLAMEParam = namedtuple("FLAMEParam", ["shape_params", "expression_params", "pose_params", "neck_pose_params", "eye_pose_params", "hair_code"])

diff_gaussian_rasterization_11 = make_diff_gaussian_rasterization(channels=11)
diff_gaussian_rasterization_14 = make_diff_gaussian_rasterization(channels=14)
diff_gaussian_rasterization_16 = make_diff_gaussian_rasterization(channels=16)
diff_gaussian_rasterization_17 = make_diff_gaussian_rasterization(channels=17)
diff_gaussian_rasterization_18 = make_diff_gaussian_rasterization(channels=18)
diff_gaussian_rasterization_19 = make_diff_gaussian_rasterization(channels=19) # [color:3 roughness:1 matallic:1] [segid:6] alpha:1 depth:1 xyz:3 nrm:3

torch_compile_or_jit = device_control.torch_compile_or_jit
torch_compile = device_control.compile

for t in ["bool", "int", "float", "complex", "object", "unicode", "str"]:
    if not hasattr(np, t):
        setattr(np, t, getattr(np, f"{t}_"))

def fix_param_as_buffer(model, name):
    p = getattr(model, name)
    delattr(model, name)
    model.register_buffer(name, p.data)

@torch_compile_or_jit
def face_normal(ver, tri):
    BS  = ver.size(0)

    faces   = ver[:, tri.flatten().long(), :].reshape(BS, -1, 3, 3)
    ori_face_nrm = torch.cross(faces[:,:,1]-faces[:,:,0], faces[:,:,2]-faces[:,:,0], dim=-1)
    return F.normalize(ori_face_nrm, dim=-1)

@torch_compile_or_jit
def vertex_normal(ver, tri):
    BS  = ver.size(0)

    tri = tri.long()

    ori_face_nrm = face_normal(ver, tri)

    v_nrm = torch.zeros_like(ver)
    v_nrm.scatter_add_(1, tri[None, :, 0, None].expand(BS, -1, 3), ori_face_nrm)
    v_nrm.scatter_add_(1, tri[None, :, 1, None].expand(BS, -1, 3), ori_face_nrm)
    v_nrm.scatter_add_(1, tri[None, :, 2, None].expand(BS, -1, 3), ori_face_nrm)
    return F.normalize(v_nrm, dim=-1)

@torch_compile_or_jit
def normal_from_pos(pos_2d):
    '''
    pos_2d: [BS, H, W, 3]
    '''
    u2d = F.pad(torch.diff(pos_2d, dim=2), (0, 0, 1, 0), "constant", 0.0)
    v2d = F.pad(torch.diff(pos_2d, dim=1), (0, 0, 0, 0, 1, 0), "constant", 0.0)
    nrm_2d = F.normalize(torch.cross(v2d, u2d, dim=-1), dim=-1)
    return nrm_2d, v2d, u2d

@torch_compile_or_jit
def sigmoid_clamp(value, eps:float = 0.001):
    return torch.sigmoid(value)*(1 + 2*eps) - eps

@torch_compile_or_jit
def nerf_opacity(value, scale:float = 1, bias:float = 1):
    return 1 - torch.exp(-F.softplus(value*scale + bias))

@torch_compile_or_jit
def resize_2d(img_bhwc, H:int, W:int):
    b, h, w, c = img_bhwc.shape

    if (h, w) != (H, W):
        img_bhwc = F.interpolate(img_bhwc.permute(0, 3, 1, 2), (H, W), 
            mode="bilinear", align_corners=False, antialias=True).permute(0, 2, 3, 1)
    return img_bhwc

@torch_compile_or_jit
def compute_triangle_area(v10, v20):
    if v10.size(-1) == 3:
        return 0.5*torch.linalg.norm(torch.cross(v10, v20, dim=-1), dim=-1)
    elif v10.size(-1) == 2:
        return 0.5*torch.det(torch.stack([v10, v20], dim=-2))

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

@torch_compile_or_jit
def normal2quater(n):
    # this is NOT the only solution
    # rot_vec = normalize(0.5*(n+[0,0,1]))
    # rot_vec = normalize(n+[0,0,1])
    rot_vec = torch.stack((n[..., 0], n[..., 1], n[..., 2]+1), dim=-1)
    rot_vec = F.normalize(rot_vec, dim=-1)

    gs_rot  = F.pad(rot_vec, (1, 0), "constant", 0.0)
    return gs_rot

@torch_compile_or_jit
def quater2normal(r):
    q = F.normalize(r, dim=-1)
    w = q[..., 0]
    x = q[..., 1]
    y = q[..., 2]
    z = q[..., 3]
    return torch.stack([2 * (x*z + w*y), 2 * (y*z - w*x), 1 - 2 * (x*x + y*y)], dim=-1)    # 3rd col vector
    # return torch.stack([2 * (x*z - w*y), 2 * (y*z + w*x), 1 - 2 * (x*x + y*y)], dim=-1)  # 3rd row vector

    # full quater2rotation
    # R[:, 0, 0] = 1 - 2 * (y*y + z*z)
    # R[:, 0, 1] = 2 * (x*y - r*z)
    # R[:, 0, 2] = 2 * (x*z + r*y)
    # R[:, 1, 0] = 2 * (x*y + r*z)
    # R[:, 1, 1] = 1 - 2 * (x*x + z*z)
    # R[:, 1, 2] = 2 * (y*z - r*x)
    # R[:, 2, 0] = 2 * (x*z - r*y)
    # R[:, 2, 1] = 2 * (y*z + r*x)
    # R[:, 2, 2] = 1 - 2 * (x*x + y*y)

    # R = [ [Xaxis_x, Yaxis_x, Zaxis_x],
    #       [Xaxis_y, Yaxis_y, Zaxis_y],
    #       [Xaxis_z, Yaxis_z, Zaxis_z], ]
    # x_axis = R[:, :, 0]
    # y_axis = R[:, :, 1]
    # y_axis = R[:, :, 2]

@torch_compile_or_jit
def quater2rotation(r):

    q = F.normalize(r, dim=-1)
    w = q[..., 0]
    x = q[..., 1]
    y = q[..., 2]
    z = q[..., 3]
    # return torch.stack([2 * (x*z + w*y), 2 * (y*z - w*x), 1 - 2 * (x*x + y*y)], dim=-1)

    # full quater2rotation
    # R = torch.empty(len(r), 3, 3, device=r.device)
    # R[:, 0, 0] = 1 - 2 * (y*y + z*z)
    # R[:, 0, 1] = 2 * (x*y - w*z)
    # R[:, 0, 2] = 2 * (x*z + w*y)
    # R[:, 1, 0] = 2 * (x*y + w*z)
    # R[:, 1, 1] = 1 - 2 * (x*x + z*z)
    # R[:, 1, 2] = 2 * (y*z - w*x)
    # R[:, 2, 0] = 2 * (x*z - w*y)
    # R[:, 2, 1] = 2 * (y*z + w*x)
    # R[:, 2, 2] = 1 - 2 * (x*x + y*y)

    R = torch.stack([
                 1 - 2 * (y*y + z*z),
                 2 * (x*y - w*z),
                 2 * (x*z + w*y),
                 2 * (x*y + w*z),
                 1 - 2 * (x*x + z*z),
                 2 * (y*z - w*x),
                 2 * (x*z - w*y),
                 2 * (y*z + w*x),
                 1 - 2 * (x*x + y*y),
                ], dim=-1).unflatten(-1, (3, 3))
    return R

def render_3dgs(cv_objpose, ndc_proj, gs_xyz, gs_rgb, gs_nrm, gs_opa, gs_sca, gs_rot, H=512, W=512, fovx=45, fovy=45, streams=None):
    # from diff_gaussian_rasterization import GaussianRasterizationSettings, GaussianRasterizer
    from torch.utils.cpp_extension import load  #~/.cache/torch_extensions/py310_cu117/GaussRasterize/
    N_COLOR   = gs_rgb[0].size(-1)
    N_CHANNEL = N_COLOR+8
    if N_COLOR == 3:
        GaussianRasterizationSettings = diff_gaussian_rasterization_11.GaussianRasterizationSettings
        GaussianRasterizer            = diff_gaussian_rasterization_11.GaussianRasterizer
    elif N_COLOR == 6:
        GaussianRasterizationSettings = diff_gaussian_rasterization_14.GaussianRasterizationSettings
        GaussianRasterizer            = diff_gaussian_rasterization_14.GaussianRasterizer
    elif N_COLOR == 8:
        GaussianRasterizationSettings = diff_gaussian_rasterization_16.GaussianRasterizationSettings
        GaussianRasterizer            = diff_gaussian_rasterization_16.GaussianRasterizer
    elif N_COLOR == 9:
        GaussianRasterizationSettings = diff_gaussian_rasterization_17.GaussianRasterizationSettings
        GaussianRasterizer            = diff_gaussian_rasterization_17.GaussianRasterizer
    elif N_COLOR == 10:
        GaussianRasterizationSettings = diff_gaussian_rasterization_18.GaussianRasterizationSettings
        GaussianRasterizer            = diff_gaussian_rasterization_18.GaussianRasterizer
    elif N_COLOR == 11:
        GaussianRasterizationSettings = diff_gaussian_rasterization_19.GaussianRasterizationSettings
        GaussianRasterizer            = diff_gaussian_rasterization_19.GaussianRasterizer

    # print("render 3dge")

    BS = len(gs_xyz)
    device = gs_xyz[0].device

    default_stream = torch.cuda.default_stream(device)
    if streams is not None:
        stream_list = streams # [streams[bi%len(streams)] for bi in range(BS)]
        # sync   = lambda i : self.streams[i].wait_stream(default_stream)
        used_stream_index = set()
        def branch(i):
            ii = i%len(stream_list)
            used_stream_index.add(ii)
            stream_list[ii].wait_stream(default_stream)
            return torch.cuda.stream(stream_list[ii])
        def join(i):
            ii = i%len(stream_list)
            default_stream.wait_stream(stream_list[ii])
        def join_used():
            for ii in used_stream_index:
                default_stream.wait_stream(stream_list[ii])
    else:
        branch    = nullcontext
        join      = lambda i:i
        join_used = lambda : None

    # Settings
    convert_SHs_python   = False
    tanfovx = np.tan( np.deg2rad(fovx)/2 ) # maybe it is NOT important ?
    tanfovy = np.tan( np.deg2rad(fovy)/2 ) # maybe it is NOT important ?

    flip_axes = torch.diag(torch.as_tensor([-1,1,-1], dtype=torch.float, device=device))
    # state_mean2d, state_radii = [], []
    image_l, alpha_l, depth_l, normal_l, state_l = [], [], [], [], []
    bg_color = torch.zeros(N_COLOR+8, device=device)
    for bi in range(BS):

        # if streams is not None:
        #     stream_list[bi].wait_stream(default_stream)

        #     cv_objpose.record_stream(stream_list[bi])
        #     ndc_proj.record_stream(stream_list[bi])
        #     bg_color.record_stream(stream_list[bi])

        #     # camera_centers.record_stream(stream_list[bi])

        #     gs_xyz.record_stream(stream_list[bi])
        #     gs_nrm.record_stream(stream_list[bi])
        #     gs_opa.record_stream(stream_list[bi])
        #     gs_sca.record_stream(stream_list[bi])
        #     gs_rot.record_stream(stream_list[bi])
        #     gs_rgb.record_stream(stream_list[bi])

        #     ctx_fn = lambda : torch.cuda.stream(stream_list[bi])
        # else:
        #     ctx_fn = lambda : nullcontext("")
        
        with branch(bi):
            gs_position = gs_xyz[bi]
            gs_normal   = gs_nrm[bi]
            gs_opacity  = gs_opa[bi]
            gs_scaling  = gs_sca[bi]
            gs_rotation = gs_rot[bi]
            override_color = gs_rgb[bi]

            gs_normal = F.normalize(gs_normal @ (flip_axes @ cv_objpose[bi,:3,:3]).T, dim=-1)

            world_view_transform = cv_objpose[bi].T
            projection_matrix    = ndc_proj[bi].T

            full_proj_transform = (world_view_transform.unsqueeze(0).bmm(projection_matrix.unsqueeze(0))).squeeze(0)
            camera_center       = world_view_transform.inverse()[3, :3]
            
            screenspace_points = torch.zeros_like(gs_position, requires_grad=True)
            try:
                screenspace_points.retain_grad()
            except:
                pass

            raster_settings = GaussianRasterizationSettings(
                image_height=H, image_width=W,
                tanfovx=tanfovx[bi], tanfovy=tanfovy[bi],
                bg=bg_color,
                scale_modifier=1,
                viewmatrix=world_view_transform, projmatrix=full_proj_transform,
                sh_degree=0,
                campos=camera_center,
                prefiltered=False, debug=False
            )

            rasterizer = GaussianRasterizer(raster_settings=raster_settings)

            means3D = gs_position
            means2D = screenspace_points
            opacity = gs_opacity

            scales = None
            rotations = None
            cov3D_precomp = None

            if gs_rotation.ndim == 3:
                R = gs_rotation
                L = torch.diag_embed(gs_scaling)
                L = R@L
                actual_covariance = L @ L.transpose(1, 2) # N, 3, 3
                cov3D_precomp = actual_covariance.flatten(1)[:, [0, 1, 2, 4, 5, 8]]
            else:
                scales    = gs_scaling
                rotations = gs_rotation

            shs = None
            colors_precomp = None
            if override_color is None:
                if convert_SHs_python:
                    shs_view = pc.get_features.transpose(1, 2).view(-1, 3, (pc.max_sh_degree+1)**2)
                    dir_pp = (pc.get_xyz - viewpoint_camera.camera_center.repeat(pc.get_features.shape[0], 1))
                    dir_pp_normalized = dir_pp/dir_pp.norm(dim=1, keepdim=True)
                    sh2rgb = eval_sh(pc.active_sh_degree, shs_view, dir_pp_normalized)
                    colors_precomp = torch.clamp_min(sh2rgb + 0.5, 0.0)
                else:
                    shs = gs_features
            else:
                colors_precomp = override_color

            # z-depth
            # gs_view_pos = F.pad(gs_position, (0, 1), "constant", 1) @ world_view_transform
            # m_d = F.pad(gs_view_pos[:, 2:3], (2, 0), "constant", 1)

            # dist-depth
            dist = torch.cdist(gs_position, camera_center[None]) # N, 1
            # m_d = F.pad(dist, (2, 0), "constant", 1)

            ones_dist = F.pad(dist, (1, 0), mode="constant", value=1)

            packed, radii = rasterizer(
                means3D = means3D,
                means2D = means2D,
                shs = shs,
                colors_precomp = torch.cat([colors_precomp, ones_dist, gs_position, gs_normal], dim=-1),
                opacities = opacity,
                scales = scales,
                rotations = rotations,
                cov3D_precomp = cov3D_precomp)
            
            # assert packed.size(0) == N_CHANNEL, f"{packed.shape} {N_CHANNEL}"

            image, mask, depth, pos, normal = torch.split(packed, (N_COLOR, 1, 1, 3, 3), dim=0)

            depth = depth / mask
            depth = torch.nan_to_num(depth, float('inf'))
            depth = torch.clamp(depth, torch.min(dist), torch.max(dist))

            d_pos = torch.cat([depth, pos], dim=0)

            image_l.append(image.permute(1,2,0))
            alpha_l.append(mask[0,:,:,None])
            # depth_l.append(depth[0,:,:,None])
            depth_l.append(d_pos.permute(1,2,0))
            normal_l.append(normal.permute(1,2,0))

            # if bi == 0:
            #     state_l.append([])
            #     state_l.append([])
            # state_l[0].append(means2D)
            # state_l[1].append(radii)
    
    join_used()

    for bi in range(BS):
        image_l[bi].contiguous()
        alpha_l[bi].contiguous()
        depth_l[bi].contiguous()
        normal_l[bi].contiguous()
    
    # if streams is not None:
    #     for s in stream_list:
    #         default_stream.wait_stream(s)
    return torch.stack(image_l), torch.stack(alpha_l), torch.stack(depth_l), torch.stack(normal_l), state_l

@dataclasses.dataclass
class EGHeadConfig:
    texture_res:       int = 512
    head_gs_spp:     float = 1
    head_gs_init:      str = "uniform"

    head_gs_opacity: float = 0.95
    head_gs_scale:   Tuple = (0.0006, 0.0004) # base, range
    head_gs_normal:    str = "simple"
    head_bump_range: float = 0.0025           # bump

    free_gs_ratio:    Dict = dataclasses.field(default_factory=dict)
    free_gs_scale:   Tuple = (0.0030, 0.0030) # base, range
    free_gs_offset:  float = 0.0500           # offset range
    free_gs_offset_b: bool = True             # offset free gs center by bumped_posmap - base_posmap

    # flame
    flame_cfg              = None
    flame_offset:    Tuple = (0.0, 0.0, 0.05)

    # field (tri-plane)
    dim_feature:       int = 32
    field_kwargs           = None
    field_num_band:    int = 0
    dec_hidden_dim:    int = 64
    triplane_index:  Tuple = None

    # misc
    fast_init:        bool = False

class EGHead(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = copy.deepcopy(config)

        texture_res = config.texture_res
        head_gs_spp = config.head_gs_spp
        head_gs_init = config.head_gs_init

        self.mat_channels = 3

        num_param = self.mat_channels + 4 + 3 + 1 # mat, quterion, scale, opacity

        self.drctx = None
        device = torch.device('cuda')

        # FLAME
        with nullcontext("head-GS"), torch.no_grad():
            self.register_buffer("flame_offset", torch.as_tensor(config.flame_offset).reshape(1, 1, 3))

            flame     = FLAME(config.flame_cfg)

            fix_param_as_buffer(flame, "eye_pose")
            fix_param_as_buffer(flame, "neck_pose")

            # flame_mod = pickle.load(open(config.flame_model_path, "rb"), encoding='latin1')
            flame_tex = np.load(config.flame_cfg.flame_tex_path)
            uv        = torch.as_tensor(flame_tex["vt"]).float()
            uv_tri    = torch.as_tensor(flame_tex["ft"].astype(np.int32)).int()

            uv[:, 1] = 1 - uv[:, 1]

            # self.register_buffer("tri",      torch.as_tensor(flame_mod["f"]).int())
            # self.register_buffer("tri_long", torch.as_tensor(flame_mod["f"]).long())
            self.register_buffer("tri",      flame.faces_tensor.int())
            self.register_buffer("tri_long", flame.faces_tensor.long())
            self.register_buffer("uv",       uv)
            self.register_buffer("uv_1_1",   2*uv-1)
            self.register_buffer("uv_tri",   uv_tri)

            self.flame = flame

            self.head_gs_opacity     = config.head_gs_opacity
            self.head_gs_scale_base, self.head_gs_scale_range = config.head_gs_scale

            self.free_gs_offset_range = config.free_gs_offset
            self.free_gs_scale_base, self.free_gs_scale_range = config.free_gs_scale

            RES    = int(texture_res*head_gs_spp)

            if config.fast_init:
                num_head_gs = RES*RES

                TEX_H, TEX_W = texture_res, texture_res

                self.register_buffer("gs2d_mask",   torch.empty( num_head_gs,   dtype=torch.bool     ))
                # self.register_buffer("gs2d_uv",     torch.empty( num_head_gs, 2, dtype=torch.float32 ))
                self.register_buffer("gs2d_uv_1_1", torch.empty( num_head_gs, 2, dtype=torch.float32 ))
                self.register_buffer("gs2d_ti",     torch.empty( num_head_gs,    dtype=torch.long    ))
                self.register_buffer("gs2d_w3",     torch.empty( num_head_gs, 3, dtype=torch.float32 ))
                self.register_parameter("gs_scale_delta", nn.Parameter(torch.empty( 1, num_head_gs, 3, dtype=torch.float32 )))
            
                self.register_buffer("fixed_bump", torch.empty( 1, TEX_H, TEX_W, 1,  dtype=torch.float32 ))
                self.register_buffer("area_base",  torch.empty( 1, self.tri.size(0), dtype=torch.float32 ))
            else:
                with nullcontext("fixed UV"):
                    head_gs_tex_epsilon = 1

                    ver  = (self.flame.v_template.unsqueeze(0)).to(device)
                    
                    area = self.get_area(ver).flatten()

                    faces_uv = uv[uv_tri.flatten().long(), :].reshape(-1, 3, 2).to(device)

                    EPS   = head_gs_tex_epsilon / (texture_res) if head_gs_spp != 1 else 0
                    v, u  = torch.meshgrid(torch.linspace(0+EPS, 1-EPS, RES), torch.linspace(0+EPS, 1-EPS, RES))
                    uv_2d = torch.stack([u.flatten(), v.flatten()], dim=-1).to(device)

                    def random_weight_by_area(num_sample):
                        index = torch.multinomial(area, num_sample, replacement=True)
                        rand2 = torch.rand((num_sample, 2))
                        sumr  = torch.sum(rand2, dim=-1, keepdim=True)
                        flip  = sumr>1
                        w0    = torch.abs(1-sumr)
                        w3    = torch.cat([w0, torch.where(flip, 1-rand2, rand2)], dim=-1)  # Ns, 3
                        gs_2d_area = torch.einsum("ni,nij->nj", w3.to(device), faces_uv[index])  # NS, 2
                        return gs_2d_area

                    if head_gs_init == "uniform":
                        gs_2d_base = uv_2d
                    elif head_gs_init == "loop_area":
                        # mesh Vertex, Edge, Face
                        edge = set()
                        for vi3 in uv_tri.detach().cpu().numpy():
                            vi3 = vi3.tolist()
                            for i in range(3):
                                e0 = min(vi3[i], vi3[(i+1)%3])
                                e1 = max(vi3[i], vi3[(i+1)%3])
                                edge.add((e0, e1))
                        edge = list(edge)
                        e0   = torch.as_tensor([e01[0] for e01 in edge], dtype=torch.long)
                        e1   = torch.as_tensor([e01[1] for e01 in edge], dtype=torch.long)

                        # (1,0,0) (.5,.5,0) (.5,0,.5)
                        # (2/3,1/6,1/6)

                        gs_2d_vfe   = torch.cat([
                                uv.to(device),                                                   # V
                                # loop sub divide odd point
                                (4*uv[uv_tri[0]] + uv[uv_tri[1]] + uv[uv_tri[2]]).to(device)/6,  # F
                                (uv[uv_tri[0]] + 4*uv[uv_tri[1]] + uv[uv_tri[2]]).to(device)/6,  # F
                                (uv[uv_tri[0]] + uv[uv_tri[1]] + 4*uv[uv_tri[2]]).to(device)/6,  # F
                                (uv[uv_tri[0]] + uv[uv_tri[1]] + uv[uv_tri[2]]).to(device)/3,    # F
                                (uv[e0] + uv[e1]).to(device)/2,                                  # E
                            ], dim=0)
                        num_area_gs = RES*RES - gs_2d_vfe.size(0)

                        # weight by area
                        gs_2d_area = random_weight_by_area(num_area_gs)
                        gs_2d_base = torch.cat([gs_2d_vfe, gs_2d_area], dim=0)

                    # # curvature
                    # import trimesh
                    # bump = torch.zeros(1, texture_res, texture_res, 1, device=device)
                    # pos_2d, nrm_2d, bumped_pos_2d, bumped_nrm_2d = self.get_bump(ver, bump)[:4]
                    # u2d  = F.pad(torch.diff(bumped_nrm_2d, dim=2), (0, 0, 1, 0), "constant", 0.0)
                    # v2d  = F.pad(torch.diff(bumped_nrm_2d, dim=1), (0, 0, 0, 0, 1, 0), "constant", 0.0)
                    # grad = (u2d.norm(dim=-1) + v2d.norm(dim=-1)).reshape(texture_res, texture_res).detach().cpu().numpy()
                    # i0   = torch.multinomial(torch.as_tensor(grad).flatten(), int(0.125*RES*RES))

                    # mesh = trimesh.Trimesh(ver.squeeze(0).detach().cpu().numpy(), self.tri.detach().cpu().numpy())
                    # curv = trimesh.curvature.discrete_gaussian_curvature_measure(mesh, pos_2d.reshape(-1, 3).detach().cpu().numpy(), 0.005) # 10mm
                    # curv = np.abs(curv.reshape(texture_res, texture_res))
                    # i1   = torch.multinomial(torch.as_tensor(curv).flatten(), int(0.125*RES*RES))
                    # gs_2d = torch.cat([gs_2d_base, uv_2d[i0], uv_2d[i1]], dim=0)
                    gs_2d = gs_2d_base

                    # gs in 2D
                    faces_uv= uv[uv_tri.flatten().long(), :].reshape(-1, 3, 2).to(device)
                    v10 = faces_uv[:,1] - faces_uv[:,0]                  # K, 2
                    v20 = faces_uv[:,2] - faces_uv[:,0]                  # K, 2

                    mask_list, uv2d_list, ti_list, w3_list = [], [], [], []
                    area = compute_triangle_area(v10, v20)
                    for uv2d in torch.split(gs_2d, 1024, dim=0):
                        vp0 = uv2d[:,None,:] - faces_uv[None,:,0]      # p, K, 2

                        w2 = compute_triangle_area(v10.unsqueeze(0).expand_as(vp0), vp0) / area[None, :] # p, K
                        w1 = compute_triangle_area(vp0, v20.unsqueeze(0).expand_as(vp0)) / area[None, :] # p, K

                        c0, c1, c2 = w1>=0, w2>=0, w2+w1<=1
                        condition = torch.logical_and(c2, torch.logical_and(c0, c1))

                        mask = condition.sum(dim=1) > 0             # p

                        ti = torch.argmax(condition.float(), dim=1) # p
                        w1_, w2_ = torch.gather(w1, 1, ti[:,None]), torch.gather(w2, 1, ti[:,None])
                        w3_ = torch.cat([1-w2_-w1_, w1_, w2_], dim=-1) # p,3

                        mask_list.append(mask) 
                        uv2d_list.append(uv2d)
                        ti_list.append(ti)
                        w3_list.append(w3_)

                    mask   = torch.cat(mask_list, dim=0).cpu()
                    f2d_uv = torch.cat(uv2d_list, dim=0).cpu()
                    ti,w3_ = torch.cat(ti_list, dim=0).cpu(), torch.cat(w3_list, dim=0).cpu()

                    scale_delta = torch.full_like(w3_.unsqueeze(0), 0)

                    print(f"mask {mask.sum()/mask.numel()*100:.2f}%")

                    self.register_buffer("gs2d_mask",   mask)        # RES**2
                    # self.register_buffer("gs2d_uv",     f2d_uv)      # RES**2, 2
                    self.register_buffer("gs2d_uv_1_1", 2*f2d_uv-1)  # RES**2, 2
                    self.register_buffer("gs2d_ti",     ti)          # RES**2
                    self.register_buffer("gs2d_w3",     w3_)         # RES**2, 3
                    self.register_parameter("gs_scale_delta", nn.Parameter(scale_delta.clone().detach())) # 1, RES**2, 3

                with nullcontext("fixed bump"), torch.enable_grad():
                    tex_h, tex_w = texture_res, texture_res
                    bump         = torch.nn.Parameter(torch.zeros(1, tex_h, tex_w, 1).requires_grad_(True).to(device))

                    ver  = (self.flame.v_template.unsqueeze(0)).to(device)
                    # ver  = (self.flame.v_template.unsqueeze(0)).to(device)

                    optm = torch.optim.AdamW([bump], lr=1e-4, weight_decay=1e-8)

                    _, nrm_2d, _, bumped_nrm_2d = self.get_bump(ver.requires_grad_(True), bump)[:4]
                    loss = (1 - F.cosine_similarity(nrm_2d, bumped_nrm_2d, dim=-1)).mean()
                    print(f"fitted bump init loss: {loss.item()}")

                    for i in range(1000):
                        _, nrm_2d, _, bumped_nrm_2d = self.get_bump(ver.requires_grad_(True), bump)[:4]

                        # loss = F.mse_loss(nrm_2d, normal_from_pos(pos_2d))
                        loss = (1 - F.cosine_similarity(nrm_2d, bumped_nrm_2d, dim=-1)).mean()
                        optm.zero_grad(set_to_none=True)
                        loss.backward()
                        optm.step()

                    print(f"fitted bump with loss: {loss.item()}, {bump.shape}")

                    # get area
                    area = self.get_area(ver)
                    
                    self.register_buffer("fixed_bump", bump.data.cpu().clone().detach())
                    self.register_buffer("area_base",  area.clone().detach())
        
            # for k in ["gs2d_mask", "gs2d_uv", "gs2d_uv_1_1", "gs2d_ti", "gs2d_w3", "gs_scale_delta", "fixed_bump", "area_base"]:
            for k in ["gs2d_mask", "gs2d_uv_1_1", "gs2d_ti", "gs2d_w3", "gs_scale_delta", "fixed_bump", "area_base"]:
                v = getattr(self, k)
                print(k, v.shape, v.dtype)

        # textures
        with nullcontext("textures"):
            TEX_H, TEX_W = texture_res, texture_res
            self.uv_pos  = None

            self.albedo = nn.Parameter(torch.zeros(1,   TEX_H, TEX_W, 3))
            self.bump   = nn.Parameter(torch.zeros(1,   TEX_H, TEX_W, 1))

        # field
        with nullcontext("field"):

            self.field_kwargs = {
                "box_warp":  0.5,
                "pe_scale":  1,
                "aggregate": "mul",
                "layout":    "channel"
            }
            if config.field_kwargs is not None:
                self.field_kwargs.update(config.field_kwargs)

            FIELD_FEAT_CHANNEL, FIELD_OUT_CHANNEL = config.dim_feature, num_param

            triplane_index = torch.as_tensor([[0, 1, 2], [0, 2, 1], [2, 1, 0]] if config.triplane_index is None else config.triplane_index, dtype=torch.long)
            triplane_orthi = triplane_index.reshape(3, 3)[:, 2]    # n_planes
            triplane_index = triplane_index.reshape(3, 3)[:, :2]   # n_planes, 2
            self.register_buffer("triplane_index", triplane_index.flatten().contiguous())
            self.register_buffer("triplane_orthi", triplane_orthi.flatten().contiguous())

            self.field    = nn.Parameter(torch.zeros(1, 2*TEX_H, 2*TEX_W, FIELD_FEAT_CHANNEL))

            DEC_HIDDEN_DIM     = config.dec_hidden_dim
            if config.field_num_band > 0:
                self.field_pos_emb = neural_represent.PositionEncoding(3, config.field_num_band)
            else:
                self.field_pos_emb = None

            def inv_sigmoid(y):
                y = y.clamp(0.001, 0.999)
                return torch.log(y) - torch.log(1-y)

            def make_dec_mlp():
                if self.field_pos_emb is not None:
                    field_dec_mlp = nn.Sequential(
                        nn.Linear(FIELD_FEAT_CHANNEL + self.field_pos_emb.dim_output, DEC_HIDDEN_DIM),
                        nn.Softplus(),
                        nn.Linear(DEC_HIDDEN_DIM, FIELD_OUT_CHANNEL),
                    )
                else:
                    field_dec_mlp = nn.Sequential(
                        nn.Linear(FIELD_FEAT_CHANNEL, DEC_HIDDEN_DIM),
                        nn.Softplus(),
                        nn.Linear(DEC_HIDDEN_DIM, FIELD_OUT_CHANNEL),
                    )
                # field_dec_mlp = nn.Sequential(
                #     FullyConnectedLayer(FIELD_FEAT_CHANNEL + self.field_pos_emb.dim_output, DEC_HIDDEN_DIM),
                #     nn.Softplus(),
                #     FullyConnectedLayer(DEC_HIDDEN_DIM, FIELD_OUT_CHANNEL),
                # )

                # num_param = self.mat_channels + 4 + 3 + 1 # mat, quterion, scale, opacity
                # mat
                mat = torch.zeros((self.mat_channels,))
                # nrm: 3 or 4
                nrm = torch.zeros((4,))
                nrm.fill_(0.2)
                nrm[-1] = 0.95
                # scale: 3
                scl = inv_sigmoid(torch.full((3,), 0.0005)/self.free_gs_scale_range)  # 5mm
                # opacity: 1
                opa = torch.full((1,), -1)   # make opacity small at initial
                bias_init = torch.cat([mat, nrm, scl, opa], dim=0)
                with torch.no_grad():
                    torch.nn.init.normal_(field_dec_mlp[-1].weight, mean=0.0, std=0.001)
                    torch.nn.init.normal_(field_dec_mlp[-1].weight[-3:], mean=0.0, std=0.0001)
                    field_dec_mlp[-1].bias.copy_(bias_init)
                return field_dec_mlp

            self.field_dec_mlp = make_dec_mlp()
            
        num_free = int(512**2)

        # free-space GS morphing
        in_c = config.flame_cfg.n_hair_code + 3
        self.fs_xyz_mlp = nn.Sequential(
            nn.Linear(in_c, 64),
            nn.ELU(),
            nn.Linear(64, 64),
            nn.ELU(),
            nn.Linear(64, num_free*3, bias=False),
            nn.Tanh(),
            nn.Unflatten(1, (num_free, 3)),
        )

        with open(config.flame_cfg.mask_path, "rb") as f:
            mask_data = pickle.load(f, encoding="latin1")
        import nvdiffrast.torch as dr
        drctx = dr.RasterizeCudaContext(device=device)
        
        nv = self.flame.v_template.size(0)
        def make_seg(key_conf, palette):
            v_mask_l = []
    
            pre_select = torch.zeros(nv, dtype=torch.bool)
            for key_list in key_conf:
                v_mask = torch.zeros(nv, dtype=torch.int32)
                for k in key_list:
                    v_mask.scatter_add_(0, torch.as_tensor(mask_data[k]), torch.ones(nv, dtype=torch.int32))
                
                v_mask_l.append( torch.logical_and(v_mask > 0, torch.logical_not(pre_select)) )
                pre_select = torch.logical_or(v_mask_l[-1], pre_select)
    
            v_mask = torch.stack(v_mask_l, dim=1)  # V, C

            v_clr = torch.einsum("vk,kc->vc", v_mask.float(), palette).unsqueeze(0).to(device)
            
            tri           = self.tri.to(device)
            uv, uv_tri    = self.uv.to(device), self.uv_tri.int().contiguous().to(device)

            rast, rast_db = dr.rasterize(drctx, F.pad(2*uv[None]-1, (0, 2), mode="constant", value=1), uv_tri, resolution=[1024, 1024])
            im_seg, _     = dr.interpolate(v_clr.contiguous(), rast, tri.int().contiguous(), rast_db)
            return im_seg

        def make_xyz(num_gs):
            bump_dist = 0.02
            rand_dist = 0.02/2
            part_keys = ["scalp", "forehead", "left_ear", "right_ear"]

            ver  = self.flame.v_template.unsqueeze(0).to(device)
            bump = torch.full((1, TEX_W, TEX_H, 1), bump_dist).to(device)
            bump = bump + resize_2d(self.fixed_bump, bump.size(1), bump.size(2)).to(device)
            pos_2d, nrm_2d, b_pos_2d, b_nrm_2d = self.get_bump(ver, bump)[:4]
            fs_xyz_uniform = b_pos_2d.flatten(1, 2).clone().detach().cpu() # 1, bH*bW, 3

            if len(part_keys) > 0:
                im_seg = make_seg([part_keys], torch.as_tensor([1.]).reshape(1, 1))

                fs_weight = F.interpolate(im_seg.reshape(1, 1, 1024, 1024), (TEX_H, TEX_W), mode="bilinear", align_corners=False).flatten()
                print(f"#GS={num_gs} weighted samlping {fs_weight.shape}")
            else:
                fs_weight = torch.ones_like(fs_xyz_uniform[0, :, 0])
                print(f"#GS={num_gs} uniform samlping")
            fs_weight.nan_to_num(nan=0.0, posinf=0.0, neginf=0.0)

            index = torch.multinomial(fs_weight.clamp_min(0), num_gs, replacement=True).to(fs_xyz_uniform.device)

            fs_sel = fs_xyz_uniform[:, index, :]
            fs_xyz = fs_sel + rand_dist*torch.randn_like(fs_sel).clamp(-1, 1)

            return fs_xyz.detach().cpu()

        def make_prm(num_gs):
            return torch.zeros((1, num_gs, FIELD_OUT_CHANNEL))

        self.free_xyz  = nn.Parameter(make_xyz(num_free).requires_grad_(True))
        self.free_prm  = nn.Parameter(make_prm(num_free).requires_grad_(True))

        with nullcontext("xyz offset"), torch.no_grad():
            ver  = self.flame.v_template.unsqueeze(0).to(device)
            bump = torch.full((1, TEX_W, TEX_H, 1), 0).to(device)
            bump = bump + resize_2d(self.fixed_bump, bump.size(1), bump.size(2)).to(device)
            pos_2d, nrm_2d, b_pos_2d, b_nrm_2d = self.get_bump(ver, bump)[:4]

            # self.register_buffer("base_pos_2d", b_pos_2d)

        with nullcontext("seg color"):
            seg_img = make_seg(
                [["right_eyeball", "left_eyeball"], ["face", "forehead", "scalp", "nose", "right_ear", "left_ear", "neck", "boundary"]], 
                torch.as_tensor([
                    [1, 0, 0],
                    [0, 1, 0],
                ]).float())
            
            eye_mask = seg_img[..., :1] > 0
            seg_img  = torch.cat([eye_mask, torch.logical_not(eye_mask), torch.full_like(eye_mask, False)], dim=-1).float()

            # print(seg_img.shape, seg_img.dtype, seg_img.max())
            # import cv2
            # cv2.imwrite(os.path.join(ROOT, "temp", "seg_img.png"), (255*seg_img[0].detach().cpu().numpy()).astype(np.uint8))

            # pver_uv = self.gs2d_uv
            # uv_coord = 2*pver_uv.reshape(1,1,-1,2)-1
            uv_coord = self.gs2d_uv_1_1.reshape(1,1,-1,2)
            def sample_uv(img_2d):
                value = F.grid_sample(img_2d.permute(0, 3, 1, 2), uv_coord, mode="bilinear", padding_mode="border", align_corners=False) # BS, C, 1, N
                return value.flatten(2).transpose(1, 2) # BS, N, C

            head_gs_color = sample_uv(seg_img[..., :2].to(uv_coord.device)).detach().cpu().contiguous()  # 1, N, C
            num_face_type = head_gs_color.size(-1)

            self.register_buffer("head_gs_color",  F.pad(head_gs_color, (0, 3), "constant", 0))
            self.register_buffer("free_gs_color",  torch.as_tensor([0, 0, 1], dtype=torch.float32).reshape(1, 1, -1))

            print("GS seg mask", num_face_type, {k:getattr(self, f"free_gs_color", None)})

        with nullcontext("overall render setting"):
            # modify
            self.gs_opacity_multiplier = 1
            self.gs_scale_multiplier   = 1
            self.render_free_gs_part   = ["free"]

            # render
            self.render_seg      = False
            self.fixed_intrinsic = True

            # grad
            self.gaussian_position_grad_scale = 1

            # whether use cuda stream, None for disabled, [] for enabled
            # self.streams = None
            self.streams = []
            
        region_gs = {
            "face": self.gs2d_mask.sum().item()
        }
        for k in ["free"]:
            if hasattr(self, f"{k}_xyz"):
                xyz = getattr(self, f"{k}_xyz")
                region_gs[k] = xyz.numel()//3

        for k, n in region_gs.items():
            print(f"[{k:^9}] #GS={n} ")
        print(f"total GS: {np.sum(list(region_gs.values()))/1000:.2f}K")

    def get_area(self, ver):
        faces_xyz = ver[:, self.tri.flatten().long(), :].unflatten(1, (-1, 3))  # 1, F, 3, 3
        v10 = faces_xyz[:,:,1] - faces_xyz[:,:,0]                  # 1, F, 3
        v20 = faces_xyz[:,:,2] - faces_xyz[:,:,0]                  # 1, F, 3

        area = compute_triangle_area(v10, v20)                     # 1, F
        return area

    def get_bump(self, ver, bump):
        '''
        bump: B,H,W,1
        '''
        import nvdiffrast.torch as dr

        BS, device = ver.size(0), ver.device
        tex_h,tex_w= bump.size(1), bump.size(2)
        if self.drctx is None:
            self.drctx = dr.RasterizeCudaContext(device=device)

        uv, uv_tri    = self.uv_1_1.to(device), self.uv_tri.to(device)
        tri, tri_long = self.tri.int().to(device), self.tri_long.to(device)

        v_nrm = vertex_normal(ver, tri_long)
        # rast, rast_db = dr.rasterize(self.drctx, F.pad(2*uv[None].expand(BS,-1,-1)-1, (0, 2), mode="constant", value=1), uv_tri, resolution=[tex_h, tex_w])
        rast, rast_db = dr.rasterize(self.drctx, F.pad(uv[None].expand(BS,-1,-1), (0, 2), mode="constant", value=1), uv_tri, resolution=[tex_h, tex_w])
        im_pos, _     = dr.interpolate(ver.contiguous(), rast, tri, rast_db)
        nrm_2d, _     = dr.interpolate(v_nrm.contiguous(), rast, tri, rast_db)

        nrm_2d = F.normalize(nrm_2d, dim=-1)
        b_im_pos = im_pos + bump * nrm_2d

        b_nrm_2d, v_tang_2d, u_tang_2d = normal_from_pos(b_im_pos)

        return im_pos, nrm_2d, b_im_pos, b_nrm_2d, v_tang_2d, u_tang_2d
    
    def get_nearest_point(self, query, mesh_v, mesh_t, eps=1e-16):
        from kaolin.ops.mesh import index_vertices_by_faces
        from kaolin.metrics.trianglemesh import point_to_mesh_distance

        batch_size, num_points = query.shape[:2]

        face_vertices = index_vertices_by_faces(mesh_v, mesh_t)
        distance, index, dist_type = point_to_mesh_distance(query, face_vertices)

        closest_triangle_vertices = torch.gather(face_vertices, dim=1, index=index[:, :, None, None].expand(-1, -1, 3, 3)).view(batch_size, num_points, 3, 3)

        # Compute barycentric embedding of every point into the closest triangle
        a, b, c = closest_triangle_vertices[:, :, 0, :], closest_triangle_vertices[:, :, 1, :], closest_triangle_vertices[:, :, 2, :]   # (batch_size, num_points, 3)
        q = query

        v1 = b-a
        v2 = c-a
        n = torch.cross(v1, v2)
        n_dot_n = torch.sum(n*n, dim=-1)
        n_dot_n[n_dot_n < eps] = 1.0
        
        w = q - a
        gamma = torch.sum(torch.cross(v1, w)*n, dim=-1)/n_dot_n
        beta = torch.sum(torch.cross(w, v2)*n, dim=-1)/n_dot_n
        alpha = 1.0-beta-gamma
        return index, torch.stack((alpha, beta, gamma), dim=-1)

    
    def field_2d_querypos(self, coordinates):
        '''
        coordinates (B,N,3) -> field_query_point (B,K,N,2)
        '''
        box_warp    = self.field_kwargs['box_warp']
        layout      = self.field_kwargs.get('layout', 'channel')

        B, N, _     = coordinates.shape

        coordinates = coordinates

        triplane_querypos = coordinates[..., self.triplane_index].unflatten(-1, (3, 2))             # B,N,3,2 
        triplane_querypos = (2/box_warp) * torch.einsum("bnkc->bknc", triplane_querypos)

        if layout == "flat":
            uv_querypos    = xyz2uv(coordinates).unsqueeze(1)
            # print(triplane_querypos.shape, uv_querypos.shape)
            plane_querypos = torch.cat([triplane_querypos, uv_querypos], dim=1)
        else:
            plane_querypos = triplane_querypos

        return plane_querypos
    
    def field_feature_mod(self, features, coordinates, mod_type="mul"):

        coordinates = coordinates

        orth_coordinates = coordinates[..., self.triplane_orthi] # B,N,3
        orth_coordinates = orth_coordinates.transpose(1, 2)      # B,3,N

        feature_mod = torch.cat([orth_coordinates, torch.linalg.norm(coordinates, dim=-1).unsqueeze(1)], dim=1) # B,4,N

        if mod_type == "mul":
            features = features * feature_mod.unsqueeze(-1)
        
        return features
    
    def field_point_query(self, coordinates):
        # 3d coordinates WITHOUT offset
        # it will be offset in field_2d_querypos
        padding_mode = 'border'
        align_corners= False
        pe_scale     = self.field_kwargs.get('pe_scale', 1)
        aggregate    = self.field_kwargs.get("aggregate", "mul")
        field_query  = self.field_kwargs.get("query", "naive")
        # modulate     = self.field_kwargs.get("modulate",  None)
        box_warp     = self.field_kwargs['box_warp']
        layout       = self.field_kwargs.get("layout", "channel")

        field         = self.field.expand(len(coordinates), -1, -1, -1, -1)
        N, K, C, H, W = field.shape

        if layout == "channel":
            if field_query == "naive":
                # plane_features = self.field.permute(0, 3, 1, 2).reshape(N*3, C//3, H, W)
                plane_features = field.reshape(N*K, C, H, W)
                plane_querypos = self.field_2d_querypos(coordinates).reshape(N*K, 1, -1, 2)

                features = F.grid_sample(plane_features, plane_querypos,
                    padding_mode=padding_mode, align_corners=align_corners).permute(0, 3, 2, 1).reshape(N, K, -1, C)

                if aggregate == "avg":
                    features = features.mean(1)
                elif aggregate == "mul":
                    features = features.prod(1)

                if self.field_pos_emb is not None:
                    # pos_feat = self.field_pos_emb(coordinates) # B, N, F   # simplest
                    # pos_feat = self.field_pos_emb((coordinates)*(pe_scale*2/box_warp)) # B, N, F   # to match with EGG3D
                    pos_feat = self.field_pos_emb((coordinates)*(pe_scale*2/box_warp)) # B, N, F   # to match with EGG3D

                    features = torch.cat([features, pos_feat], dim=-1)
            
            elif field_query == "kplane":
                plane_querypos = self.field_2d_querypos(coordinates).reshape(N, K, 1, -1, 2)
                if self.field_pos_emb is not None:
                    pos_feat   = self.field_pos_emb((coordinates)*(pe_scale*2/box_warp)) # B, N, F   # to match with EGG3D
                else:
                    pos_feat   = coordinates[...,:0]

                feat_fusion   = aggregate

                features = kplane_mlp(
                    field, None,
                    plane_querypos.permute(0, 2, 3, 1, 4), pos_feat.unsqueeze(1),
                    padding_mode=padding_mode, align_corners=align_corners, feature_fusion=feat_fusion,
                    mlp_layers=0, mlp_dim_hidden=0, mlp_dim_output=0, mlp_activation="softplus",           # this is disabled since (mlp_layers=0)
                    ).reshape(N, -1, C+pos_feat.size(-1))

        out = self.field_dec_mlp(features)
        return out

    def forward_flame(self, flame_param):
        # for k, v in flame_param._asdict().items():
        #     print("flame_param", k, v.shape)

        # rest_verts = self.flame(
        #     shape_params      = flame_param.shape_params,
        #     expression_params = flame_param.expression_params,
        #     pose_params       = torch.zeros_like(flame_param.pose_params),
        #     neck_pose_params  = torch.zeros_like(flame_param.neck_pose_params),
        #     eye_pose_params   = torch.zeros_like(flame_param.eye_pose_params),
        #     )[0]

        # betas      = torch.cat([flame_param.shape_params, flame_param.expression_params], dim=1)
        # rest_verts = self.flame.v_template.unsqueeze(0) + torch.einsum("bl,mkl->bmk", betas, self.flame.shapedirs)
        
        kwargs = {k:v for k,v in flame_param._asdict().items() if k not in ["hair_code"]}
        verts, landmarks2d, landmarks3d, joint_transform = self.flame(
            **kwargs,
            with_joint_transform=True
        )
        verts = verts + self.flame_offset.reshape(1, 1, 3)
        return verts, landmarks3d, joint_transform

    def render_image(self, flame_param, m2v, ndc, H:int=512, W:int=512):
        BS = len(m2v)

        aux_d = {}

        # ver[kx3] @ m2v.T @ ndc.T

        # if self.render_method == "3dgs":
        #     reverse_y = torch.diag(torch.as_tensor([-1,1,-1,1], dtype=torch.float, device=device))
        # elif self.render_method == "2dgs":
        #     reverse_y = torch.diag(torch.as_tensor([1,-1,-1,1], dtype=torch.float, device=device))
        # elif self.render_method == "mip_3dgs":
        #     reverse_y = torch.diag(torch.as_tensor([-1,1,-1,1], dtype=torch.float, device=device))
        reverse_y  = torch.diag(torch.as_tensor([-1,1,-1,1], dtype=torch.float, device=m2v.device))
        render_m2v = reverse_y.unsqueeze(0) @ m2v 
        render_ndc = ndc

        # this matrix will render nothing
        # reverse_y  = torch.diag(torch.as_tensor([1,-1,1,1], dtype=torch.float, device=m2v.device))
        # render_m2v = m2v 
        # render_ndc = ndc @ reverse_y

        camera_center = m2v.inverse()[:, :3, 3] # BS, 3

        with nullcontext("forward_3dmm"):
            flame_vertex, flame_lmk3d, joint_transform = self.forward_flame(flame_param)
            
            flame_vertex = flame_vertex

            aux_d["vertex"]   = flame_vertex
            aux_d["triangle"] = self.tri_long

            homo_v = F.pad(flame_vertex, (0, 1), "constant", 1)
            proj_v = (homo_v @ (m2v.transpose(-1, -2) @ (ndc @ reverse_y).transpose(-1, -2)))
            # proj_v = (homo_v @ (m2v.transpose(-1, -2) @ (reverse_y.unsqueeze(0) @ ndc).transpose(-1, -2)))
            aux_d["projected_vertex"] = proj_v
        
        with nullcontext("camera ray"):
            ray_org, ray_dir = get_primary_ray(m2v, ndc, H=H, W=W)
            ray_dir = F.normalize(ray_dir, dim=-1)

            aux_d["ray_origin"]    = ray_org
            aux_d["ray_direction"] = ray_dir

        # face GS
        with record_function("face GS"):
            ti      = self.gs2d_ti[self.gs2d_mask]
            # w3_     = self.gs2d_w3[self.gs2d_mask]
            pver_uv = self.gs2d_uv_1_1[self.gs2d_mask]             # N, 2

            # print("face GS", self.gs2d_mask.shape, ti.shape, ti[:5], self.bump.min(), self.bump.max())

            bump = self.bump
            bump = bump + resize_2d(self.fixed_bump, bump.size(1), bump.size(2))

            pos_2d, nrm_2d, b_pos_2d, b_nrm_2d, v_tang_2d, u_tang_2d = self.get_bump(flame_vertex, bump)

            # get area
            area = self.get_area(flame_vertex)
            area_scale = area / self.area_base                         # B, F

            # aux_d["uv_base_pos"] = pos_2d
            # aux_d["uv_base_nrm"] = nrm_2d
            # aux_d["uv_bump_pos"] = b_pos_2d
            # aux_d["uv_bump_nrm"] = b_nrm_2d

            uv_coord = (pver_uv.reshape(1,1,-1,2)).expand(BS,-1,-1,-1)

            pos3_nrm3_mat = torch.cat(list(map(lambda t:t.permute(0, 3, 1, 2), [b_pos_2d, b_nrm_2d, self.albedo.expand(BS,-1,-1,-1)])), dim=1)

            p_n_m = F.grid_sample(pos3_nrm3_mat, uv_coord,
                padding_mode="border", align_corners=False).permute(0, 3, 2, 1).reshape(pos3_nrm3_mat.size(0), -1, pos3_nrm3_mat.size(1))
            
            gs_xyz, gs_nrm, gs_mat = torch.split(p_n_m, [b_pos_2d.size(-1), b_nrm_2d.size(-1), self.albedo.size(-1)], dim=-1)
            gs_nrm = F.normalize(gs_nrm, dim=-1)

            gs_opa = torch.full_like(gs_xyz[...,:1], self.head_gs_opacity)
            gs_sca = (self.head_gs_scale_base + self.head_gs_scale_range*self.gs_scale_delta[:, self.gs2d_mask].tanh()).expand_as(gs_xyz)

            # area_scale = area_scale[:, ti].abs().sqrt() # B, N_gs
            area_scale = area_scale[:, ti] # B, N_gs
            gs_sca  = torch.cat([gs_sca[..., :2]*area_scale.unsqueeze(-1), gs_sca[..., 2:]], dim=-1)

            if self.config.head_gs_normal == "simple":
                gs_rot  = normal2quater(gs_nrm) # TODO: local frame matrix to quanterion
            elif self.config.head_gs_normal == "matrix":
                tang_u = F.grid_sample(u_tang_2d.permute(0,3,1,2), uv_coord,
                    padding_mode="border", align_corners=False).permute(0, 3, 2, 1).reshape(BS, -1, 3)
                tang_v = F.grid_sample(v_tang_2d.permute(0,3,1,2), uv_coord,
                    padding_mode="border", align_corners=False).permute(0, 3, 2, 1).reshape(BS, -1, 3)
                tang_u = F.normalize(tang_u, dim=-1)
                tang_v = F.normalize(tang_v, dim=-1)
                matrix = torch.stack([tang_u, tang_v, gs_nrm], dim=-2)   # z=cross(y, x)
                # gs_rot = matrix2quater(matrix)
                gs_rot = matrix

            aux_d["fs_xyz"] = gs_xyz

        with record_function("free-space GS"):
            free_xyz, free_prm = self.free_xyz.expand(BS, -1, -1), self.free_prm
            aux_d["free_xyz_rest"] = free_xyz
            aux_d["free_prm_rest"] = free_prm
            
        with record_function("join free space GS"):
            if self.render_seg:
                # print(self.head_gs_color[:, self.gs2d_mask].shape, gs_mat.shape)
                gs_mat = torch.cat([self.head_gs_color[:, self.gs2d_mask].expand(gs_mat.size(0), -1, -1), gs_mat], dim=-1)

            render_gaussian = {
                "xyz": [gs_xyz],
                "rot": [gs_rot],
                "sca": [gs_sca],
                "opa": [gs_opa],
                "rgb": [gs_mat],
                "nrm": [gs_nrm],
            }

            for k in self.render_free_gs_part:
                fs_xyz = aux_d[f"{k}_xyz_rest"]

                # print("hair_code", flame_param.hair_code.shape)
                code   = torch.cat([flame_param.hair_code.expand(BS, -1), flame_param.pose_params[:, 3:]], dim=-1)
                fs_xyz = fs_xyz + self.free_gs_offset_range*self.fs_xyz_mlp(code)

                aux_d[f"{k}_xyz_query"] = fs_xyz
                fs_prm = self.field_point_query(fs_xyz) + aux_d[f"{k}_prm_rest"]

                fs_rgb, fs_rot, fs_sca, fs_opa = torch.split(fs_prm, (self.mat_channels, 4, 3, 1), dim=-1)

                aux_d[f"{k}_xyz"] = fs_xyz
                aux_d[f"{k}_prm"] = fs_prm

                # original activation
                # rotaiton: normalize
                # scaling:  exp
                # opacity:  sigmoid
                # fs_xyz = fs_xyz + self.flame_offset
                fs_rot = F.normalize(fs_rot, dim=-1)

                fs_nrm = quater2normal(fs_rot)
            
                fs_rgb = sigmoid_clamp(fs_rgb)
                fs_sca = torch.sigmoid(fs_sca) * self.free_gs_scale_range
                fs_opa = nerf_opacity(fs_opa, 1, -2)

                if self.render_seg:
                    gs_color = getattr(self, f"free_gs_color")
                    fs_rgb   = torch.cat([gs_color.expand(fs_rgb.size(0), fs_rgb.size(1), -1), fs_rgb], dim=-1)
                
                render_gaussian["xyz"].append(fs_xyz)
                if gs_rot.ndim == 4:
                    fs_rot = quater2rotation(fs_rot)
                render_gaussian["rot"].append(fs_rot)
                render_gaussian["sca"].append(fs_sca)
                render_gaussian["opa"].append(fs_opa)
                render_gaussian["rgb"].append(fs_rgb)
                render_gaussian["nrm"].append(fs_nrm)
            
            # for k, l in render_gaussian.items():
            #     print(k, [c.shape for c in l])

            # rgs_xyz = torch.cat(render_gaussian["xyz"], dim=1) + self.flame_offset
            rgs_xyz = torch.cat(render_gaussian["xyz"], dim=1)
            rgs_rgb = torch.cat(render_gaussian["rgb"], dim=1)
            rgs_nrm = torch.cat(render_gaussian["nrm"], dim=1)
            rgs_opa = torch.cat(render_gaussian["opa"], dim=1) * self.gs_opacity_multiplier
            rgs_sca = torch.cat(render_gaussian["sca"], dim=1) * self.gs_scale_multiplier
            rgs_rot = torch.cat(render_gaussian["rot"], dim=1)

            # print("xyz", rgs_xyz.min(), rgs_xyz.max())
            # print("rgb", rgs_rgb.min(), rgs_rgb.max())
            # print("nrm", rgs_nrm.min(), rgs_nrm.max())
            # print("opa", rgs_opa.min(), rgs_opa.max())
            # print("sca", rgs_sca.min(), rgs_sca.max())
            # print("rot", rgs_rot.min(), rgs_rot.max())

            if self.gaussian_position_grad_scale != 1:
                rgs_xyz = rgs_xyz.requires_grad_(True)
                rgs_xyz.register_hook(lambda g: g*self.gaussian_position_grad_scale)

        with record_function("render"):
            if self.fixed_intrinsic:
                fov_d = 2*np.rad2deg(np.arctan(1/8.5294))   # fx = 1/tan(fov/2)
                fovx_d = fovy_d = fov_d
            else:
                fx    = ndc[:, 0, 0]
                fy    = ndc[:, 1, 1]
                fovx_d = 2*np.rad2deg(np.arctan(1/fx.detach().cpu().numpy()))   # fx = 1/tan(fov/2)
                fovy_d = 2*np.rad2deg(np.arctan(1/fy.detach().cpu().numpy()))   # fx = 1/tan(fov/2)

            fovx_d = [fovx_d]*BS if np.isscalar(fovx_d) else fovx_d
            fovy_d = [fovy_d]*BS if np.isscalar(fovy_d) else fovy_d

            # print(rgs_xyz.shape, rgs_rgb.shape, rgs_nrm.shape, rgs_opa.shape, rgs_sca.shape, rgs_rot.shape)

            image, alpha, depth, normal, state = render_3dgs(render_m2v, render_ndc, rgs_xyz, rgs_rgb, rgs_nrm, rgs_opa, rgs_sca, rgs_rot, H=H, W=W, fovx=fovx_d, fovy=fovy_d)

            # normal = F.normalize(normal, dim=-1)
            
            aux_d["gs_attr"] = (rgs_xyz, rgs_rgb, rgs_nrm, rgs_opa, rgs_sca, rgs_rot)  # (xyz, rgb, nrm, opa, sca, rot)
            aux_d["unshade"] = image
        return image, alpha, depth, normal, aux_d

if __name__ == "__main__":

    fs_rot = torch.randn(2, 4, 3)
    fs_sca = torch.rand(2, 4, 3)
    
    axis   = max_scale_axis(fs_rot, fs_sca)              # BS,N,3

    print(axis)