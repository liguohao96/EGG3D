'''
application Img-3D VAE with freezed SD VAE encoder
'''
import os
import sys
import platform
import subprocess

host = platform.node()
plat = f"{platform.python_version()}"
p    = subprocess.run([sys.executable, "-c", "import torch;print(torch.__version__)"], 
    stdout=subprocess.PIPE, stderr=subprocess.PIPE)
tver = p.stdout.decode().rstrip()

assert 4 < len(tver) < 20, f"got torch.__version__ = '{tver}', by {sys.executable}, {p.stderr.decode()}"

os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
os.environ["TORCH_EXTENSIONS_DIR"] = os.path.join(os.environ["HOME"], f"TORCH_EXT_egg3d_open_{host}_{plat}_{tver}")

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributed   as dist
import numpy as np
import random

import json
import yaml
import imageio

from tqdm import tqdm
from contextlib import nullcontext
from collections import defaultdict

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))

sys.path.insert(0, ROOT)
from load_network    import load_GAN

from utils           import device_control, chunk_fn, near_far_from_sphere
from utils.graphics.mesh2sd   import Mesh2SignedDistanceFunction, point_mesh_distance
from utils.graphics.sh        import rotate_sh
sys.path.pop(0)

def resize_2d(image, H=512, W=512):
    if image.shape[1:2] != (H, W):
        image = F.interpolate(image.permute(0, 3, 1, 2), (H, W), mode="bilinear", align_corners=False).permute(0, 2, 3, 1)
    return image

def get_class(type_name):
    sys.path.insert(0, ROOT)

    import importlib
    mod_name = ".".join(type_name.split(".")[:-1])
    obj_name = type_name.split(".")[-1]

    mod = importlib.import_module(mod_name)
    print(mod_name, mod)

    sys.path.pop(0)
    return getattr(mod, obj_name)

def main(args):

    device_control.make_deterministic(0)

    with nullcontext("Config"):
        import yaml
        import zipfile

        if args.config is not None:
            with open(args.config) as f:
                config = yaml.load(f, yaml.FullLoader)
        elif args.checkpoint.endswith(".zip"):
            with zipfile.ZipFile(args.checkpoint) as zipf:
                with zipf.open("config.yaml") as f:
                    config = yaml.load(f, yaml.FullLoader)
        
        print(config)
        
        SAVE_ROOT = os.path.join(ROOT, "temp", config["name"], "noise_to_3d")
        os.makedirs(SAVE_ROOT, exist_ok=True)

    device = torch.device("cuda:0")

    G = load_GAN(args.checkpoint, config).eval().to(device)

    # TODO: find the optimal tri-plane query implement
    # G.gs_renderer.field_kwargs["query"] = "naive"
    G.gs_renderer.field_kwargs["query"] = "kplane"

    # reset floater free_gs point to 0, based on the signed distance to mesh
    with torch.inference_mode():
        flame   = G.gs_renderer.flame
        flame_o = G.gs_renderer.flame_offset.squeeze(0)
        fs_xyz  = G.gs_renderer.free_xyz
        aux_tri = np.loadtxt(os.path.join(ROOT, "Data", "FLAME2020", "additional_tri.txt"))
        aux_tri = torch.as_tensor(aux_tri, dtype=flame.faces_tensor.dtype, device=device)

        tri   = torch.cat([flame.faces_tensor, aux_tri], dim=0)
        ver   = flame.v_template + flame_o
        faces = ver[tri.flatten()].reshape(1, -1, 3, 3) # B,Nf,3,3

        dis, ind = Mesh2SignedDistanceFunction.apply(faces, fs_xyz)
        # dis = point_mesh_distance(fs_xyz, ver.unsqueeze(0), tri, signed=True)

        # remove floaters in a naive way, this may reduce image quality
        fs_xyz.copy_(torch.where((dis > 0.02)[...,None].expand(-1,-1,3), torch.zeros_like(fs_xyz), fs_xyz)) # 

    TRAIN_RES = config["model"]["G"]["kwargs"]["img_resolution"]

    # condition dataset
    with torch.inference_mode():
        datasets = {}
        for k, d_cfg in config["data"].items():
            m_type   = d_cfg.get("type")
            m_args   = d_cfg.get("args",   list())
            m_kwargs = d_cfg.get("kwargs", dict())

            D = get_class(m_type)(*m_args, **m_kwargs)

            print(f"{k} {len(D)}")

            datasets[k] = D

        all_cond_dict = defaultdict(list)
        for (file_path, label) in datasets["FFHQ"].labels:
            for k, v in label.items():
                all_cond_dict[k].append(v)
        
        all_cond_dict = {k: torch.stack(v, dim=0).to(device) for k, v in all_cond_dict.items()}

        for k, v in all_cond_dict.items():
            print(k, v.shape, v.dtype)

    conddic_list = []
    decoded_list = []

    # generate
    with torch.inference_mode():
        z_shape = []
        if hasattr(G, "z_dim"):
            z_shape.append(G.z_dim)
        if hasattr(G, "z_shape"):
            z_shape = G.z_shape
        
        BS = 2
        for i in range(8):
            noise     = torch.randn(BS, G.z_dim, device=device)

            indx      = np.random.randint(len(all_cond_dict["shape"]), size=(BS,))
            cond_dict = {k:v[indx] for k, v in all_cond_dict.items()}
            cond      = G.pack_condition(cond_dict).to(device)

            print(noise.shape, cond.shape)
            pixel_render, pixel_alpha, pixel_depth, pixel_normal, aux = G(noise, cond, truncation_psi=0.7, truncation_cutoff=14)

            # if i < 10:
            if True:
                img_list = [pixel_render, aux["unshade"], aux["albedo"], 0.5+0.5*pixel_normal]
                if "shading" in aux:
                    img_list.append(aux["shading"])

                depth = pixel_depth[..., :1]
                depth = (depth-depth.min()) / (depth.max()-depth.min())
                img_list.append(depth.expand(-1,-1,-1,3))

                img_list.append(pixel_depth[..., 1:])

                log_im = torch.cat(img_list, dim=2)
                log_im = log_im.flatten(0, 1).clamp(0, 1).detach().cpu().numpy()

                imageio.imwrite(os.path.join(SAVE_ROOT, f"log-{i}.png"), (255*log_im).astype(np.uint8))
            
            for bi in range(cond.size(0)):
                generate_dict = {k: aux[k][bi:bi+1] for k in ["albedo", "bump", "field", "background", "hair_code"]}
                # generate_dict["m2v"] = cond_dict["m2v"]
                # generate_dict["ndc"] = cond_dict["ndc"]
                decoded_list.append(generate_dict)
                conddic_list.append({k: v[bi:bi+1] for k, v in cond_dict.items()})
        
    H = W = TRAIN_RES
    # editting
    with torch.inference_mode():
        for i in range(len(decoded_list)):
            decoded = dict(decoded_list[i])
            decoded["background"] = torch.ones((1, 1, 1, 3), device=device) # reset background to pure white

            cond_dict = dict(conddic_list[i])

            # novel exp
            nexp_list = []
            for vi in range(7):
                indx       = np.random.randint(len(all_cond_dict["shape"]))
                cond_dict_ = dict(cond_dict)
                cond_dict_["hair_code"] = decoded["hair_code"]
                cond_dict_["exp"]       = all_cond_dict["exp"][indx].to(device).unsqueeze(0)
                cond_dict_["jaw_pose"]  = all_cond_dict["jaw_pose"][indx].to(device).unsqueeze(0)

                gen_ret = G.render_3dgs(decoded, cond_dict_, cond_dict_["m2v"], cond_dict_["ndc"], H=H, W=W)
                nexp_list.append(gen_ret[0])

            # novel view
            azimuth_deg = [45, 30, 15, 0, -15, -30, -45]
            nview_list = []
            # for vi in tqdm(range(len(azimuth_deg)), desc="orbit"):
            for vi in range(len(azimuth_deg)):
                c = np.cos(np.deg2rad(azimuth_deg[vi]))
                s = np.sin(np.deg2rad(azimuth_deg[vi]))

                m2v = np.array([
                    [c, 0,-s,  0],
                    [0, 1, 0,  0],
                    [s, 0, c, -1.2],
                    [0, 0, 0,  1],
                ])

                m2v = torch.as_tensor(m2v, dtype=torch.float32, device=device)[None]

                cond_dict_ = dict(cond_dict)
                cond_dict_["hair_code"] = decoded["hair_code"]
                cond_dict_["m2v"]       = m2v

                gen_ret = G.render_3dgs(decoded, cond_dict_, cond_dict_["m2v"], cond_dict_["ndc"], H=H, W=W)
                nview_list.append(gen_ret[0])

            # novel light
            n_light   = 7
            light_deg = np.linspace(0, 180, n_light+1)[:n_light]
            nlight_list = []
            # for vi in tqdm(range(len(light_deg)), desc="light"):
            for vi in range(len(light_deg)):
                c = np.cos(np.deg2rad(light_deg[vi]))
                s = np.sin(np.deg2rad(light_deg[vi]))

                R = np.array([
                    [c,-s, 0],
                    [s, c, 0],
                    [0, 0, 1],
                ])

                origin_light = cond_dict["light"]
                rotated_light = rotate_sh(origin_light, torch.as_tensor([R], dtype=torch.float32))

                cond_dict_ = dict(cond_dict)
                cond_dict_["hair_code"] = decoded["hair_code"]
                cond_dict_["light"]     = rotated_light

                gen_ret = G.render_3dgs(decoded, cond_dict_, cond_dict_["m2v"], cond_dict_["ndc"], H=H, W=W)
                nlight_list.append(gen_ret[0])

            ne_im = torch.cat(nexp_list,   dim=2).flatten(0, 1)
            nv_im = torch.cat(nview_list,  dim=2).flatten(0, 1)
            nl_im = torch.cat(nlight_list, dim=2).flatten(0, 1)

            log_im = torch.cat([ne_im, nv_im, nl_im], dim=0).clamp(0, 1).detach().cpu().numpy()
            imageio.imwrite(os.path.join(SAVE_ROOT, f"editting_{i}.png"), (255*log_im).astype(np.uint8))

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint",  type=str, help='', required=False, default=None)
    parser.add_argument("--config",      type=str, help='', required=False, default=None)

    args = parser.parse_args()
    main(args)