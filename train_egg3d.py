'''
train EGG3D "Generating Editable Head Avatars with 3D Gaussian GANs"
'''
import os
import sys
import platform
import subprocess
ROOT = os.path.join(os.path.abspath("."))

host = platform.node()
plat = f"{platform.python_version()}"
p    = subprocess.run([sys.executable, "-c", "import torch;print(torch.__version__)"], 
    stdout=subprocess.PIPE, stderr=subprocess.PIPE)
tver = p.stdout.decode().rstrip()

assert 4 < len(tver) < 20, f"got torch.__version__ = '{tver}', by {sys.executable}, {p.stderr.decode()}"

os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
os.environ["TORCH_EXTENSIONS_DIR"] = os.path.join(os.environ["HOME"], f"TORCH_EXT_egg3d_{host}_{plat}_{tver}")

import json
import yaml
import time
import shutil

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.profiler import record_function
# import torch.autograd.profiler as profiler

import torchvision

torch.hub.set_dir(os.path.join(ROOT, "Data", "torch", "hub"))
os.environ["HF_HOME"] = os.path.join(ROOT, "Data", "huggingface")
os.environ["HUGGINGFACE_HUB_CACHE"] = os.path.join(ROOT, "Data", "huggingface", "hub")

from torch.utils.tensorboard import SummaryWriter

import cv2
import copy
from tqdm import tqdm
from glob import glob
from collections import defaultdict, OrderedDict
from contextlib  import nullcontext, contextmanager
from functools   import reduce
from easydict    import EasyDict

sys.path.insert(0, ROOT)
from datasets import IndexDataset, JoinDataset
from utils import device_control
from utils import chunk_fn
# from utils.graphics.raytrace import PrimaryRayFunction
from utils.graphics.sh import rotate_sh
from utils.graphics.mesh2sd  import Mesh2SignedDistanceFunction, mesh2sd_impl, point_mesh_distance

from utils.training_dynamics import MagnitudeTracker
import models.networks_stylegan2 as stylegan

from metric import get_fid
sys.path.pop(0)

SEED = 210
device_control.make_deterministic(SEED)

def get_class(type_name):
    import importlib
    mod_name = ".".join(type_name.split(".")[:-1])
    obj_name = type_name.split(".")[-1]

    mod = importlib.import_module(mod_name)
    print(mod_name, mod)

    return getattr(mod, obj_name)

def resize_2d(img_bhwc, H, W):
    b, h, w, c = img_bhwc.shape

    if (h, w) != (H, W):
        img_bhwc = F.interpolate(img_bhwc.permute(0, 3, 1, 2), (H, W), mode="bilinear", align_corners=False).permute(0, 2, 3, 1)
    return img_bhwc

def total_variance_loss(image, n_sample=1000, window_width=1):
    '''
    image: (B, H, W, C)
    '''
    device = image.device

    initial_coordinates = 2*torch.rand((image.shape[0], n_sample, 2), device=device) - 1
    perturbed_coordinates = initial_coordinates + torch.randn_like(initial_coordinates)*(window_width/image.size(1))

    all_coordinates = torch.cat([initial_coordinates, perturbed_coordinates], dim=1)

    value = F.grid_sample(image.permute(0, 3, 1, 2), all_coordinates.unflatten(1, (-1, 1)), mode="bilinear", align_corners=False)
    value = value.flatten(-2).transpose(-2, -1)

    value_initial   = value[:, :value.shape[1]//2]
    value_perturbed = value[:, value.shape[1]//2:]

    l_tv = F.l1_loss(value_initial, value_perturbed)
    return l_tv

def draw_from_gaussian(mu, lv):
    return  mu + torch.exp(lv/2)*torch.randn_like(mu)

def kl_normal_loss(mu, lv):
    return 0.5*(mu.square() + lv.exp() - lv - 1).sum(dim=-1).mean()

def R1(output, data):
    scalar_out = output.sum()
    grad = torch.autograd.grad(outputs=scalar_out, inputs=data, 
        create_graph=True, only_inputs=True)[0]
    
    loss = grad.pow(2).flatten(1, -1).sum(1).mean()
    return loss

@contextmanager
def eval_model(*model_list):
    try:
        for m in model_list:
            m.eval()
        yield model_list
    finally:
        for m in model_list:
            m.train()

@contextmanager
def phase_train_ctx(phase, record_time=False):
    try:
        phase.start_event.record()
        # for pg in phase.opt.param_groups:
        #     params = pg['params']
        #     for p in params:
        #         p.requires_grad_(True)
        for p in phase.all_params:
            p.requires_grad_(True)

        yield phase.amp, phase.opt, phase.get("lr_scheduler", None)
    finally:
        # for pg in phase.opt.param_groups:
        #     params = pg['params']
        #     for p in params:
        #         p.requires_grad_(False)
        for p in phase.all_params:
            p.requires_grad_(False)
        phase.end_event.record()
        # if record_time:
        #     phase.end_event.synchronize()
        #     phase.time_measure.append(phase.start_event.elapsed_time(phase.end_event))

def grid_image(image_list, n_col=1, h=512, w=512):
    if isinstance(image_list, (tuple, list)):
        img = torch.cat(image_list, dim=0)
    else:
        img = image_list
    
    n_r = len(image_list) // n_col
    img = torchvision.utils.make_grid(img.reshape(len(image_list), h, w, -1).permute(0, 3, 1, 2),
                                         scale_each=False,
                                         normalize=False,
                                         nrow=n_r)
    return img.permute(1, 2, 0)

def train(args):
    start_time = time.time()

    if args.debug:
        torch.autograd.detect_anomaly(True)
    
    # speed up a lot for Greg/Dreg
    torch.backends.cudnn.benchmark     = True

    deterministic = args.deterministic
    low_precision = not deterministic

    # numerical accuracy
    torch.backends.cudnn.deterministic    = deterministic
    torch.use_deterministic_algorithms(deterministic, warn_only=True)

    # torch.set_float32_matmul_precision("high")
    torch.set_float32_matmul_precision("medium")

    torch.backends.cuda.matmul.allow_tf32 = low_precision
    torch.backends.cudnn.allow_tf32       = low_precision
    # torch.backends.cuda.matmul.allow_fp16_reduced_precision_reduction = low_precision

    with open(args.config) as f:
        config = yaml.load(f, yaml.FullLoader)

    SAVE_ROOT = os.path.join(ROOT, "temp", config["name"])
    os.makedirs(SAVE_ROOT, exist_ok=True)

    device = device_control.init_device(args.enable_ddp)
    device_control.ddp_enabled = args.enable_ddp

    print(f"SAVE_ROOT @ {SAVE_ROOT}")

    np.random.seed(        SEED + device_control.get_rank())
    torch.manual_seed(     SEED + device_control.get_rank())
    torch.cuda.manual_seed(SEED + device_control.get_rank())

    with nullcontext("writer"):

        device_control.local_print(f'run name: {config["name"]}')

        log_dir = os.path.join(ROOT, "Data", "log", config["name"])
        if device_control.get_rank() == 0:
            os.makedirs(log_dir, exist_ok=True)
            writer = SummaryWriter(log_dir)

            import zipfile
            with zipfile.ZipFile(os.path.join(SAVE_ROOT, "code.zip"), 'w') as zipfile:
                def log_file(file_path, save_name):
                    with open(file_path, "r") as f:
                        content = f.read()
                        md_str  = f"## Python Code\n\n<pre><code>\n{content}</pre></code>\n"
                        writer.add_text(save_name, md_str, 0)

                        zipfile.writestr(save_name, content)
                    # shutil.copyfile(file_path, os.path.join(SAVE_ROOT, save_name))

                log_file(__file__, "train.py")
                log_file(os.path.join(ROOT, "models", "EGG3Dv0.py"), "GANs.py")
                log_file(os.path.join(ROOT, "models", "SRs.py"), "SRs.py")
                log_file(args.config, "config.yaml")
        else:
            writer = None

    data_cfg  = config["data"]
    model_cfg = config["model"]
    train_cfg = config["train"]
    phase_cfg = config.get("phase", dict())
    loss_cfg  = config.get("loss", dict())
    optim_cfg = config["optim"]
    
    # if args.max_length is not None:
    #     data_cfg["max_length"] = args.max_length
    if args.batch_size is not None:
        train_cfg["batch_size"] = args.batch_size

    with nullcontext("dataset"):
        datasets = {}
        for k, d_cfg in data_cfg.items():
            m_type   = d_cfg.get("type")
            m_args   = d_cfg.get("args",   list())
            m_kwargs = d_cfg.get("kwargs", dict())

            D = get_class(m_type)(*m_args, **m_kwargs)

            print(f"{k} {len(D)}")

            datasets[k] = D
        train_dataset = IndexDataset(JoinDataset(datasets))

        if args.max_length is not None and args.max_length < len(train_dataset):
            train_dataset = torch.utils.data.Subset(train_dataset, range(args.max_length))

        eval_dataset = copy.deepcopy(train_dataset)

        # cond_dataset = copy.deepcopy(datasets["FFHQ"])
        # labels = datasets["FFHQ"].labels

        all_cond_dict = defaultdict(list)
        for (file_path, label) in datasets["FFHQ"].labels:
            for k, v in label.items():
                all_cond_dict[k].append(v)
        
        all_cond_dict = {k: torch.stack(v, dim=0).to(device) for k, v in all_cond_dict.items()}

        for k, v in all_cond_dict.items():
            print(k, v.shape, v.dtype)

    NOISE_SCALE = 1

    model_dict = {}
    with nullcontext("model"):
        for k, m_cfg in model_cfg.items():
            
            if "ema" in k:
                ref_k = m_cfg
                m_cfg = copy.deepcopy(model_cfg[ref_k])
                m_cfg["kwargs"]["fast_init"] = True

            m_type   = m_cfg.get("type")
            m_args   = m_cfg.get("args",   list())
            m_kwargs = m_cfg.get("kwargs", dict())

            M = get_class(m_type)(*m_args, **m_kwargs)
            M.train()
            model_dict[k] = M.requires_grad_(False)
        
        z_shape = []
        if "G" in model_dict:
            if hasattr(model_dict["G"], "z_dim"):
                z_shape.append(model_dict["G"].z_dim)
            if hasattr(model_dict["G"], "z_shape"):
                z_shape = model_dict["G"].z_shape

        if "G_ema" in model_cfg and "G" in model_dict:
            model_dict["G_ema"].load_state_dict(model_dict["G"].state_dict())
            model_dict["G_ema"] = model_dict["G_ema"].eval()
        
        # render segment for training
        print("render_seg", model_dict["G"].gs_renderer.render_seg    )
        print("render_seg", model_dict["G_ema"].gs_renderer.render_seg)

        model_dict["G"].gs_renderer.render_seg     = True
        model_dict["G_ema"].gs_renderer.render_seg = True

    device_control.sync_ddp()

    with nullcontext("load checkpoint"):
        state_dict = None
        resume_stat = {}
        if args.checkpoint is not None:
            if os.path.splitext(args.checkpoint)[-1] == ".zip":
                import zipfile
                with zipfile.ZipFile(args.checkpoint, "r").open("param.pth") as f:
                    state_dict = torch.load(f, map_location="cpu")
            elif os.path.splitext(args.checkpoint)[-1] == ".pth":
                state_dict = torch.load(args.checkpoint, map_location="cpu")
            else:
                raise Exception(f"unknown checkpoint type '{args.checkpoint}'")

        if state_dict is not None:
            with torch.no_grad():
                succ = True
                for key, mod in [
                    ("G",     model_dict["G"]    ), 
                    ("G_ema", model_dict["G_ema"]), 
                    ("D",     model_dict["D"]    ), 
                    ]:
                    if key in state_dict:
                        src_state   = state_dict[key]
                        mod_state   = mod.state_dict()
                        mismatch    = { k for k, v in mod_state.items()                             if k in src_state and v.shape != src_state[k].shape }
                        model_state = OrderedDict([ [k, src_state[k]] for k, v in mod_state.items() if k in src_state and v.shape == src_state[k].shape ])

                        for k, tgt_v in mod_state.items():
                            if k in src_state and tgt_v.shape != src_state[k].shape:
                                src_v = src_state[k]
                                # only 1s are missing
                                if src_v.squeeze().shape == tgt_v.squeeze().shape:
                                    print(k, src_v.shape, tgt_v.shape)
                                    model_state[k] = src_v.clone().reshape(tgt_v.shape).detach()

                                    mismatch.discard(k)

                        missing, wrong = mod.load_state_dict(model_state, strict=False)
                        # print(f"load [{key}] missig:{missing} wrong:{wrong}")
                        print(f"load [{key}] missig:{missing} wrong:{wrong} mismatch:{mismatch}")

                        torch.cuda.synchronize()

                        mod_a, mod_b = mod.state_dict(), model_state

                        for k in mod_a:
                            if k not in mod_b:
                                continue
                        
                            p_a, p_b = mod_a[k], mod_b[k]

                            if p_a.shape != p_b.shape:
                                continue

                            close = torch.allclose(p_a.to(p_b.device), p_b)

                            if close:
                                pass
                                # print(f"[{key}] {k} succ")
                            else:
                                dist = (p_a.to(p_b.device) - p_b).abs().mean().item()
                                info = lambda t: (type(t), t.device, t.shape, t.dtype, getattr(t, "memory_format", t.layout), t.grad_fn)
                                device_control.local_print(f"[{key}] {k} fail, {dist}, model: {info(p_a)}, state: {info(p_b)}")

                                succ = False

                        if key in ["G", "G_ema"] and train_cfg.get("reset_outbox_fs", False) is True:
                            fs_xyz = mod.fs_xyz

                            fs_xyz = fs_xyz + (mod.fs_xyz_init - fs_xyz).mean(dim=1, keepdim=True)

                            # box_warp = mod.vr_kwargs["box_warp"]
                            box_warp = G.gs_renderer.field_kwargs.get('box_warp', 1)
                            
                            box_pos = torch.as_tensor([ 0.6*box_warp/2,  box_warp/2, 0.6*box_warp/2], device=fs_xyz.device).reshape(1, 1, 3)
                            box_neg = torch.as_tensor([-0.6*box_warp/2, -box_warp/2,    -box_warp/2], device=fs_xyz.device).reshape(1, 1, 3)

                            le = fs_xyz  < box_pos
                            gt = box_neg < fs_xyz

                            inside_bbox = (le.sum(dim=-1) + gt.sum(dim=-1)) == 6
                            inside_bbox = inside_bbox.unsqueeze(-1).expand(-1, -1, 3)

                            fs_xyz = torch.where(inside_bbox, fs_xyz, mod.fs_xyz_init)
                            mod.fs_xyz.copy_(fs_xyz)

                        if key in ["G", "G_ema"] and train_cfg.get("reset_sdf_fs", False) is True:
                            tri     = mod.flame.faces_tensor
                            aux_tri = np.loadtxt(os.path.join(ROOT, "Data", "FLAME2020", "additional_tri.txt"))
                            aux_tri = torch.as_tensor(aux_tri, dtype=tri.dtype, device=tri.device)

                            tri = torch.cat([tri, aux_tri], dim=0)
                            ver = mod.flame.v_template + mod.flame_offset
                            faces = ver[tri.flatten()].reshape(1, -1, 3, 3) # B,Nf,3,3

                            dis     = Mesh2SignedDistanceFunction.apply(faces.to(device), mod.fs_xyz_init.to(device))[0].to(mod.fs_xyz_init.device)
                            reset   = (dis.abs() < 0.001).unsqueeze(-1).expand(-1, -1, 3).to(mod.fs_xyz_init.device)
                            fs_xyz_init = torch.where(reset, 
                                                        mod.fs_xyz_init + (dis[...,None].abs())*F.normalize(mod.fs_xyz_init, dim=-1), mod.fs_xyz_init)

                            mod.fs_xyz_init.copy_(fs_xyz_init)

                            dis   = Mesh2SignedDistanceFunction.apply(faces.to(device), mod.fs_xyz.to(device))[0].to(faces.device)

                            reset = (dis.abs()>0.05).unsqueeze(-1).expand(-1, -1, 3).to(mod.fs_xyz.device)

                            fs_xyz = torch.where(reset, mod.fs_xyz_init, mod.fs_xyz)
                            mod.fs_xyz.copy_(fs_xyz)

                        if key in ["G", "G_ema"] and train_cfg.get("reset_fs_xyz_mlp", False) is True:
                            for n, p in mod.fs_xyz_mlp.named_parameters():
                                # p.copy_(p.lerp(torch.zeros_like(p), 1))
                                # p.copy_(p.lerp(torch.zeros_like(p), 1))
                                p.normal_(mean=0, std=0.00001)

                        if key in ["G"] and "fs_xyz" in mismatch:
                            print("recompute fs_xyz")

                            fs_xyz, fs_prm = state_dict[key]["fs_xyz"].squeeze(0), state_dict[key]["fs_prm"].squeeze(0)
                            
                            np_fs_xyz = fs_xyz.detach().cpu().numpy()

                            from sklearn.neighbors import KernelDensity

                            kde = KernelDensity(kernel="gaussian", bandwidth=0.03).fit(np_fs_xyz)
                            new_xyz = kde.sample(mod_state["fs_xyz"].size(1))

                            # from scipy.spatial   import KDTree
                            # kd_tree = KDTree(np_fs_xyz)
                            # indexes = kd_tree.query_ball_tree(new_xyz, r=0.2)
                            # indx    = torch.as_tensor(indexes, dtype=torch.long, device=fs_prm.device)
                            # new_prm = fs_prm[indx]

                            new_xyz = torch.as_tensor(new_xyz, dtype=torch.float32, device=fs_xyz.device)
                            def get_prm_by_nn(query_xyz):
                                dist    = torch.cdist(query_xyz, fs_xyz) # N_new, N_old
                                indx    = torch.argmin(dist, dim=1)      # N_new
                                new_prm = fs_prm[indx]
                                return new_prm
                            new_prm = chunk_fn(get_prm_by_nn, 1024, [new_xyz])

                            new_xyz, new_prm = new_xyz.unsqueeze(0), new_prm.unsqueeze(0)

                            assert new_xyz.shape == mod.fs_xyz.shape, f"{new_xyz.shape} != {mod.fs_xyz.shape}"
                            assert new_prm.shape == mod.fs_prm.shape, f"{new_prm.shape} != {mod.fs_prm.shape}"

                            mod.fs_xyz.copy_(new_xyz)
                            mod.fs_prm.copy_(new_prm)

                            if "G_ema" in model_dict:
                                model_dict["G_ema"].fs_xyz.copy_(new_xyz)
                                model_dict["G_ema"].fs_prm.copy_(new_prm)

                        if key in ["G"] and "vr_decoder.net.0.weight" in mismatch:
                            print("reset vr_decoder")

                            vr_w = state_dict[key]["vr_decoder.net.0.weight"]

                            weight = model_dict["G"].vr_decoder.net[0].weight

                            if vr_w.shape != weight.shape:
                                vr_w = torch.cat([vr_w, 0.001*torch.randn(weight.size(0), weight.size(1)-vr_w.size(1))], dim=1)

                            weight.copy_(vr_w)
                            model_dict["G_ema"].vr_decoder.net[0].weight.copy_(vr_w)
                    else:
                        print(f"[{key}] is not found in {state_dict.keys()}")
        
                if succ is False:
                    device_control.local_print("some value mismatch between checkpoint")

                if args.resume and "stat" in state_dict:
                    resume_stat.update(state_dict["stat"])

            state_dict.clear()

        torch.cuda.empty_cache()
        device_control.sync_ddp()

    for k, v in model_dict.items():
        if k in ["G", "D"]:
            if writer:
                writer.add_text(f"NetArch/{k}", f"## {k} arch\n\n<pre><code>\n{v}</pre></code>\n", 0)

    for k, v in model_dict.items():
        if k.find("ema") >= 0:
            continue
        print(f"{k} {v}")

    # broadcast after loading checkpoint
    if device_control.ddp_enabled is True:
        for k, v in model_dict.items():
            v.requires_grad_(True)
    multi_device_ctx, main_process_dcr, model_dict = device_control.send_model_to_device(model_dict, device, int(10*1e3))
    # multi_device_ctx, main_process_dcr, model_dict = device_control.send_model_to_device(model_dict, device, 100)
    for k, v in model_dict.items():
        v.requires_grad_(False)

    device_control.sync_ddp()
    
    # optim_dict = {}
    with nullcontext("train setup"):
        G  = model_dict.get("G", {})
        DM = model_dict.get("DM", {})

        NAME       = config["name"]
        if hasattr(G, "z_dim"):
            NUM_LATENT   = G.z_dim
            SHAPE_LATENT = [G.z_dim]
        elif hasattr(G, "z_shape"):
            SHAPE_LATENT = list(G.z_shape)
        
        TOTAL_KIMG = train_cfg.get("total_kimg", 1000)
        NUM_JOBERS = device_control.get_world_size()
        EPOCHS     = (TOTAL_KIMG * 1000 + len(train_dataset) - 1) // len(train_dataset)

        GLOBAL_BATCH_SIZE = train_cfg.get("batch_size")
        LOCAL_BATCH_SIZE  = GLOBAL_BATCH_SIZE // NUM_JOBERS
        VALID_BATCH_SIZE  = min(4, LOCAL_BATCH_SIZE)

        EMA_KIMG   = train_cfg.get("ema_kimg",   GLOBAL_BATCH_SIZE*10/32)
        EMA_RAMPUP = train_cfg.get("ema_rampup", 0.05)

        LOG_I  = train_cfg.get("log_interval")
        SAVE_I = train_cfg.get("save_interval")
        EVAL_I = train_cfg.get("eval_kimg", 100)*1000
        ADA_I  = train_cfg.get("ada_kimg",  100)*1000
        RES    = G.img_resolution if "G" in model_dict else DM.img_resolution

        # old value, set by hand
        # EMA_KIMG   = 1.25
        # EMA_RAMPUP = 0.05

        targets   = train_cfg.get("targets", ["GAN"])
        TRAIN_GAN = "GAN" in targets 

        RAND_CONDITION = train_cfg.get("rand_condition",  True)
        TV_WINDOW_WIDTH= train_cfg.get("tv_window_width", 1)

        GRAD_CLAMP = train_cfg.get("grad_clamp", None)
        ZERO_GRAD_NONE = train_cfg.get("zero_grad_none", False)

        USING_AMP  = args.enable_amp
        ACCUMULATE_ITER = train_cfg.get("accumulate_iter", 1)
        UPDATE_ALIVE    = train_cfg.get("update_alive", False)

        global_amp_scaler = torch.cuda.amp.GradScaler(init_scale=1, enabled=USING_AMP)

        num_workers = train_cfg.get("num_workers", 0)
        persistent_workers = True if num_workers > 0 else False
        # if args.enable_ddp:
        if device_control.ddp_enabled:
            DistributedSampler = lambda ds : torch.utils.data.DistributedSampler(ds, seed=SEED) # global seed, if not set default(0) is used
            train_loader  = torch.utils.data.DataLoader(train_dataset, batch_size=LOCAL_BATCH_SIZE, 
                                                        shuffle=False, drop_last=True, sampler=DistributedSampler(train_dataset),
                                                        num_workers=num_workers, pin_memory=True, persistent_workers=persistent_workers)
            valid_loader  = torch.utils.data.DataLoader(train_dataset, batch_size=VALID_BATCH_SIZE, 
                                                        num_workers=0, shuffle=False, drop_last=True,)
            eval_loader   = torch.utils.data.DataLoader(eval_dataset,  batch_size=LOCAL_BATCH_SIZE, 
                                                        shuffle=False, drop_last=False, sampler=DistributedSampler(eval_dataset),
                                                        num_workers=num_workers, pin_memory=True)
            eval_loader.sampler.set_epoch(0)
        else:
            train_loader  = torch.utils.data.DataLoader(train_dataset, batch_size=LOCAL_BATCH_SIZE, 
                                                        shuffle=True, drop_last=True,
                                                        num_workers=num_workers, pin_memory=True, persistent_workers=persistent_workers)
            valid_loader  = torch.utils.data.DataLoader(train_dataset, batch_size=VALID_BATCH_SIZE, 
                                                        num_workers=0, shuffle=False, drop_last=True,)
            eval_loader   = torch.utils.data.DataLoader(eval_dataset,  batch_size=LOCAL_BATCH_SIZE, 
                                                        shuffle=False, drop_last=False,
                                                        num_workers=num_workers, pin_memory=True)

        STEPS  = EPOCHS*len(train_loader)

        all_params = list(model_dict["G"].parameters()) + list(model_dict["D"].parameters())
        rank_ps = device_control.deepspeed_param_setting(all_params, zero=train_cfg.get("deepspeed", "1"))

        all_nprm = np.sum([p.numel() for p in all_params])
        dps_nprm = 0
        for i in range(device_control.get_world_size()):
            params = np.sum([p.numel() for p in rank_ps[i]])
            print(f"Rank[{i}] deepspeed param:{params/2**20:.2f}MB")
            dps_nprm += params

        print(f"all_params:{all_nprm/2**20:.2f}MB || ds_params:{dps_nprm/2**20:.2f}MB")

        rank = device_control.get_rank()
        
        phases = []
        if TRAIN_GAN:
            G = model_dict["G"]
            D = model_dict["D"]
            G_opt_config = optim_cfg["G"]
            D_opt_config = optim_cfg["D"]

            Greg_interval = train_cfg.get("Greg_interval", 4)
            Dreg_interval = train_cfg.get("Dreg_interval", 16)

            import re
            # G_n_params   = G.named_parameters()
            G_p_groups   = G_opt_config.get("groups", [{"regex": ".*", "name": "default"}]) # default: add all
            G_params     = []
            covered      = set()
            for config in G_p_groups:
                regex = config.pop("regex")
                group_dict = copy.deepcopy(config)
                params     = []
                for k, v in G.named_parameters():
                    if k not in covered \
                        and re.match(regex, k) is not None \
                        and v.data_ptr() in [p.data_ptr() for p in rank_ps[rank]]:
                        params.append(v)
                        covered.add(k)
                if len(params) == 0:
                    print(f"could not match with regex {regex}")
                    continue
                    params.append(nn.Parameter(torch.zeros(4)))
                group_dict["params"] = params

                lr = config.get("lr", None)
                if lr is None or lr > 0:
                    G_params.append(group_dict)

            D_p_groups   = D_opt_config.get("groups", [{"regex": ".*", "name": "default"}]) # default: add all
            D_params     = []
            covered      = set()
            for config in D_p_groups:
                regex = config.pop("regex")
                group_dict = copy.deepcopy(config)
                params     = []
                for k, v in D.named_parameters():
                    if k not in covered \
                        and re.match(regex, k) is not None \
                        and v.data_ptr() in [p.data_ptr() for p in rank_ps[rank]]:
                        params.append(v)
                        covered.add(k)
                if len(params) == 0:
                    params.append(nn.Parameter(torch.zeros(4)))
                group_dict["params"] = params

                lr = config.get("lr", None)
                if lr is None or lr > 0:
                    D_params.append(group_dict)

            for name, params, opt_config, reg_interval in [('G', G_params, G_opt_config, Greg_interval), ('D', D_params, D_opt_config, Dreg_interval)]:
                OP_type = opt_config.get("type", "torch.optim.Adam")
                op_args = opt_config.get("args", list())
                op_kwargs= opt_config.get("kwargs", dict())    # default optimizer hparam
                # params   = filter(lambda p:p.numel()>0, params)

                all_params = list(model_dict[name].parameters())

                if reg_interval is None:
                    opt = get_class(OP_type)(params, *op_args, **op_kwargs)
                    amp = torch.cuda.amp.GradScaler(init_scale=128, enabled=USING_AMP)
                    phases += [EasyDict(name=name+'both', amp=amp, opt=opt, interval=1)]
                else: # Lazy regularization.
                    mb_ratio = reg_interval / (reg_interval + 1)
                    op_kwargs = EasyDict(op_kwargs)
                    op_kwargs.lr = op_kwargs.lr * mb_ratio
                    op_kwargs.betas = [beta ** mb_ratio for beta in op_kwargs.betas]
                    for pg in params:
                        if "lr" in pg:
                            pg["lr"] = pg["lr"] * mb_ratio
                        if "betas" in pg:
                            pg["betas"] = [beta ** mb_ratio for beta in pg["betas"]]
                    opt    = get_class(OP_type)(params, *op_args, **op_kwargs)
                    amp    = torch.cuda.amp.GradScaler(init_scale=128, enabled=USING_AMP)
                    phases += [EasyDict(name=name+'main', amp=amp, opt=opt, all_params=all_params, interval=1)]
                    phases += [EasyDict(name=name+'reg',  amp=amp, opt=opt, all_params=all_params, interval=reg_interval)]
            
        phases = [ p for p in phases if p.name not in train_cfg.get("skip_phase", list()) ]

        train_steps = len(train_loader)*EPOCHS

        param_info = []

        optim_params = set()
        for phase in phases:
            phase.start_event = None
            phase.end_event   = None
            phase.start_event = torch.cuda.Event(enable_timing=True)
            phase.end_event   = torch.cuda.Event(enable_timing=True)
            # phase.time_measure = []

            phase.last_log_step = 0

            config = phase_cfg.get(phase.name, {"sched": None})

            sched = config["sched"]
            if sched == "warmup":
                def warmup(step):
                    cur_nimg = (step * phase.interval) * GLOBAL_BATCH_SIZE
                    mul_num  = min(1, cur_nimg / (0.1*TOTAL_KIMG*1e3))  # 2.5 M image
                    return max(0.5, mul_num)
                phase.lr_scheduler = torch.optim.lr_scheduler.LambdaLR(phase.opt, warmup)
            elif sched == "step":
                phase.lr_scheduler = torch.optim.lr_scheduler.StepLR(phase.opt, len(train_loader)//phase.interval, 0.99)
            else:
                phase.lr_scheduler = None

            optim_total_num = 0
            optim_train_num = 0
            lr, wd = 0, 0

            opt_type = type(phase.opt).__name__
            for pgn, pg in enumerate(phase.opt.param_groups):
                group_train_num, group_total_num = 0, 0
                for param in pg["params"]:
                    optim_params.add(param)
                    if param.requires_grad:
                        group_train_num += param.numel()
                    group_total_num += param.numel()

                lr, wd = pg["lr"], pg["weight_decay"]
                betas, eps = pg["betas"], pg["eps"]

                if "name" in pg:
                    pgn = pg["name"]

                param_info.append({
                    "name": f"phase:{phase.name}({opt_type}[{pgn}])", 
                    "train_param": group_train_num, "total_param": group_total_num, 
                    "other": f"lr={lr} wd={wd} betas={betas} eps={eps}"
                    })

                optim_train_num += group_train_num
                optim_total_num += group_total_num

            if optim_total_num == 0:
                print(f"[phase:{phase.name}] Params optim train/total: {optim_train_num/2**20:.2f}/{optim_total_num/2**20:.2f}%")
            else:
                print(f"[phase:{phase.name}] Params optim train/total: {optim_train_num/2**20:.2f}/{optim_total_num/2**20:.2f}, {optim_train_num/optim_total_num*100:.2f}%")
            for pg in phase.opt.param_groups:
                name = pg.get("name", None)
                group_total_num = 0
                for param in pg["params"]:
                    optim_params.add(param)
                    lr, wd = pg["lr"], pg["weight_decay"]
                    group_total_num += param.numel()
                    betas, eps = pg["betas"], pg["eps"]
                print(f"    pg:{name} params: {group_total_num} lr:{lr}, wd:{wd}, betas:{betas}, eps:{eps}, scheduler: {phase.get('lr_scheduler', None)}")
        
        # os._exit(1)
        for k, v in model_dict.items():
            model_train_num = sum(p.numel() for p in list(v.parameters()) if p.requires_grad)
            model_total_num = sum(p.numel() for p in list(v.parameters()) )
            model_optim_num = sum(p.numel() for p in list(v.parameters()) if p in optim_params)

            model_buffer_num = sum(p.numel() for p in list(v.buffers()) )

            not_optimized   = [n for n, p in v.named_parameters() if p not in optim_params]
            if "ema" not in k and len(not_optimized) > 0:
                print(f"[model:{k}] {not_optimized}")

            print(f"[model:{k}] Params model train/total: {model_train_num}/{model_total_num}, {model_train_num/max(model_total_num,1)*100:.2f}% | optim/total: {model_optim_num}/{model_total_num}, {model_optim_num/max(model_total_num,1)*100:.2f}%")

            if model_total_num < 1024*1024:
                print(f"too few paramters for {k}, please double check")
                print([pn for pn, _ in v.named_parameters()])

            mod_type = type(v).__name__
            param_info.append({
                "name": f"{k}({mod_type})", 
                "train_param": model_train_num, "total_param": model_total_num, 
                "other": f"buffer={model_buffer_num}"
                })

        pit_key   = ["name", "train_param", "total_param", "other"]
        pit_head  = "|".join([k for k in pit_key])
        pit_sepr  = "|".join([ "---" for k in pit_key])

        pit_data = []
        for pi in param_info:
            pit_data.append("|".join([ str(pi.get(k, None)) for k in pit_key]))
        pit_data  = "\n".join(pit_data)
        pi_str    = f"{pit_head}\n{pit_sepr}\n{pit_data}"

        if writer:
            print(pi_str)
            writer.add_text(f"ParamInfo", pi_str, 0)

        print(f"using fp16: {USING_AMP}")

        # training_stats = 
        from models.torch_utils import training_stats
        import models.Augment
        import models.ADAAugment

        stats_collector = training_stats.Collector(regex='.*')
        ada_stats  = training_stats.Collector(regex='Loss/signs/real')
        ada_target = train_cfg.get("ada_target", 0.8)
        aug_diff   = models.Augment.Diffusion(ts_dist="uniform")
        aug_pipe   = models.ADAAugment.AugmentPipe(
            xflip=0, rotate90=0, xint=0, scale=0, rotate=0, aniso=0, xfrac=0, 
            brightness=0.1, contrast=0.1, lumaflip=0, hue=0.1, saturation=0.1, 
            imgfilter=1, noise=1, cutout=1
        ).to(device)
        aug_pipe.p.copy_(torch.full([], 0.0))

        if device_control.ddp_enabled:
            for tensor in list(aug_pipe.parameters()) + list(aug_pipe.buffers()):
                device_control.broadcast_to_all(tensor)
        
        NORMAL_CLAMP = train_cfg.get("normal_clamp", 1.5)

        attr_dict = {
            "r1_gamma":        2,
            "pl_mean":         torch.zeros([], device=device),
            "pl_batch_shrink": 2,
            "pl_decay":        0.01,
            "pl_weight":       2,
            "blur_init_sigma": 10,
            "blur_fade_kimg":  1000,
            "tex_symmetric":     0.1,
            "tex_smooth":        0.5,
            "pos_smooth":        0,
            "img_smooth":        0,
            "depth_smooth":      0,
            "normal_smooth":     0,
            "bump_smooth":       0.5,
            "back_smooth":       0.2,

            "init_prob":                0,
            "adaptive_prob":            1.0,
            "mask_dropout_max":         0.8,
            "m2v_reg_fade_kimg":        1000,
            "m2v_reg_prob":             0,
            "rot_reg_fade_kimg":        1000,
            "rot_reg_prob":             0,
            "rot_hard_reg_fade_kimg":   1000,
            "rot_hard_reg_prob":        0,
            "t3d_reg_fade_kimg":        1000,
            "t3d_reg_prob":             0,
            "t3d_scale_reg_fade_kimg":  1000,
            "t3d_scale_reg_prob":       0,
            "light_reg_fade_kimg":      1000,
            "light_reg_prob":           0,
            "lightrot_reg_fade_kimg":   1000,
            "lightrot_reg_prob":        0,

            "pose_reg_fade_kimg":       1000,
            "pose_reg_prob":            0,
            "id_reg_fade_kimg":         1000,
            "id_reg_prob":              0,
            "idswap_reg_fade_kimg":     1000,
            "idswap_reg_prob":          0,
            "exp_reg_fade_kimg":        1000,
            "exp_reg_prob":             0,

            "reg_bump":          0,
            "reg_scale":         0,
            "reg_sorted":        0,
            "reg_opacity":       0,
            "reg_opacity_beta":  0,
            "reg_opacity_l1":    0,
            "reg_opacity_l2":    0,
            "reg_normal":        0,
            "reg_triplane":      0,
            "reg_surface_pull":  0,
            "reg_inter_push":    0,
            "reg_fs_xyz_init":   0,
            "reg_fs_xyz_init_l2":0,
            "reg_fs_xyz_norm":   0,
            "reg_fs_xyz_anchor": 0,
            "reg_fs_xyz_offset": 0,
            "Greg_method":     "texture_symmetric",
            "styleganxl_loss":  False,
            "augment":         [],
            "last_ada_step":   0,
            }
        for k, v in loss_cfg.get("attr", {}).items():
            attr_dict[k] = v
        attr_dict = EasyDict(attr_dict)

        sdf_tri = torch.as_tensor(np.loadtxt(os.path.join(ROOT, "Data", "FLAME2020", "sdf_tri.txt")), dtype=torch.long, device=device)
        attr_dict.water_tight_tri = sdf_tri

        # def total_variance_loss(image, n_sample=1000, window_width=TV_WINDOW_WIDTH):
        #     '''
        #     image: (B, H, W, C)
        #     '''
        #     initial_coordinates = 2*torch.rand((image.shape[0], n_sample, 2), device=device) - 1
        #     perturbed_coordinates = initial_coordinates + torch.randn_like(initial_coordinates)*(window_width/image.size(1))

        #     all_coordinates = torch.cat([initial_coordinates, perturbed_coordinates], dim=1)

        #     value = F.grid_sample(image.permute(0, 3, 1, 2), all_coordinates.unflatten(1, (-1, 1)), mode="bilinear", align_corners=False)
        #     value = value.flatten(-2).transpose(-2, -1)

        #     value_initial   = value[:, :value.shape[1]//2]
        #     value_perturbed = value[:, value.shape[1]//2:]

        #     l_tv = F.l1_loss(value_initial, value_perturbed)
        #     return l_tv
        
        def save_D_image(img, fname):
            return
            img = Ni.invert(img)
            BS, H, W = img.shape[:3]
            msk, rgb = img.split([img.size(-1)-3, 3], dim=-1) # B,h,w,c

            img = torch.cat([msk, rgb], dim=-2).reshape(BS*H, 2*W, 3)
            img = (255*img.detach().cpu().numpy()).astype(np.uint8)

            img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
            cv2.imwrite(os.path.join(SAVE_ROOT, f"{fname}"), img)

        # Ni,No= model_dict["norm_i"], model_dict["norm_o"]
        def compute_loss(phase, data, gain, step):
            image   = data["image"].to(device)
            loss_d  = {}
            out_d   = {}

            if phase.name in ["Gmain", "Greg", "Dmain", "Dreg"]:

                G, D = model_dict["G"].train(), model_dict["D"].train()
                # Ni,No= model_dict["norm_i"], model_dict["norm_o"]

                cur_nimg = step * GLOBAL_BATCH_SIZE
                blur_sigma = max(1 - cur_nimg / (attr_dict.blur_fade_kimg * 1e3), 0) * attr_dict.blur_init_sigma if attr_dict.blur_fade_kimg > 0 else 0
                blur_size = np.floor(blur_sigma * 3)

                if blur_size > 0:
                    f = torch.arange(-blur_size, blur_size + 1, device=device).div(blur_sigma).square().neg().exp2()
                    f = f / f.sum()
                    from models.torch_utils.ops import upfirdn2d
                    def blur_fn(image):
                        with record_function('blur'):
                            return upfirdn2d.filter2d(image.permute(0, 3, 1, 2), f).permute(0, 2, 3, 1)
                else:
                    blur_fn = lambda x:x

                if "diffusion" in attr_dict.augment:
                    def run_D(img, msk, cond):
                        img = blur_fn(img)
                        # img, t = aug_diff(img)
                        # msk, rgb = img.split([img.size(-1)-3, 3], dim=-1)
                        img, t   = aug_diff(img)
                        # img = torch.cat([msk, rgb], dim=-1)
                        # cond_time = torch.cat([cond, t.reshape(-1, 1)], dim=-1)
                        # return D(img, cond_time)
                        return D(img, msk, cond)
                elif "ada" in attr_dict.augment:
                    def run_D(img, msk, cond):
                        save_D_image(img, f"{step}_{phase.name}_before_aug.png")
                        img = blur_fn(img)
                        # only change rgb, leave mask unchanged
                        # msk, rgb = img.split([img.size(-1)-3, 3], dim=-1)
                        img = aug_pipe(img.permute(0, 3, 1, 2)).permute(0, 2, 3, 1)
                        # img = torch.cat([msk, rgb], dim=-1)
                        save_D_image(img, f"{step}_{phase.name}_after_aug.png")
                        return D(img, msk, cond)
                else:
                    def run_D(img, msk, cond):
                        img = blur_fn(img)
                        # if img.size(-1) > 6:
                        #     img = img[..., :6]
                        return D(img, msk, cond)

                noise     = NOISE_SCALE*torch.randn([image.size(0)]+SHAPE_LATENT, device=device)

                with torch.no_grad():
                    data["rot"] = data["m2v"][:, :3, :3]

                    if "light" in data:
                        light         = data["light"].reshape(-1, 9, 3).clone()
                        sh0, sh1, sh2 = torch.split(light, [1, 3, 5], dim=1)
                        data["sh0"] = sh0
                        data["sh1"] = sh1
                        data["sh2"] = sh2

                    condition   = G.pack_condition(data)
                    condition_d = G.disc_G.pack_condition(data)
                    if condition is not None:
                        # rcondition = condition[torch.randperm(image.size(0)).to(device)]

                        init_prob = attr_dict.init_prob

                        def get_swapped(final_prob, fade_kimg):
                            final_prob = final_prob * attr_dict.adaptive_prob
                            alpha = min(cur_nimg / (fade_kimg*1e3), 1) if fade_kimg > 0 else 1
                            prob  = (1-alpha) * init_prob + alpha * final_prob
                            swapped = torch.rand((image.size(0), 1), device=device) < prob
                            return swapped

                        # training_set.get_label(np.random.randint(len(training_set)))

                        if RAND_CONDITION:
                            index    = torch.randint(0, len(all_cond_dict["shape"]), (len(image),), device=device)
                            gen_cond = {k: v[index].clone() for k, v in all_cond_dict.items()} # clone to prevent inplace modify
                        else:
                            gen_cond = {k:v for k, v in data.items() if k not in ["image", "segim"]} # it is not deep copy !!!

                        if "m2v" in gen_cond:
                            if attr_dict.m2v_reg_prob is not None and attr_dict.m2v_reg_prob > 0:
                                swapped = get_swapped(attr_dict.m2v_reg_prob, attr_dict.m2v_reg_fade_kimg)
                                origin  = gen_cond["m2v"]
                                # rolled = torch.roll(origin, 1, 0)

                                sy  = torch.linalg.norm(origin[:, :2, 0], dim=-1)
                                yaw = torch.arctan2(origin[:, 0, 2], sy)

                                index  = torch.multinomial(yaw.abs(), origin.size(0), replacement=True)
                                rolled = origin[index]
                                gen_cond["m2v"] = torch.where(swapped.unsqueeze(-1), rolled, origin)

                            if attr_dict.rot_reg_prob is not None and attr_dict.rot_reg_prob > 0:
                                swapped = get_swapped(attr_dict.rot_reg_prob, attr_dict.rot_reg_fade_kimg)
                                origin  = gen_cond["m2v"]

                                sy  = torch.linalg.norm(origin[:, :2, 0], dim=-1)
                                yaw = torch.arctan2(origin[:, 0, 2], sy)

                                index  = torch.multinomial(yaw.abs(), origin.size(0), replacement=True)
                                rolled = origin[index].clone()
                                rolled[:, :3, 3:] = origin[:, :3, 3:]   # copy original t3d

                                gen_cond["m2v"] = torch.where(swapped.unsqueeze(-1), rolled, origin)

                            if attr_dict.rot_hard_reg_prob is not None and attr_dict.rot_hard_reg_prob > 0:
                                swapped = get_swapped(attr_dict.rot_hard_reg_prob, attr_dict.rot_hard_reg_fade_kimg)
                                origin = gen_cond["m2v"]

                                m2v = []
                                S   = origin.size(0)
                                for sign, deg in zip(np.random.rand(S), np.random.rand(S)*60+30):
                                    r   = np.deg2rad(deg) if sign < 0.5 else -np.deg2rad(deg)
                                    rot = np.array([
                                        [ np.cos(r), 0, np.sin(r),    0],
                                        [         0, 1,         0,    0],
                                        [-np.sin(r), 0, np.cos(r),    0],
                                        [         0, 0,         0,    1],
                                    ])
                                    m2v.append(torch.as_tensor(rot, dtype=torch.float32))

                                # print([getattr(t, "shape", type(t)) for t in m2v])
                                
                                rolled = torch.stack(m2v, dim=0).to(device)

                                rolled[:, :3, 3:] = origin[:, :3, 3:]   # copy original t3d
                                gen_cond["m2v"] = torch.where(swapped.unsqueeze(-1), rolled, origin)

                            if attr_dict.t3d_reg_prob is not None and attr_dict.t3d_reg_prob > 0:
                                swapped = get_swapped(attr_dict.t3d_reg_prob, attr_dict.t3d_reg_fade_kimg)
                                origin = gen_cond["m2v"]
                                dist   = torch.linalg.norm(origin[:, :3, 3].flatten(1), dim=-1)
                                index  = torch.multinomial(dist, origin.size(0), replacement=True)
                                rolled = origin.clone()
                                rolled[:, :3, 3:] = origin[index][:, :3, 3:]   # copy selected t3d

                                gen_cond["m2v"] = torch.where(swapped.unsqueeze(-1), rolled, origin)

                            if attr_dict.t3d_scale_reg_prob is not None and attr_dict.t3d_scale_reg_prob > 0:
                                swapped = get_swapped(attr_dict.t3d_scale_reg_prob, attr_dict.t3d_scale_reg_fade_kimg)
                                origin = gen_cond["m2v"]
                                s      = 1.0 + 0.1*(2*torch.rand(origin.size(0), device=device)-1)   # [0.9 - 1.1]
                                rolled = origin.clone()
                                rolled[:, :3, 3] = rolled[:, :3, 3]*s.reshape(-1, 1)    # copy modified t3d

                                gen_cond["m2v"] = torch.where(swapped.unsqueeze(-1), rolled, origin)

                        if "light" in gen_cond:
                            if attr_dict.light_reg_prob is not None and attr_dict.light_reg_prob > 0:
                                swapped = get_swapped(attr_dict.light_reg_prob, attr_dict.light_reg_fade_kimg)
                                origin = gen_cond["light"]
                                rolled = torch.roll(origin, 1, 0)
                                gen_cond["light"] = torch.where(swapped.unsqueeze(-1), rolled, origin)

                            if attr_dict.lightrot_reg_prob is not None and attr_dict.lightrot_reg_prob > 0:
                                swapped = get_swapped(attr_dict.lightrot_reg_prob, attr_dict.lightrot_reg_fade_kimg)
                                origin = gen_cond["light"]

                                rad = (2*torch.rand(origin.size(0))-1)*torch.pi/2  # [-pi/4, pi/4]

                                cos = torch.cos(rad)
                                sin = torch.sin(rad)

                                b33 = torch.eye(3)[None].repeat(rad.size(0), 1, 1) # B, 3, 3
                                b33[:, 0, 0] = cos
                                b33[:, 0, 1] = sin
                                b33[:, 1, 0] = -sin
                                b33[:, 1, 1] = cos

                                # rot = np.array([
                                #     [ np.cos(r), np.sin(r), 0],
                                #     [-np.sin(r), np.cos(r), 0],
                                #     [         0,         0, 1],
                                # ])

                                rolled = rotate_sh(origin, b33)
                                gen_cond["light"] = torch.where(swapped.unsqueeze(-1), rolled, origin)

                        if "shape" in gen_cond:
                            if attr_dict.id_reg_prob is not None and attr_dict.id_reg_prob > 0:
                                swapped = get_swapped(attr_dict.id_reg_prob, attr_dict.id_reg_fade_kimg)
                                origin = gen_cond["shape"]
                                # rolled = torch.randn_like(origin).clamp(-1.5, 1.5)
                                rolled = torch.randn_like(origin).clamp(-NORMAL_CLAMP, NORMAL_CLAMP)
                                gen_cond["shape"] = torch.where(swapped, rolled, origin)

                            if attr_dict.idswap_reg_prob is not None and attr_dict.idswap_reg_prob > 0:
                                swapped = get_swapped(attr_dict.id_reg_prob, attr_dict.idswap_reg_fade_kimg)
                                origin = gen_cond["shape"]
                                rolled = torch.roll(origin, 1, 0)
                                gen_cond["shape"] = torch.where(swapped, rolled, origin)

                        if "exp" in gen_cond:
                            if attr_dict.exp_reg_prob is not None and attr_dict.exp_reg_prob > 0:
                                swapped = get_swapped(attr_dict.exp_reg_prob, attr_dict.exp_reg_fade_kimg)
                                origin = gen_cond["exp"]
                                # rolled = torch.randn_like(origin)
                                rolled = torch.roll(origin, 1, 0)
                                gen_cond["exp"] = torch.where(swapped, rolled, origin)

                        if "pose" in gen_cond:
                            if attr_dict.pose_reg_prob is not None and attr_dict.pose_reg_prob > 0:
                                swapped = get_swapped(attr_dict.pose_reg_prob, attr_dict.pose_reg_fade_kimg)
                                origin = gen_cond["pose"]
                                norm   = torch.linalg.norm(origin, dim=-1)
                                index  = torch.multinomial(norm, origin.size(0), replacement=True)
                                rolled = origin[index]
                                gen_cond["pose"] = torch.where(swapped, rolled, origin)

                        gen_cond["rot"] = gen_cond["m2v"][:, :3, :3].clone()

                        if "light" in gen_cond:
                            light         = gen_cond["light"].reshape(-1, 9, 3).clone()
                            sh0, sh1, sh2 = torch.split(light, [1, 3, 5], dim=1)
                            gen_cond["sh0"] = sh0
                            gen_cond["sh1"] = sh1
                            gen_cond["sh2"] = sh2
                        
                        rcondition   = G.pack_condition(gen_cond)
                        rcondition_d = G.disc_G.pack_condition(gen_cond)
                    else:
                        rcondition = rcondition_d = None

                l_G, l_reg, l_gen = None, None, None
                l_D, l_r1,  l_dis = None, None, None

                def data2Dimg(data, req_grad=False):
                    seg_img = data["segim"]
                    ori_img = data["image"]

                    hH, hW  = ori_img.size(1), ori_img.size(2)
                    seg_img = resize_2d(seg_img,  hH, hW)
                    if req_grad:
                        seg_img = seg_img.requires_grad_(True)
                        ori_img = ori_img.requires_grad_(True)
                    # return torch.cat([seg_img, ori_img], dim=-1), (seg_img, ori_img)
                    return ori_img, seg_img, (ori_img, seg_img)

                if phase.name == "Gmain":
                    # D = D.eval()
                    # Gmain
                    with record_function('Gmain_forward'):
                        g_img, g_alp, g_dep, g_nrm, aux_d = G(noise, rcondition, update_emas=True)
                        f_scr = run_D(g_img, aux_d["segim"], rcondition_d)

                        l_G   = F.softplus(-f_scr).mean()
                        l_reg = 0

                    loss_d.update({"G": l_G, "Reg": l_reg, "all": l_G + l_reg})
                    out_d["Dx_fake"] = f_scr.mean()

                if phase.name == "Greg":
                    # Greg
                    l_G   = 0
                    l_reg = 0
                    l_gs  = 0

                    if attr_dict["Greg_method"] == "texture_symmetric":
                        with record_function('Greg_forward'):
                            loss_dict = {}

                            gen_img, g_alp, g_dep, g_nrm, aux_d = G(noise, rcondition, update_emas=True) # B,H,W,C

                            # if "gs_attr" in aux_d and "fs_xyz" in aux_d:
                            #     (gs_xyz, gs_rgb, gs_nrm, gs_opa, gs_sca, gs_rot) = aux_d["gs_attr"]   # (xyz, rgb, nrm, opa, sca, rot)
                            #     fs_xyz = aux_d["fs_xyz"]
                            #     hs_xyz = aux_d["hs_xyz"]
                            #     ts_xyz = aux_d.get("ts_xyz", None)

                            #     N_HS   = hs_xyz.size(1)
                            #     N_FS   = fs_xyz.size(1)
                            #     N_TS   = ts_xyz.size(1) if ts_xyz is not None else 0

                            #     if N_TS > 0:
                            #         hs_opa, fs_opa, ts_opa = torch.split(gs_opa, (N_HS, N_FS, N_TS), dim=1)
                            #         hs_sca, fs_sca, ts_sca = torch.split(gs_sca, (N_HS, N_FS, N_TS), dim=1)

                            #         training_stats.report('3DGS/head/opacity', hs_opa.mean())
                            #         training_stats.report('3DGS/head/scale',   hs_sca.mean())

                            #         training_stats.report('3DGS/free/opacity', fs_opa.mean())
                            #         training_stats.report('3DGS/free/scale',   fs_sca.mean())

                            #         training_stats.report('3DGS/teeth/opacity', ts_opa.mean())
                            #         training_stats.report('3DGS/teeth/scale',   ts_sca.mean())
                            #     else:
                            #         hs_opa, fs_opa = torch.split(gs_opa, (N_HS, N_FS), dim=1)
                            #         hs_sca, fs_sca = torch.split(gs_sca, (N_HS, N_FS), dim=1)

                            #         training_stats.report('3DGS/head/opacity', hs_opa.mean())
                            #         training_stats.report('3DGS/head/scale',   hs_sca.mean())

                            #         training_stats.report('3DGS/free/opacity', fs_opa.mean())
                            #         training_stats.report('3DGS/free/scale',   fs_sca.mean())

                            # gen_tex = aux_d["texture"] # B,H,W,C
                            gen_alb = aux_d["albedo"]  # B,H,W,C
                            gen_bmp = aux_d["bump"]    # B,H,W,C

                            # initial_coordinates = 2*torch.rand((image.shape[0], 1000, 2), device=device) - 1
                            # perturbed_coordinates = initial_coordinates + torch.randn_like(initial_coordinates)*(1/gen_alb.size(1))

                            # all_coordinates = torch.cat([initial_coordinates, perturbed_coordinates], dim=1)

                            # albedo = F.grid_sample(gen_alb.permute(0, 3, 1, 2), all_coordinates.unflatten(1, (-1, 1)), mode="bilinear", align_corners=False)
                            # albedo = albedo.flatten(-2).transpose(-2, -1)
                            # albedo_initial   = albedo[:, :albedo.shape[1]//2]
                            # albedo_perturbed = albedo[:,  albedo.shape[1]//2:]

                            # bump = F.grid_sample(gen_bmp.permute(0, 3, 1, 2), all_coordinates.unflatten(1, (-1, 1)), mode="bilinear", align_corners=False)
                            # bump = bump.flatten(-2).transpose(-2, -1)
                            # bump_initial   = bump[:, :albedo.shape[1]//2]
                            # bump_perturbed = bump[:,  albedo.shape[1]//2:]

                            gen_alb_flip = torch.flip(gen_alb, [-2])
                            l_sm = F.l1_loss(gen_alb, gen_alb_flip)
                            # l_alb_tv = F.l1_loss(albedo_initial, albedo_perturbed)
                            # l_bmp_tv = F.l1_loss(bump_initial, bump_perturbed)
                            l_alb_tv = total_variance_loss(gen_alb)
                            l_bmp_tv = total_variance_loss(gen_bmp)

                            l_tex = attr_dict.tex_symmetric*l_sm + attr_dict.tex_smooth*l_alb_tv + attr_dict.bump_smooth*l_bmp_tv

                            loss_dict["tex_symmetric"] = l_sm
                            loss_dict["tex_smooth"]    = l_alb_tv
                            loss_dict["bump_smooth"]   = l_bmp_tv

                            if attr_dict.depth_smooth > 0 and g_dep is not None:
                                l_dep_smooth = total_variance_loss(g_dep[..., :1])
                                l_tex = l_tex + attr_dict.depth_smooth*l_dep_smooth 
                                loss_dict["depth_smooth"] = l_dep_smooth

                            if attr_dict.pos_smooth > 0 and g_dep is not None and g_dep.size(-1) > 1:
                                l_pos_smooth = total_variance_loss(g_dep[..., 1:])
                                l_tex = l_tex + attr_dict.pos_smooth*l_pos_smooth
                                loss_dict["pos_smooth"] = l_pos_smooth

                            if attr_dict.img_smooth > 0:
                                # l_img_smooth = total_variance_loss(gen_img[..., -3:])
                                l_img_smooth = total_variance_loss(gen_img)
                                l_tex = l_tex + attr_dict.img_smooth*l_img_smooth
                                loss_dict["img_smooth"] = l_img_smooth

                            if attr_dict.normal_smooth > 0 and g_nrm is not None:
                                l_nrm_smooth = total_variance_loss(g_nrm)
                                l_tex = l_tex + attr_dict.normal_smooth*l_nrm_smooth
                                loss_dict["normal_smooth"] = l_nrm_smooth

                            if "background" in aux_d:
                                l_back_smooth = total_variance_loss(aux_d["background"])
                                l_tex = l_tex + attr_dict.back_smooth*l_back_smooth
                                loss_dict["back_smooth"] = l_back_smooth

                            if attr_dict.reg_bump is True:
                                l_bmp = gen_bmp.abs().mean()
                                l_tex = l_tex + attr_dict.reg_bump*l_bmp
                                loss_dict["reg_bump"] = l_bmp

                            if attr_dict.reg_triplane > 0:
                                if "vr_planes" in aux_d:
                                    planes = aux_d["vr_planes"]
                                elif "planes" in aux_d:
                                    planes = aux_d["planes"]

                                box_warp = 1.1*G.gs_renderer.field_kwargs.get("box_warp", 1)

                                initial_coordinates = torch.rand((image.shape[0], 1000, 3), device=device) - 0.5  # inside box_warp=1 [-0.5, -0.5, -0.5] [0.5, 0.5, 0.5]
                                initial_coordinates = box_warp * initial_coordinates
                                perturbed_coordinates = initial_coordinates + torch.randn_like(initial_coordinates) * G.gs_renderer.field_kwargs.get('density_reg_p_dist', 0.002)

                                all_coordinates = torch.cat([initial_coordinates, perturbed_coordinates], dim=1)
                                # sigma = G.sample_planes(all_coordinates, torch.randn_like(all_coordinates), planes)['rgb']
                                with G.using_generated(aux_d):
                                    sigma = G.gs_renderer.field_point_query(all_coordinates)
                                sigma_initial = sigma[:, :sigma.shape[1]//2]
                                sigma_perturbed = sigma[:, sigma.shape[1]//2:]
                                l_tex = l_tex + attr_dict.reg_triplane*F.l1_loss(sigma_initial, sigma_perturbed)
                            
                            if attr_dict.reg_scale > 0 and "gs_attr" in aux_d:
                                (gs_xyz, gs_rgb, gs_nrm, gs_opa, gs_sca, gs_rot) = aux_d["gs_attr"]   # (xyz, rgb, nrm, opa, sca, rot)
                                # l_sca = (torch.norm(gs_sca, p=float('inf'), dim=-1) - 0.005).relu().mean()
                                # 5e-4
                                l_sca = (gs_sca - 0.0005).square().mean()
                                l_gs  = l_gs + attr_dict.reg_scale*l_sca
                                loss_dict["reg_scale"] = l_sca

                            if attr_dict.reg_sorted > 0 and "gs_attr" in aux_d:
                                (gs_xyz, gs_rgb, gs_nrm, gs_opa, gs_sca, gs_rot) = aux_d["gs_attr"]   # (xyz, rgb, nrm, opa, sca, rot)
                                l_sort= F.relu(torch.diff(gs_sca, dim=-1)).mean()
                                l_gs  = l_gs + attr_dict.reg_sorted*l_sort
                                loss_dict["reg_sorted"] = l_sort

                            if attr_dict.reg_opacity > 0 and "gs_attr" in aux_d:
                                (gs_xyz, gs_rgb, gs_nrm, gs_opa, gs_sca, gs_rot) = aux_d["gs_attr"]   # (xyz, rgb, nrm, opa, sca, rot)
                                l_opa = (-gs_opa.mean())
                                l_gs  = l_gs + attr_dict.reg_opacity*l_opa
                                loss_dict["reg_opacity"] = l_opa

                            if attr_dict.reg_opacity_beta > 0 and "gs_attr" in aux_d:
                                (gs_xyz, gs_rgb, gs_nrm, gs_opa, gs_sca, gs_rot) = aux_d["gs_attr"]   # (xyz, rgb, nrm, opa, sca, rot)
                                # l_opa = (torch.log(gs_opa) + torch.log(1 - gs_opa)).mean()
                                l_opa = torch.log( (gs_opa*(1 - gs_opa)).clamp_min(1e-5) ).mean()
                                l_gs  = l_gs + attr_dict.reg_opacity_beta*l_opa
                                loss_dict["reg_opacity_beta"] = l_opa

                            # if attr_dict.reg_opacity_l1 > 0 and "gs_attr" in aux_d:
                            #     (gs_xyz, gs_rgb, gs_nrm, gs_opa, gs_sca, gs_rot) = aux_d["gs_attr"]   # (xyz, rgb, nrm, opa, sca, rot)
                            #     # l_opa = (torch.log(gs_opa) + torch.log(1 - gs_opa)).mean()
                            #     l_opa = (0.5 - (gs_opa - 0.5).abs()).mean()
                            #     l_gs  = l_gs + attr_dict.reg_opacity_l1*l_opa
                            #     loss_dict["reg_opacity_l1"] = l_opa

                            if attr_dict.reg_opacity_l2 > 0 and "gs_attr" in aux_d:
                                (gs_xyz, gs_rgb, gs_nrm, gs_opa, gs_sca, gs_rot) = aux_d["gs_attr"]   # (xyz, rgb, nrm, opa, sca, rot)
                                # l_opa = (torch.log(gs_opa) + torch.log(1 - gs_opa)).mean()
                                l_opa = (gs_opa - gs_opa.square()).mean()
                                l_gs  = l_gs + attr_dict.reg_opacity_l2*l_opa
                                loss_dict["reg_opacity_l2"] = l_opa

                            if attr_dict.reg_normal > 0 and g_nrm is not None and "ray_direction" in aux_d:
                                ray_dir  = aux_d["ray_direction"]
                                assert ray_dir.shape == g_nrm.shape, f"{ray_dir.shape} != {g_nrm.shape}"
                                l_back   = F.relu(F.cosine_similarity(g_nrm, ray_dir, dim=-1)).mean()
                                l_gs     = l_gs + attr_dict.reg_normal*l_back
                                loss_dict["reg_normal"] = l_back

                            if attr_dict.reg_surface_pull > 0 and "vertex" in aux_d:
                                num_free_gs = 0
                                dist        = 0
                                fs_xyz      = []
                                for name in ["hair", "glass", "hairg", "cloth"]:
                                    if f"{name}_xyz_rest" in aux_d:
                                        fs_xyz.append(aux_d[f"{name}_xyz_rest"])
                                fs_xyz = torch.cat(fs_xyz, dim=1)

                                dist   = point_mesh_distance(fs_xyz, aux_d["vertex"].detach(), attr_dict.water_tight_tri, signed=False).mean()
                                l_p2s  = dist

                                l_gs  = l_gs + attr_dict.reg_surface_pull*l_p2s

                            if attr_dict.reg_inter_push > 0:
                                fs_xyz = aux_d["fs_xyz"]
                                dist = torch.cdist(fs_xyz, fs_xyz)  # B, N, N

                                l_gs = l_gs + attr_dict.reg_inter_push * torch.exp(-dist).mean()

                            if attr_dict.reg_fs_xyz_init > 0:
                                num_free_gs = 0
                                dist = 0
                                # for k, v in aux_d.items():
                                #     if torch.is_tensor(v):
                                #         print(k, v.shape)
                                for name in ["hair", "glass", "hairg", "teeth", "cloth", "free"]:
                                    if f"{name}_xyz_rest" in aux_d:
                                        dist = dist + (aux_d[f"{name}_xyz_rest"] - getattr(G, f"{name}_xyz_fixed")).square().sum()

                                        num_free_gs += aux_d[f"{name}_xyz_rest"].size(1)
                                
                                # print(dist, LOCAL_BATCH_SIZE, num_free_gs)
                                dist = dist / (LOCAL_BATCH_SIZE*num_free_gs)

                                # dist = [(aux_d[f"{name}_xyz_rest"] - getattr(G, f"{name}_xyz_fixed")).square().mean() for name in ["hair", "glass", "hairg", "teeth", "cloth"] if f"{name}_xyz_rest" in aux_d]
                                # dist = reduce(lambda a,b : a+b, dist)

                                l_gs = l_gs + (attr_dict.reg_fs_xyz_init)*dist
                                loss_dict["reg_fs_xyz_init"] = dist

                            # if attr_dict.reg_fs_xyz_init_l2 > 0:
                            #     l_free  = (aux_d["fs_xyz"] - G.fs_xyz_init).square().mean()
                            #     l_teeth = 0

                            #     if "ts_xyz_rest" in aux_d:
                            #         l_teeth = (aux_d["ts_xyz_rest"] - G.ts_xyz_init).square().mean() # only teeth gs

                            #     l_gs = l_gs + (attr_dict.reg_fs_xyz_init_l2)*l_free + (3*attr_dict.reg_fs_xyz_init_l2)*l_teeth
                            #     loss_dict["reg_fs_xyz_init_l2.free"]  = l_free
                            #     loss_dict["reg_fs_xyz_init_l2.teeth"] = l_teeth

                            # if attr_dict.reg_fs_xyz_norm > 0:
                                
                            #     l_free  = (torch.linalg.norm(aux_d["fs_xyz"], dim=-1)).mean()
                            #     l_teeth = 0
                            #     # if "ts_xyz_rest" in aux_d:
                            #     #     l_teeth = (torch.linalg.norm(aux_d["ts_xyz_rest"], dim=-1)).mean()
                            #     l_gs = l_gs + (attr_dict.reg_fs_xyz_norm) * l_free + (3*attr_dict.reg_fs_xyz_norm)*l_teeth
                            #     loss_dict["reg_fs_xyz_norm.free"]  = l_free
                            #     loss_dict["reg_fs_xyz_norm.teeth"] = l_teeth

                            # if attr_dict.reg_fs_xyz_offset > 0:
                            #     l_free  = (aux_d["fs_xyz"] - G.fs_xyz.detach()).square().mean()
                            #     l_teeth = 0
                            #     if "ts_xyz_rest" in aux_d:
                            #         l_teeth = (aux_d["ts_xyz_rest"] - G.ts_xyz.detach()).square().mean() # only teeth gs

                            #     l_gs = l_gs + (attr_dict.reg_fs_xyz_offset)*l_free + (3*attr_dict.reg_fs_xyz_offset)*l_teeth
                            #     loss_dict["reg_fs_xyz_delta.free"]  = l_free
                            #     loss_dict["reg_fs_xyz_delta.teeth"] = l_teeth

                            # if attr_dict.reg_fs_xyz_anchor > 0:
                            #     l_free  = (G.fs_xyz - G.fs_xyz_init).square().mean()
                            #     l_teeth = 0
                            #     if hasattr(G, "ts_xyz"):
                            #         l_teeth = (G.ts_xyz - G.ts_xyz_init).square().mean() # only teeth gs

                            #     l_gs = l_gs + (attr_dict.reg_fs_xyz_anchor)*l_free + (3*attr_dict.reg_fs_xyz_anchor)*l_teeth
                            #     loss_dict["reg_fs_xyz_align.free"]   = l_free
                            #     loss_dict["reg_fs_xyz_align.teeth"]  = l_teeth

                            l_reg += l_tex

                    loss_d.update({"G": l_G, "Reg": l_reg, "Gs": l_gs, "all": l_G + l_reg + l_gs})
                    loss_d["detail"] = loss_dict

                if phase.name == "Dreg":
                    # Dreg
                    with record_function('Dreg_forward'):
                        r_img, r_msk, i_lst = data2Dimg(data, req_grad=True)
                        r_scr = run_D(r_img, r_msk, condition_d)

                        l_D   = 0

                        with record_function('r1_grads'), stylegan.conv2d_gradfix.no_weight_gradients():
                            # print(r_scr.numel(), r_img.size(0))

                            if r_scr.numel() == r_img.size(0):
                                r1_grads = torch.autograd.grad(outputs=[r_scr.sum()], inputs=i_lst, create_graph=True, only_inputs=True)
                            else:
                                N_SCORE  = r_scr.numel() // r_img.size(0)
                                r1_grads = torch.autograd.grad(outputs=[r_scr.flatten(1).sum(dim=0)], inputs=i_lst, create_graph=True, only_inputs=True, 
                                    grad_outputs=torch.eye(N_SCORE, device=r_scr.device), is_grads_batched=True)

                        r1_gammas  = attr_dict.r1_gamma if isinstance(attr_dict.r1_gamma, (list, tuple)) else [attr_dict.r1_gamma] * len(r1_grads)
                        r1_penalty = [(gamma/2)*grad.square().sum([1,2,3]).mean() for gamma, grad in zip(r1_gammas, r1_grads)]
                        l_r1       = reduce(lambda a,b: a+b, r1_penalty)

                    loss_d.update({"D": l_D, "R1": l_r1, "all": l_D + l_r1})
                
                if phase.name == "Dmain":
                    # Dmain
                    with record_function('Dmain_forward'):
                        with torch.inference_mode():
                            g_img, g_alp, g_dep, g_nrm, aux_d = G(noise, rcondition)
                            # g_img = torch.cat([aux_d["segim"], g_img], dim=-1)
                            # f_img = Ni(No.invert(g_img))

                        f_img, f_msk = g_img.clone().requires_grad_(True), aux_d["segim"].clone().requires_grad_(True)
                        r_img, r_msk, i_lst = data2Dimg(data, req_grad=False)

                        f_scr = run_D(f_img, f_msk, rcondition_d)
                        r_scr = run_D(r_img, r_msk, condition_d)
                        with torch.no_grad():
                            if isinstance(r_scr, (tuple, list)):
                                all_r_scr = torch.cat(r_scr)
                            else:
                                all_r_scr = r_scr
                            
                            if device_control.ddp_enabled:
                                all_r_scr = device_control.gather(all_r_scr)
                            training_stats.report('Loss/signs/real', all_r_scr.sign())

                        l_D   = F.softplus(f_scr).mean() + F.softplus(-r_scr).mean()
                    l_r1  = 0

                    loss_d.update({"D": l_D, "R1": l_r1, "all": l_D + l_r1})
                    out_d["Dx_fake"] = f_scr.mean()
                    out_d["Dx_real"] = r_scr.mean()

            if (step != attr_dict.last_ada_step):
                ada_interval = 4
                # update_t
                if step % ada_interval == 0:
                    ada_stats.update()
                    # print(ada_stats['Loss/signs/real'], ada_target, GLOBAL_BATCH_SIZE, ada_interval, ADA_I)
                    adjust = np.sign(ada_stats['Loss/signs/real'] - ada_target) * (GLOBAL_BATCH_SIZE * ada_interval / ADA_I)
                    aug_diff.p = (aug_diff.p + adjust).clip(min=0., max=1.)
                    if "diffusion" in attr_dict.augment:
                        aug_diff.update_T()

                    if "ada" in attr_dict.augment:
                        aug_pipe.p.copy_((aug_pipe.p + adjust).clamp(0.0, 1.0))

                    if "adaptive_prob" in attr_dict.augment:
                        attr_dict.adaptive_prob = np.clip(attr_dict.adaptive_prob - adjust/4, 0.0, 1.0)

                    if "adaptive_mask" in attr_dict.augment:
                        # attr_dict.adaptive_mask = 
                        # D.current_seg_dropout_p
                        model_dict["D"].seg_dropout_target_prob = np.clip(model_dict["D"].seg_dropout_target_prob + adjust/4, 0.0, attr_dict.mask_dropout_max)

                    # if "adaptive_lr" in attr_dict.augment:
                    #     for phase in phases:
                    #         if phase.name not in ["Gmain", "Greg"]:
                    #             continue
                            
                    #         optim = phase.opt

                    #         for pg in optim.param_groups:
                    #             pg["lr"] = 

                attr_dict.last_ada_step = step

            loss_d["all"] = gain * loss_d["all"]

            return loss_d, out_d

        def compute_after_grad(phase, data, gain, step):
            image   = data["image"].to(device)
            loss_d  = {}

            if UPDATE_ALIVE and phase.name == "Gmain" and step % 500 == 0:
                G = model_dict["G"]
                alive = torch.rand_like(G.fs_xyz)[..., :1] > min(step / (100*1000), 1)

                if device_control.ddp_enabled:
                    device_control.broadcast_to_all(alive)

                G.fs_xyz_alive.copy_(alive)
                G.fs_xyz_fixed.copy_(G.fs_xyz)

                print("update free-GS alive", step, alive.shape, alive.float().mean())

        iter_valid_loader = None
        @main_process_dcr
        @torch.no_grad()
        def log_function(epoch, step, locals={}):
            if "snapshot_data" not in locals:
                gw = np.clip(7680 // RES, 7, 16)
                gh = np.clip(4320 // RES, 4, 16)
                G  = model_dict["G_ema"].eval()

                noise_all, cond_all = [], []
                total_length = 0
                snapshot_num = gh*gw

                while total_length < snapshot_num:
                    for data in train_loader:
                        condition = G.pack_condition(data)
                        noise     = NOISE_SCALE*torch.randn([data["image"].size(0)]+SHAPE_LATENT)

                        noise_all.append(noise)
                        if condition is not None:
                            cond_all.append(condition)

                        total_length += data["image"].size(0)

                        if total_length > snapshot_num:
                            break
                
                noise_all = torch.cat(noise_all, dim=0)[:snapshot_num].to(device)
                cond_all  = torch.cat(cond_all,  dim=0)[:snapshot_num].to(device) if len(cond_all) > 0 else [None]*snapshot_num

                line1, line2, line3 = cond_all[:gw].repeat(3, 1).split((gw, gw, gw), dim=0)

                # m2v
                data = G.unpack_condition(line1.clone())
                origin = data["m2v"]
                # rot
                sy  = torch.linalg.norm(origin[:, :2, 0], dim=-1)
                yaw = torch.arctan2(origin[:, 0, 2], sy)
                index  = torch.multinomial(yaw.abs(), origin.size(0), replacement=True)
                rolled = origin[index].clone()
                rolled[:, :3, 3:] = origin[:, :3, 3:]   # copy original t3d
                # t3d
                dist   = torch.linalg.norm(origin[:, :3, 3].flatten(1), dim=-1)
                index  = torch.multinomial(dist, origin.size(0), replacement=True)
                s      = 1.2 + 0.3*(2*torch.rand(origin.size(0), device=device)-1)   # [0.9 - 1.5]
                rolled[:, :3, 3] = origin[index][:, :3, 3]*s.reshape(-1, 1)    # copy modified t3d
                data["m2v"] = rolled
                l_m2v = G.pack_condition(data)

                # shape
                data = G.unpack_condition(line2.clone())
                data["shape"] = torch.randn_like(data["shape"])
                l_shape = G.pack_condition(data)

                # light
                data = G.unpack_condition(line3.clone())
                rad = (2*torch.rand(line3.size(0))-1)*torch.pi/2  # [-pi/4, pi/4]
                cos = torch.cos(rad)
                sin = torch.sin(rad)
                b33 = torch.eye(3)[None].repeat(rad.size(0), 1, 1) # B, 3, 3
                b33[:, 0, 0] = cos
                b33[:, 0, 1] = sin
                b33[:, 1, 0] = -sin
                b33[:, 1, 1] = cos
                data["light"] = rotate_sh(data["light"], b33)
                l_light = G.pack_condition(data)

                noise_all[gw:4*gw] = noise_all[:gw].repeat(3, 1)
                cond_all[ gw:4*gw] = torch.cat((l_light, l_shape, l_m2v), dim=0)

                locals["snapshot_data"] = ((gh, gw), noise_all, cond_all)

            nonlocal iter_valid_loader
            try:
                data = next(iter_valid_loader)
            except:
                iter_valid_loader = iter(valid_loader)
                data = next(iter_valid_loader)
            
            print(data["glass_prob"])

            G  = model_dict["G_ema"].eval()
            # No = model_dict["norm_o"]
            image   = data["image"].to(device)
            index   = data["index"].to(device)
            condition = G.pack_condition(data)
            # condition_d = G.disc_G.pack_condition(data)
            if condition is not None:
                condition = condition.to(device)
                rcondition= condition[torch.randperm(image.size(0)).to(device)]

                nview_data= {}
                nview_data.update(data)
                if "m2v" in data:
                    m2v = data["m2v"]
                    nview_data["m2v"] = m2v.roll(1, 0)
                nview_cond= G.pack_condition(nview_data).to(device)
            else:
                rcondition= None
                nview_cond= None
            # K, pose = data["K"].to(device), data["pose"].to(device)
            # rK,rpose= gen_k_pose(image.size(0))
            pix_c   = image   # N, 3
            _,h,w,c = image.shape

            t0, t1, t2, t3 = [torch.cuda.Event(enable_timing=True) for _ in range(4)]

            def get_save_image(image, render_ret):
                BS    = image.size(0)
                ilist = []

                if "segim" in data:
                    segim = grid_image(data["segim"].to(device), h=RES, w=RES).clamp(0, 1)
                    print("segim", segim.shape, data["segim"].shape)
                    ilist.append(segim)

                i_img = grid_image(image, h=RES, w=RES).clamp(0, 1)
                ilist.append(i_img)

                if render_ret[0].size(-1) == 3:
                    recon = grid_image(render_ret[0][..., :3], h=RES, w=RES).clamp(0, 1)
                    ilist += [recon]
                else:
                    recon = grid_image(render_ret[0][..., :-3], h=RES, w=RES).clamp(0, 1).expand_as(i_img)
                    ilist += [recon]

                    recon2 = grid_image(render_ret[0][..., -3:], h=RES, w=RES).clamp(0, 1)
                    ilist += [recon2]
                    # ilist += list(map(lambda im: grid_image(im, h=RES, w=RES).clamp(0, 1), render_ret[0].split(3, dim=-1)))
                
                if "segim" in render_ret[4]:
                    recon = grid_image(render_ret[4]["segim"], h=RES, w=RES).clamp(0, 1)
                    ilist += [recon]

                if render_ret[2] is not None:
                    if render_ret[2].size(-1) == 1:
                        depth = grid_image(render_ret[2].expand(-1,-1,-1,3), h=RES, w=RES)
                        # depth = -depth
                        depth = (depth-depth.min()) / (depth.max()-depth.min())
                        ilist.append(depth)
                    else:
                        depth = grid_image(render_ret[2][..., :1].expand(-1,-1,-1,3), h=RES, w=RES)
                        # depth = -depth
                        depth = (depth-depth.min()) / (depth.max()-depth.min())
                        ilist.append(depth)

                        ilist.append(grid_image(render_ret[2][..., 1:]*2+0.5, h=RES, w=RES).clamp(0, 1))
                if render_ret[3] is not None:
                    normal = grid_image(render_ret[3].expand(-1,-1,-1,3), h=RES, w=RES)
                    ilist.append(0.5+0.5*normal)
                if render_ret[4] is not None:
                    for k, v in render_ret[4].items():
                        print(k, getattr(v, "shape", type(v)))

                    if "shading" in render_ret[4]:
                        ilist.append(grid_image(render_ret[4]["shading"], h=RES, w=RES).clamp(0, 1))

                    # if "ray_origin" in render_ret[4] and "ray_direction" in render_ret[4] and "vertex" in render_ret[4]:
                    #     ver, tri = render_ret[4]["vertex"], render_ret[4]["triangle"]
                    #     org, dir = render_ret[4]["ray_origin"], render_ret[4]["ray_direction"]
                    #     b,h,w,_  = dir.shape

                    #     query_ray = torch.cat([org.expand_as(dir), dir], dim=-1) # 1,N,6
                    #     faces = ver[:, tri.flatten()].reshape(b, -1, 3, 3)
                    #     f_nrm = F.normalize(torch.cross(faces[:,:,1]-faces[:,:,0], faces[:,:,2]-faces[:,:,0],dim=-1),dim=-1)
                    #     i_map, d_map, p_map = PrimaryRayFunction.apply(faces, query_ray.flatten(1, 2))                    

                    #     i_map = i_map.reshape(b, h, w)

                    #     im_nrm = torch.zeros_like(image, device=device)
                    #     for bi in range(b):
                    #         mask = i_map[bi]>=0
                    #         im_nrm[bi][mask] = f_nrm[bi, i_map[bi][mask]]

                    #     im_nrm = grid_image(im_nrm, h=RES, w=RES)
                    #     ilist.append(0.5+0.5*im_nrm)

                    if "background_uv" in render_ret[4]:
                        b,h,w,_= image.shape
                        uv     = render_ret[4]["background_uv"].flatten(1, 2)
                        u, v   = torch.split(uv, 1, dim=-1)
                        pix_id = ((v*h).long()*w + (u*w).long()).clamp_max(h*w-1)
                        im_spl = torch.zeros_like(image[..., :1].reshape(b,-1,1), device=device)
                        im_spl.scatter_add_(1, pix_id, torch.ones_like(pix_id, dtype=im_spl.dtype))
                        im_spl = grid_image((im_spl>0).reshape(b,h,w,-1).float().expand(-1,-1,-1,3), h=RES, w=RES)

                        ilist.append(im_spl)

                    if "depth2" in render_ret[4]:
                        depth2 = resize_2d(render_ret[4]["depth2"], RES, RES)
                        depth2 = grid_image(depth2.expand(-1,-1,-1,3), h=RES, w=RES)
                        depth2 = -depth2
                        depth2 = (depth2-depth2.min()) / (depth2.max()-depth2.min())
                        ilist.append(depth2)
                    if "albedo_init" in render_ret[4] and render_ret[4]["albedo_init"] is not None:
                        texture = render_ret[4]["albedo_init"].permute(0, 3, 1, 2)
                        texture = F.interpolate(texture, (RES,RES)).permute(0, 2, 3, 1)
                        texture = grid_image(texture, h=RES, w=RES)
                        albedo  = texture[..., :3]
                        ilist.append(albedo)
                    if "albedo" in render_ret[4]:
                        texture = render_ret[4]["albedo"].permute(0, 3, 1, 2)
                        texture = F.interpolate(texture, (RES,RES)).permute(0, 2, 3, 1)
                        texture = grid_image((texture), h=RES, w=RES)
                        albedo  = texture[..., :3]
                        ilist.append(albedo)
                    if "background" in render_ret[4]:
                        background = render_ret[4]["background"].permute(0, 3, 1, 2)
                        background = F.interpolate(background, (RES,RES)).permute(0, 2, 3, 1)
                        background = grid_image((background), h=RES, w=RES).clamp(0, 1)
                        # ilist.append(background[..., :3])
                        ilist.extend(background[..., :render_ret[0].size(-1)].split(3, dim=-1))
                    print([im.shape for im in ilist], RES)
                    i_img = torch.cat(ilist, dim=0).clamp(0, 1)
                else:
                    i_img = torch.cat([i_img, recon], dim=0)
                return i_img

            def get_save_plane(image, render_ret):
                BS    = image.size(0)
                ilist = []

                if render_ret[4] is not None:
                    aux_d = render_ret[4]
                    planes= None
                    if "vr_planes" in aux_d:
                        planes= aux_d["vr_planes"]
                    if "planes" in aux_d:
                        planes= aux_d["planes"]
                    
                    norms  = torch.linalg.norm(planes, dim=2) # B,K,H,W
                    pH,pW  = norms.shape[-2:]
                    norms  = (norms.flatten(2) / norms.flatten(2).max(dim=-1, keepdim=True).values).unflatten(-1, (pH, pW))
                    p0, p1, p2 = norms[:, 0], norms[:, 1], norms[:, 2]

                    ilist.append(grid_image(p0, h=pH, w=pW))
                    ilist.append(grid_image(p1, h=pH, w=pW))
                    ilist.append(grid_image(p2, h=pH, w=pW))

                    if hasattr(G.gs_renderer, "triplane_index"):
                        H, W   = norms.shape[-2:]

                        coordinates    = torch.cat([aux_d[f"{name}_xyz_query"] for name in ["hair", "glass", "hairg", "teeth", "cloth", "free"] if f"{name}_xyz_query" in aux_d], dim=1)

                        projected_coordinates = G.gs_renderer.field_2d_querypos(coordinates) # B,K,P,2
                        N, n_planes = coordinates.size(0), 3

                        u, v   = torch.split((projected_coordinates.reshape(N*n_planes,-1,2)*0.5 + 0.5).clamp(0, 1), 1, dim=-1)
                        pix_id = (v*(H-1)).long().clamp(0, H-1)*W + (u*(W-1)).long().clamp(0, W-1)
                        im_spl = torch.zeros((N*n_planes, H*W, 1), device=device)
                        # im_opa = torch.zeros((N*n_planes, H*W, 1), device=device)
                        im_spl.scatter_add_(1, pix_id, torch.ones_like(pix_id, dtype=im_spl.dtype))
                        # im_opa.scatter_add_(1, pix_id, fs_opa.reshape(N, 1, M, 1).expand(-1, n_planes, -1, -1).flatten(0, 1))
                        # im_opa = im_opa / im_spl

                        im_spl = (im_spl>0).reshape(N,n_planes,H,W,-1).float()

                        ilist.append(grid_image(im_spl[:, 0], h=pH, w=pW))
                        ilist.append(grid_image(im_spl[:, 1], h=pH, w=pW))
                        ilist.append(grid_image(im_spl[:, 2], h=pH, w=pW))

                        hs_uv  = G.gs_renderer.gs2d_uv_1_1[G.gs_renderer.gs2d_mask]
                        im_uv  = torch.zeros((N, H*W, 1), device=device)
                        u, v   = torch.split((hs_uv.reshape(1,-1,2).expand(N,-1,-1)*0.5 + 0.5).clamp(0, 1), 1, dim=-1)
                        pix_id = (v*(H-1)).long().clamp(0, H-1)*W + (u*(W-1)).long().clamp(0, W-1)
                        im_uv.scatter_add_(1, pix_id, torch.ones_like(pix_id, dtype=im_uv.dtype))
                        ilist.append(grid_image(im_uv, h=pH, w=pW))

                        # im_opa = (im_opa.clamp(0, 1)).reshape(N*n_planes,H,W,-1).float().expand(-1,-1,-1,3)

                        # im_spl = im_spl.reshape(N, n_planes, H, W, -1).detach().cpu().numpy()
                        # im_opa = im_opa.reshape(N, n_planes, H, W, -1).detach().cpu().numpy()

                i_img = torch.cat(ilist, dim=0).clamp(0, 1)
                return i_img

            def get_save_grad(image, render_ret, disc_cond):
                return
                ilist = []

                f_img = render_ret[0]
                f_msk = render_ret[4]["segim"]
                with torch.enable_grad():
                    f_img = f_img.requires_grad_(True)
                    f_scr = D(f_img, f_msk, disc_cond.to(f_img.device))
                    grads = torch.autograd.grad(outputs=[f_scr.sum()], inputs=(f_img, f_msk), only_inputs=True)[0]

                B,pH,pW,C = grads.shape

                g_seg, g_img = torch.split(grads, [grads.size(-1)-3, 3], dim=-1)
                i_seg, i_img = torch.split(f_img, [grads.size(-1)-3, 3], dim=-1)

                n_seg = torch.linalg.norm(g_seg, dim=-1, keepdim=True).expand(-1, -1, -1, 3) # B,H,W
                n_img = torch.linalg.norm(g_img, dim=-1, keepdim=True).expand(-1, -1, -1, 3) # B,H,W

                denom = (n_seg + n_img).max()
                n_seg = n_seg / denom
                n_img = n_img / denom

                ilist.append(grid_image(i_img, h=pH, w=pW))
                ilist.append(grid_image(n_seg, h=pH, w=pW))
                ilist.append(grid_image(n_img, h=pH, w=pW))

                i_img = torch.cat(ilist, dim=0).clamp(0, 1)
                return i_img

            save_image_dict = {}

            if TRAIN_GAN:
                # G = model_dict["G"]
                noise = NOISE_SCALE*torch.randn([image.size(0)]+SHAPE_LATENT, device=device)

                # with eval_model(G):
                print("eval_model", type(G), getattr(condition, "shape", type(condition)))
                render_ret = G(noise, condition)
                gan_img = get_save_image(image.clone(), render_ret)
                save_image_dict["GAN"] = gan_img

                gan_tpl = get_save_plane(image.clone(), render_ret)
                save_image_dict["triplane"] = gan_tpl

                # gan_grd = get_save_grad(image.clone(), render_ret, condition_d)
                # save_image_dict["Gradient"] = gan_grd

                # render_ret = G(noise, nview_cond)
                # gan_img = get_save_image(image.clone(), render_ret)
                # save_image_dict["GAN.random_pose"] = gan_img
                data_rot = {}
                for k, v in data.items():
                    if isinstance(v, torch.Tensor):
                        data_rot[k] = v[:1].expand(4, *([-1]*(v.ndim-1)))
                
                m2v = []
                for i, deg in enumerate([0, 30, 60, 90]):
                    r   = np.deg2rad(deg)
                    rot = np.array([
                        [ np.cos(r), 0, np.sin(r),    0],
                        [         0, 1,         0,    0],
                        [-np.sin(r), 0, np.cos(r), -1.5],
                        [         0, 0,         0,    1],
                    ])
                    # data_rot["m2v"][i].copy_(torch.as_tensor(rot).float().to(data_rot["m2v"].device))
                    m2v.append(torch.as_tensor(rot).float().to(data_rot["m2v"].device))

                data_rot["m2v"] = torch.stack(m2v, dim=0).float().to(data_rot["m2v"].device)

                # swapping_m2v_prob = 0.5
                # swapped = torch.rand((4, 1), device=data_rot["m2v"].device) < swapping_m2v_prob
                # print(swapped)
                # if "m2v" in data_rot:
                #     origin = data_rot["m2v"].clone()
                #     rolled = torch.roll(origin, 1, 0)
                #     # print(swapped.shape, rolled.shape, origin.shape)
                #     data_rot["m2v"] = torch.where(swapped.unsqueeze(-1), rolled, origin)
                
                noise      = noise[:1].expand(4, -1)
                nview_cond = G.pack_condition(data_rot).to(device)
                render_ret = G(noise[:image.size(0)], nview_cond[:image.size(0)])
                gan_img = get_save_image(image.clone(), render_ret)
                save_image_dict["GAN.random_pose"] = gan_img

                @torch.no_grad()
                def get_image(n, c):
                    image = G(n, c)[0]
                    image = (image)
                    return image.cpu().clamp(0, 1)

                if step % 5000 == 0:
                    (gh, gw), noise, cond = locals["snapshot_data"]
                    image = chunk_fn(get_image, LOCAL_BATCH_SIZE, [noise, cond]) # gh*gw,H,W,3
                    # if device_control.ddp_enabled:
                    #     image = device_control.gather(image)
                    image = image[:gh*gw].unflatten(0, (gh, gw)).transpose(1, 2).flatten(0, 1).flatten(1, 2)
                    save_image_dict["GAN.snapshot"] = image

            for k, v in save_image_dict.items():
                img = (255*v.detach().cpu().numpy()).astype(np.uint8).copy()
                if writer:
                    if RES < 256:
                        img = cv2.resize(img, None, fx=256/RES, fy=256/RES)
                    
                    print(k, img.shape, img.dtype)
                    writer.add_image(k, img, step, dataformats="HWC")
                    img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
                    cv2.imwrite(os.path.join(SAVE_ROOT, f"{k}-{step}.png"), img)

        @main_process_dcr
        @torch.no_grad()
        def save_checkpoint(epoch, step, cur_nimg, tag=None):
            import io
            import shutil
            from zipfile import ZipFile

            if tag is not None:
                filename = f"{epoch}-{step}-{tag}"
            else:
                filename = f"{epoch}-{step}"
            src = os.path.join(SAVE_ROOT, "code.zip")
            dst = os.path.join(SAVE_ROOT, filename+".zip")
            shutil.copyfile(src, dst)            
            pth_bits = io.BytesIO()

            with ZipFile(dst, "a") as zf:
                state_dict = {
                    "stat": {"epoch": epoch, "steps": step, "nimg": cur_nimg}
                }
                for k, m in model_dict.items():
                    state_dict[k] = m.state_dict()
                # torch.save(state_dict, os.path.join(SAVE_ROOT, filename+".pth"))
                torch.save(state_dict, pth_bits)

                zf.writestr("param.pth", pth_bits.getbuffer())

            # with open(os.path.join(SAVE_ROOT, filename+".pkl"), "wb") as f:
            #     import pickle
            #     pickle.dump(state_dict, f)

        device_control.sync_ddp()

        print("checking G")
        device_control.sync_check(model_dict["G"],     check_buffer=True)
        print("checking G_ema")
        device_control.sync_check(model_dict["G_ema"], check_buffer=True)
        print("checking D")
        device_control.sync_check(model_dict["D"],     check_buffer=True)

        device_control.sync_ddp()

        magnitude_tracker = MagnitudeTracker()

        # magnitude_tracker.track_weight(model_dict["G"].weight,   f"wgt_mlp.{i}")
        # magnitude_tracker.track_weight(model_dict["G"].fs_xyz,   f"wgt_fs_xyz")
        # magnitude_tracker.track_weight(model_dict["G"].fs_prm,   f"wgt_fs_prm")
        # magnitude_tracker.track_gradient(model_dict["G"].fs_xyz, f"grd_fs_xyz")
        # magnitude_tracker.track_gradient(model_dict["G"].fs_prm, f"grd_fs_prm")
        D = model_dict["D"]
        for res in ["4", "8", "16", "32", "64", "128", "256", "512"]:
            if hasattr(D, f"b{res}"):
                block = getattr(D, f"b{res}")

                if res in ["8"]:
                    magnitude_tracker.track_output(block,   f"act_b{res}")

                # conv  = None
                # for name in ["conv", "conv0"]:
                #     if hasattr(block, name):
                #         conv = getattr(block, name)
                #         break
                
                # if conv is not None and hasattr(conv, "weight"):
                #     magnitude_tracker.track_weight(conv.weight,   f"wgt_b{res}.{name}")
                #     magnitude_tracker.track_gradient(conv.weight, f"grd_b{res}.{name}")

            if hasattr(D, f"sb{res}"):
                block = getattr(D, f"sb{res}")

                if res in ["8"]:
                    magnitude_tracker.track_output(block,   f"act_sb{res}")

                # conv  = None
                # for name in ["conv", "conv0"]:
                #     if hasattr(block, name):
                #         conv = getattr(block, name)
                #         break
                
                # if conv is not None and hasattr(conv, "weight"):
                #     magnitude_tracker.track_weight(conv.weight,   f"wgt_sb{res}.{name}")
                #     magnitude_tracker.track_gradient(conv.weight, f"grd_sb{res}.{name}")

        # with torch.cuda.amp.autocast(enabled=global_amp_scaler.is_enabled()), torch.no_grad():
        #     log_function(0, -1)
        #     save_checkpoint(0, -1, -1, "init")
    
    if args.profile:
        from torch.profiler import profile, ProfilerActivity
        train_loop_ctx = lambda : profile(
            activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA],
            schedule=torch.profiler.schedule(
                            wait=5,
                            warmup=5,
                            active=10,
                        ),
            with_stack=True, with_flops=True, with_modules=True,
            profile_memory=True
            )
    else:
        train_loop_ctx = lambda : nullcontext("train loop")

    with train_loop_ctx() as train_ctx:
        step     = resume_stat.get("step",  0)
        start_e  = resume_stat.get("epoch", 0)
        cur_nimg = resume_stat.get("nimg",  0)
        last_eval_nimg  = cur_nimg
        tick_start_nimg = cur_nimg
        tick_start_time = time.time()
        maintenance_time = tick_start_time - start_time
        for epoch in range(start_e, EPOCHS):
            # if args.enable_ddp:
            if device_control.ddp_enabled:
                train_loader.sampler.set_epoch(epoch)
            pbar = tqdm(total=len(train_loader),desc=f"{NAME} Epoch[{epoch}/{EPOCHS}]", mininterval=1)
            for i, data in enumerate(train_loader):
                with record_function('data_fetch'):
                    # prepare data
                    for k, v in data.items():
                        if torch.is_tensor(v):
                            data[k] = data[k].to(device, copy=True, non_blocking=True)

                            # print(k, v.shape)

                # phase-based training
                for phase in phases:
                    if step % phase.interval != 0:
                        continue
                    
                    # device_control.local_print(f"{step} {phase.interval} {phase.name}")

                    with phase_train_ctx(phase, record_time=step%100==0 or step-phase.last_log_step>100) as (amp_scaler, optim, lr_schd):

                        if ZERO_GRAD_NONE is True:
                            for param in phase.all_params:
                                param.grad = None
                        else:
                            for param in phase.all_params:
                                if param.grad is not None:
                                    param.grad.zeros_()

                        params = []
                        for pg in optim.param_groups:
                            for p in pg["params"]:
                                params.append(p)
                        with torch.cuda.amp.autocast(enabled=amp_scaler.is_enabled()), multi_device_ctx(params), record_function("forward"):
                            loss_d, out_d = compute_loss(phase, data, gain=phase.interval, step=step)
                            loss          = loss_d["all"] / ACCUMULATE_ITER

                        with record_function("backward"):
                            amp_scaler.scale(loss).backward()
                        
                        with record_function("optimizer"):
                            magnitude_tracker.manual_trigger_gradient()

                            # # FIXME: phase with strange interval may accumulate too long i.e. interval=7, acc_iter=8
                            # if ( step+1 ) % ACCUMULATE_ITER == 0:
                            amp_scaler.unscale_(optim)

                            device_control.broadcast_optimizer(optim, rank_ps) # async_op is already waited

                            for pg in optim.param_groups:
                                params = pg['params']
                                for p in filter(lambda p: p.grad is not None, params):
                                    p.grad.data.nan_to_num_(nan=0.0, posinf=0.0, neginf=0.0)
                                if GRAD_CLAMP is not None:
                                    torch.nn.utils.clip_grad_value_(params, GRAD_CLAMP)
                            amp_scaler.step(optim)
                            amp_scaler.update()
                            compute_after_grad(phase, data, phase.interval, step)

                            # optim.zero_grad(set_to_none=ZERO_GRAD_NONE) # optim may only contain partial parameters due to ZeRO
                            # if ZERO_GRAD_NONE is True:
                            #     for param in phase.all_params:
                            #         param.grad = None
                            # else:
                            #     for param in phase.all_params:
                            #         if param.grad is not None:
                            #             param.grad.zeros_()

                            device_control.broadcast_parameter(rank_ps)

                            magnitude_tracker.manual_trigger_weight()

                        if lr_schd is not None:
                            lr_schd.step()

                    # logging training dynamics
                    if not args.profile and (step % 200 == 0 or step - phase.last_log_step > 200):
                        if writer:
                            for k, v in loss_d.items():
                                if torch.is_tensor(v):
                                    print(f"[{phase.name:^10}] {k:^6} {v.item()}")
                                    writer.add_scalar(f"loss/{phase.name}/{k}", v.item(), step)

                            for k, v in loss_d.get("detail", {}).items():
                                if torch.is_tensor(v):
                                    print(f"[{phase.name:^10}] {k:^6} {v.item()}")
                                    # writer.add_scalar(f"{phase.name}/{k}", v.item(), step)

                            # State
                            if phase.name == "Dmain":
                                Dx_dict = {k: out_d[k].item() for k in ["Dx_fake", "Dx_real"] if k in out_d}
                                
                                for k, v in Dx_dict.items():
                                    writer.add_scalar(f"State/{k}", v, step)
                            
                                writer.add_scalar("State/signs_real", ada_stats['Loss/signs/real'], step)

                                writer.add_scalar(f"aug/diffusion", aug_diff.p, step)
                                writer.add_scalar(f"aug/aug_pipe",  aug_pipe.p, step)

                        print(f"aug_diff.p {aug_diff.p}, aug_pipe.p {aug_pipe.p}")

                        phase.last_log_step = step

                # Update G_ema.
                if "G" in model_dict and "G_ema" in model_dict:
                    G, G_ema = model_dict["G"], model_dict["G_ema"]
                    with record_function('Gema'), torch.inference_mode():
                        ema_nimg = EMA_KIMG * 1000
                        if EMA_RAMPUP is not None:
                            ema_nimg = min(ema_nimg, cur_nimg * EMA_RAMPUP)
                        ema_beta = 0.5 ** (GLOBAL_BATCH_SIZE / max(ema_nimg, 1e-8))
                        for p_ema, p in zip(G_ema.parameters(), G.parameters()):
                            p_ema.copy_(p.lerp(p_ema, ema_beta))
                        for b_ema, b in zip(G_ema.buffers(), G.buffers()):
                            b_ema.copy_(b)

                    G_ema.train_step = G.train_step

                # logging
                if not args.profile and (step % LOG_I == 0 or step in [STEPS-1]):
                    torch.cuda.empty_cache()
                    with torch.cuda.amp.autocast(enabled=global_amp_scaler.is_enabled()):
                        log_function(epoch, step)
                    torch.cuda.empty_cache()

                    for phase in phases:
                        phase.end_event.synchronize()
                        time_ms = phase.start_event.elapsed_time(phase.end_event)
                        print(f"{phase.name} time:{time_ms:.2f}ms last_log_step:{phase.last_log_step}")

                        if writer:
                            if step >= 100:
                                writer.add_scalar(f"Timing/{phase.name}", time_ms, step)

                            # lr, wd = 0, 0
                            # for param_group in phase.opt.param_groups:
                            #     lr = param_group['lr']
                            #     wd = param_group["weight_decay"]
                            #     betas = param_group["betas"]
                            #     print(f"lr: {lr:.10f}, wd: {wd:.10f}, betas: {betas}")
                            
                            # writer.add_scalar(f"params/{phase.name}/lr", lr, step)
                            # writer.add_scalar(f"params/{phase.name}/wd", wd, step)
                            # writer.add_scalar(f"params/{phase.name}/beta0", betas[0], step)
                            # writer.add_scalar(f"params/{phase.name}/beta1", betas[1], step)

                    stats_collector.update()
                    stats_dict = stats_collector.as_dict()

                    # Update logs.
                    if writer is not None:
                        global_step = int(cur_nimg / 1e3)
                        # value = stats_dict["Loss/signs/real"]
                        # writer.add_scalar("loss/signs/real", value.mean, step)

                        writer.add_scalar("State/dropout/seg", D.current_seg_dropout_p, step)
                        writer.add_scalar("State/adaptive_p",  attr_dict.adaptive_prob, step)
                        # writer.add_scalar("loss/adaptive_m",  attr_dict.adaptive_mask, step)

                        if hasattr(G, "shading_alpha"):
                            if torch.is_tensor(G.shading_alpha):
                                writer.add_scalar("State/shading_alpha",  G.shading_alpha.item(), step)

                        for k in ["head", "free", "teeth"]:
                            for attr_name in ["opacity", "scale"]:
                                key = f"3DGS/{k}/{attr_name}"
                                if key not in stats_dict:
                                    continue
                                
                                print(key)
                                value = stats_dict[key]
                                writer.add_scalar(f"3dgs/{k}/{attr_name}", value.mean, step)

                        avg_dict, cur_dict = magnitude_tracker.as_dict()
                        for k, v in avg_dict.items():
                            print(f"[magnitude] {k:^6} {v.item()}")
                            writer.add_scalar(f"magnitude/{k}", v.item(), step)

                        # fs_rgb, fs_nrm, fs_sca, fs_opa = torch.split(G.fs_prm, (G.rgb_channels, 4, 3, 1), dim=-1)
                        # for n, v in zip(
                        #     ["rgb", "rot", "scale", "opacity"],
                        #     [fs_rgb, fs_nrm, fs_sca, fs_opa]
                        #     ):
                        #     v = torch.linalg.vector_norm(v)
                        #     print(f"[magnitude] fs_{n} {v.item()}")
                        #     writer.add_scalar(f"magnitude/fs_{n}", v.item(), step)

                        magnitude_tracker.clear()

                # if (cur_nimg != 0 and cur_nimg % EVAL_I == 0) or (step in [STEPS-1]):
                if not args.profile and ((cur_nimg != 0 and (cur_nimg - last_eval_nimg) >= EVAL_I) or (step in [STEPS-1])):
                    last_eval_nimg = cur_nimg
                    G_eval, No = model_dict["G"], model_dict["norm_o"]
                    if "G_ema" in model_dict:
                        G_eval = model_dict["G_ema"].eval()

                    def gen_sample_fn(data):
                        # pixel in range [0, 1]
                        noise      = torch.randn([data["image"].size(0)]+SHAPE_LATENT, device=device)
                        # cond       = data
                        index      = torch.randint(0, len(all_cond_dict["shape"]), (len(noise),), device=device)
                        cond       = {k: v[index].clone() for k, v in all_cond_dict.items()}

                        condition  = G_eval.pack_condition(cond)
                        if condition is not None:
                            condition = condition.to(device)
                        gen_images = (G_eval(noise, condition)[0]).clamp(0, 1)
                        if gen_images.size(-1) > 3:
                            gen_images = gen_images[..., -3:].contiguous()
                        return gen_images
                        
                    K = 5 if G.img_resolution < 512 else 3
                    fid, (real_img, gen_img, gen_img_sorted) = get_fid(eval_loader, gen_sample_fn, device, num_generate=50*1000, K=K, verbose=True)

                    G_eval = model_dict["G"].eval()
                    def gen_sample_fn_g(data):
                        # pixel in range [0, 1]
                        noise      = torch.randn([data["image"].size(0)]+SHAPE_LATENT, device=device)
                        # cond       = data
                        index      = torch.randint(0, len(all_cond_dict["shape"]), (len(noise),), device=device)
                        cond       = {k: v[index].clone() for k, v in all_cond_dict.items()}

                        condition  = G_eval.pack_condition(cond)
                        if condition is not None:
                            condition = condition.to(device)
                        gen_images = (G_eval(noise, condition)[0]).clamp(0, 1)
                        if gen_images.size(-1) > 3:
                            gen_images = gen_images[..., -3:].contiguous()
                        return gen_images
                    fid_g, (_, _, _) = get_fid(eval_loader, gen_sample_fn_g, device, num_generate=50*1000, K=1, verbose=True)

                    if writer:
                        global_step = int(cur_nimg / 1e3)
                        writer.add_image(f"Eval/real", real_img, global_step, dataformats="HWC")
                        cv2.imwrite(os.path.join(SAVE_ROOT, f"real-{global_step}.png"), cv2.cvtColor(real_img, cv2.COLOR_RGB2BGR))

                        writer.add_image(f"Eval/fake", gen_img, global_step, dataformats="HWC")
                        cv2.imwrite(os.path.join(SAVE_ROOT, f"fake-{global_step}.png"), cv2.cvtColor(gen_img, cv2.COLOR_RGB2BGR))

                        writer.add_image(f"Eval/fake_sorted", gen_img_sorted, global_step, dataformats="HWC")
                        cv2.imwrite(os.path.join(SAVE_ROOT, f"fake-{global_step}.png"), cv2.cvtColor(gen_img_sorted, cv2.COLOR_RGB2BGR))

                    
                    print(f"FID: {fid}/{fid_g}")
                    if writer:
                        writer.add_scalar(f"Metric/FID",   float(fid),   global_step)
                        writer.add_scalar(f"Metric/FID_G", float(fid_g), global_step)

                        val = {
                            "FID_50K": float(fid)
                        }

                        mdt_key   = ["FID_50K"]
                        mdt_head  = "|".join([k for k in mdt_key])
                        mdt_sepr  = "|".join([ "---" for k in mdt_key])
                        mdt_data  = "|".join([ str(val.get(k, None)) for k in mdt_key])
                        md_str    = f"{mdt_head}\n{mdt_sepr}\n{mdt_data}"

                        writer.add_text(f"Eval", md_str, global_step)

                        save_checkpoint(epoch, step, cur_nimg, f"fid{fid:.4f}")

                # saving
                if not args.profile and step % SAVE_I == 0:
                    save_checkpoint(epoch, step, cur_nimg)

                # all_steps = device_control.gather(step)
                # if step % 10 == 0:
                #     device_control.local_print(f"all_steps={all_steps}")
                # device_control.sync_ddp()

                cur_nimg += GLOBAL_BATCH_SIZE
                step += 1
                pbar.update()

                if args.profile:
                    train_ctx.step()

                    if i > 100:
                        break

            pbar.close()
            if args.profile:
                break

        save_checkpoint(epoch, step, cur_nimg)

    if args.profile:
        if writer is not None:
            with open("profile.log", "w") as f:
                print(train_ctx.key_averages().table(sort_by="self_cuda_time_total", row_limit=100), file=f)
                print("================== CPU ====================", file=f)
                print(train_ctx.key_averages().table(sort_by="self_cpu_time_total", row_limit=100), file=f)

        os._exit(0)

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("-c", "--config",   type=str, required=True)
    parser.add_argument("-f", "--function", type=str, default="train")

    parser.add_argument("--checkpoint", type=str, default=None)

    parser.add_argument("--max_length",  type=int, default=None)
    parser.add_argument("--batch_size", type=int, default=None)
    parser.add_argument("--enable_ddp", type=lambda x:bool(eval(x)), default=False)
    parser.add_argument("--enable_amp", type=lambda x:bool(eval(x)), default=False)
    parser.add_argument("--deterministic", type=lambda x:bool(eval(x)), default=False)

    parser.add_argument("--debug", action="store_true")

    args, unparsed = parser.parse_known_args()
    print(args)
    print(unparsed)

    parser = argparse.ArgumentParser()
    parser.add_argument("--resume",  type=lambda x:bool(eval(x)), default=False)
    parser.add_argument("--profile", type=lambda x:bool(eval(x)), default=False)
    ext_args = parser.parse_args(unparsed)

    for k, v in vars(ext_args).items():
        setattr(args, k, v)

    train(args)