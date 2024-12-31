import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.cpp_extension import load

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))

EXT  = os.path.join(os.path.dirname(__file__), "ext")
NAME = "kplane_mlp"
ext_root = os.path.join(EXT, NAME)
kplane_mlp_impl = load(name=NAME, 
                        sources=[
                            os.path.join(ext_root, 'kplane_mlp_cuda.cpp'), 
                            os.path.join(ext_root, 'kplane_mlp_cuda.cu'),
                            os.path.join(ext_root, 'kplane_cuda.cu'),
                            # os.path.join(ext_root, 'kplane_mlp_cuda.cuh'),
                            # os.path.join(ext_root, 'cuda_fc.cuh'),
                            ], verbose=True)

num_bad = lambda x:x.numel() - torch.isfinite(x).sum().item()
num_nan = lambda x:torch.isnan(x).sum().item()
num_inf = lambda x:torch.isinf(x).sum().item()

debug = False

def kplane_mlp(kplane_param, mlp_param, grid, skip, 
    mode="bilinear", padding_mode="zeros", align_corners=True, feature_fusion="sum", 
    mlp_layers=2, mlp_dim_hidden=64, mlp_dim_output=32, mlp_activation="softplus"):
    '''
    kplane_param: [B, K, C, H, W]
    mlp_param:    [B, (ci*c), ...]
    grid:         [B, p, q, K, 2]
    skip:         [B, p, q, skip_C]
    '''
    # feat = Triplane(grid)
    # out  = MLP(feat, skip)

    if mode == "bilinear":
        mode_enum = 0
    elif mode == "nearest":
        mode_enum = 1
    # else:  # mode == 'bicubic'
    #     mode_enum = 2

    if padding_mode == "zeros":
        padding_mode_enum = 0
    elif padding_mode == "border":
        padding_mode_enum = 1
    else:  # padding_mode == 'reflection'
        padding_mode_enum = 2

    if feature_fusion == "sum":
        feature_fusion_enum = 0
    elif feature_fusion == "avg":
        feature_fusion_enum = 1
    elif feature_fusion == "mul":
        feature_fusion_enum = 2
    else:  # feature_fusion == "cat'
        feature_fusion_enum = 3

    if mlp_activation == "softplus":
        mlp_activation_enum = 0
    elif mlp_activation == "relu":
        mlp_activation_enum = 1
    
    kplane_param = kplane_param.permute(0, 3, 4, 1, 2).contiguous()
    # grid         = grid.contiguous()
    # skip         = skip.contiguous()

    if mlp_layers == 0:
        return _Kplane.apply(kplane_param, 
            grid.type_as(kplane_param), skip.type_as(kplane_param), 
            mode_enum, padding_mode_enum, align_corners, feature_fusion_enum,  # triplane grid sampling
        )
    else:
        if mlp_param is None:
            mlp_param = torch.empty((kplane_param.size(0), 1), device=kplane_param.device).requires_grad_(False)
        
        if mlp_layers == 1:
            mlp_dim_hidden = mlp_dim_output
        

        return _KplaneMLP.apply(kplane_param, mlp_param, 
            grid, skip, 
            mode_enum, padding_mode_enum, align_corners, feature_fusion_enum,  # triplane grid sampling
            mlp_layers, mlp_dim_hidden, mlp_dim_output, mlp_activation_enum,   # MLP config
        )

class _Kplane(torch.autograd.Function):
    @staticmethod
    def forward(ctx, kpl_param, grid, skip, 
        mode=0, padding_mode=0, align_corners=False, feature_fusion=0,
        ):

        assert kpl_param.ndim == 5, f"kplane feature shold be [B, H, W, K, C] got {kpl_param.shape}"
        assert kpl_param.shape[4] % 4 == 0, f"num of channel ({kpl_param.shape}) is not divided by 4"
        assert grid.ndim == 5, f"grid shold be [B, p, q, K, 2] got {grid.shape}"
        assert grid.shape[4] == 2, f"grid shold be [B, p, q, K, 2] got {grid.shape}"
        assert kpl_param.shape[0] == grid.shape[0], f"num batch is mismatch got [B,H,W,K,C]={kpl_param.shape} [B,p,q,K,2]={grid.shape}"
        assert kpl_param.shape[-2] == grid.shape[-2], f"num plane is mismatch got [B,H,W,K,C]={kpl_param.shape} [B,p,q,K,2]={grid.shape}"

        assert grid.shape[:3] == skip.shape[:3], f"grid [B, p, q] != skip [B, p, q] got {grid.shape} {skip.shape}"

        # dtypes  = [kpl_param.dtype, mlp_param.dtype, grid.dtype, skip.dtype]
        # devices = [kpl_param.device, mlp_param.device, grid.device, skip.device]
        # assert kpl_param.dtype == mlp_param.dtype, f"{dtypes}"
        # assert kpl_param.dtype == grid.dtype,      f"{dtypes}"
        # assert kpl_param.dtype == skip.dtype,      f"{dtypes}"

        # assert kpl_param.is_cuda, f"{devices}"
        # assert grid.is_cuda,      f"{devices}"
        # assert skip.is_cuda,      f"{devices}"

        B, p, q = grid.shape[:3]

        ctx.tpl_conf = (mode, padding_mode, align_corners, feature_fusion)

        req_grad = list(map(lambda x:x.requires_grad, [kpl_param, grid, skip]))
        ctx.req_grad = req_grad

        # oC = mlp_dim_output if mlp_layers > 0 else kpl_param.size(2) + skip.size(-1)
        # output = torch.empty((B, p, q, oC), dtype=kpl_param.dtype, device=kpl_param.device)

        try:
            output = kplane_mlp_impl.kplane_forward(kpl_param, grid, skip, *ctx.tpl_conf)

            if debug:
                assert num_bad(output) == 0, f"bad grid {num_bad(grid)}, bad skip {num_bad(skip)}, output got NaN {num_nan(output)}, Inf {num_inf(output)}, conf={ctx.tpl_conf}"
        except Exception as ex:
            dump_obj = {
                "input": [kpl_param, grid, skip, *ctx.tpl_conf, *ctx.mlp_conf],
                "output": output,
                "req_grad":  req_grad
            }
            dump_pth = os.path.join(ROOT, "temp", "kernel_dump", "kplane_mlp_forward.pth")

            os.makedirs(os.path.dirname(dump_pth), exist_ok=True)
            torch.save(dump_obj, dump_pth)

            print(f"dump @ {dump_pth}")

            raise ex

        ctx.save_for_backward(kpl_param, grid, skip)

        return output

    @staticmethod
    def backward(ctx, grad_output):
        # input, grid = ctx.saved_tensors

        kpl_param, grid, skip = ctx.saved_tensors

        req_grad = ctx.req_grad

        grad_kpl = grad_mlp = grad_grid = grad_skip = None

        try:
            grad_kpl, grad_grid, grad_skip = kplane_mlp_impl.kplane_backward(
                grad_output,
                kpl_param, grid, skip, *ctx.tpl_conf, req_grad)

            if debug: 
                if req_grad[0]:
                    print(0, kpl_param.shape, grad_kpl.shape, num_bad(grad_kpl))
                    assert num_bad(grad_kpl) == 0
                    grad_kpl.contiguous()
                if req_grad[2]:
                    print(2, grad_grid.shape, grid.shape, num_bad(grad_grid))
                    grad_grid.contiguous()
                if req_grad[3]:
                    print(2, grad_skip.shape, skip.shape, num_bad(grad_skip))
                    grad_skip.contiguous()
        except Exception as ex:
            dump_obj = {
                "input": [kpl_param, grid, skip, *ctx.tpl_conf, *ctx.mlp_conf],
                "grad_output": [grad_output],
                "req_grad":    req_grad
            }
            dump_pth = os.path.join(ROOT, "temp", "kernel_dump", "kplane_mlp_backward.pth")

            os.makedirs(os.path.dirname(dump_pth), exist_ok=True)
            torch.save(dump_obj, dump_pth)

            print(f"dump @ {dump_pth}")

            raise ex
        
        if not kpl_param.requires_grad:
            grad_kpl = None
        if not grid.requires_grad:
            grad_grid = None
        if not skip.requires_grad:
            grad_skip = None
        
        # print(kpl_param.shape, grad_kpl.shape)
        # print(grid.shape, grad_grid.shape)
        # print(skip.shape, grad_skip.shape)

        return grad_kpl, grad_grid, grad_skip, \
            None, None, None, None, \
            None, None, None, None,

class _KplaneMLP(torch.autograd.Function):
    @staticmethod
    def forward(ctx, kpl_param, mlp_param, grid, skip, 
        mode=0, padding_mode=0, align_corners=False, feature_fusion=0,
        mlp_layers=2, mlp_dim_hidden=64, mlp_dim_output=32, mlp_activation=0
        ):

        assert kpl_param.ndim == 5, f"kplane feature shold be [B, H, W, K, C] got {kpl_param.shape}"
        assert kpl_param.shape[4] % 4 == 0, f"num of channel ({kpl_param.shape}) is not divided by 4"
        assert grid.ndim == 5, f"grid shold be [B, p, q, K, 2] got {grid.shape}"
        assert grid.shape[4] == 2, f"grid shold be [B, p, q, K, 2] got {grid.shape}"
        assert kpl_param.shape[0] == grid.shape[0], f"num batch is mismatch got [B,H,W,K,C]={kpl_param.shape} [B,p,q,K,2]={grid.shape}"
        assert kpl_param.shape[-2] == grid.shape[-2], f"num plane is mismatch got [B,H,W,K,C]={kpl_param.shape} [B,p,q,K,2]={grid.shape}"

        assert grid.shape[:3] == skip.shape[:3], f"grid [B, p, q] != skip [B, p, q] got {grid.shape} {skip.shape}"

        # dtypes  = [kpl_param.dtype, mlp_param.dtype, grid.dtype, skip.dtype]
        # devices = [kpl_param.device, mlp_param.device, grid.device, skip.device]
        # assert kpl_param.dtype == mlp_param.dtype, f"{dtypes}"
        # assert kpl_param.dtype == grid.dtype,      f"{dtypes}"
        # assert kpl_param.dtype == skip.dtype,      f"{dtypes}"

        # assert kpl_param.is_cuda, f"{devices}"
        # assert grid.is_cuda,      f"{devices}"
        # assert skip.is_cuda,      f"{devices}"

        B, p, q = grid.shape[:3]

        ctx.tpl_conf = (mode, padding_mode, align_corners, feature_fusion)
        ctx.mlp_conf = (mlp_layers, mlp_dim_hidden, mlp_dim_output, mlp_activation)

        req_grad = list(map(lambda x:x.requires_grad, [kpl_param, mlp_param, grid, skip]))
        req_grad[1] = False
        ctx.req_grad = req_grad

        # oC = mlp_dim_output if mlp_layers > 0 else kpl_param.size(2) + skip.size(-1)
        # output = torch.empty((B, p, q, oC), dtype=kpl_param.dtype, device=kpl_param.device)

        try:
            output = kplane_mlp_impl.kplane_mlp_forward(kpl_param, mlp_param, grid, skip, *ctx.tpl_conf, *ctx.mlp_conf)

            if debug:
                assert num_bad(output) == 0, f"bad grid {num_bad(grid)}, bad skip {num_bad(skip)}, output got NaN {num_nan(output)}, Inf {num_inf(output)}, conf={ctx.tpl_conf} {ctx.mlp_conf}"
        except Exception as ex:
            dump_obj = {
                "input": [kpl_param, mlp_param, grid, skip, *ctx.tpl_conf, *ctx.mlp_conf],
                "output": output,
                "req_grad":  req_grad
            }
            dump_pth = os.path.join(ROOT, "temp", "kernel_dump", "kplane_mlp_forward.pth")

            os.makedirs(os.path.dirname(dump_pth), exist_ok=True)
            torch.save(dump_obj, dump_pth)

            print(f"dump @ {dump_pth}")

            raise ex

        ctx.save_for_backward(kpl_param, mlp_param, grid, skip)

        return output

    @staticmethod
    def backward(ctx, grad_output):
        # input, grid = ctx.saved_tensors

        kpl_param, mlp_param, grid, skip = ctx.saved_tensors

        req_grad = ctx.req_grad

        grad_kpl = grad_mlp = grad_grid = grad_skip = None

        try:
            grad_kpl, grad_mlp, grad_grid, grad_skip = kplane_mlp_impl.kplane_mlp_backward(
                grad_output,
                kpl_param, mlp_param, grid, skip, *ctx.tpl_conf, *ctx.mlp_conf, req_grad)

            if debug: 
                if req_grad[0]:
                    print(0, kpl_param.shape, grad_kpl.shape, num_bad(grad_kpl))
                    assert num_bad(grad_kpl) == 0
                    grad_kpl.contiguous()
                if req_grad[2]:
                    print(2, grad_grid.shape, grid.shape, num_bad(grad_grid))
                    grad_grid.contiguous()
                if req_grad[3]:
                    print(2, grad_skip.shape, skip.shape, num_bad(grad_skip))
                    grad_skip.contiguous()
        except Exception as ex:
            dump_obj = {
                "input": [kpl_param, mlp_param, grid, skip, *ctx.tpl_conf, *ctx.mlp_conf],
                "grad_output": [grad_output],
                "req_grad":    req_grad
            }
            dump_pth = os.path.join(ROOT, "temp", "kernel_dump", "kplane_mlp_backward.pth")

            os.makedirs(os.path.dirname(dump_pth), exist_ok=True)
            torch.save(dump_obj, dump_pth)

            print(f"dump @ {dump_pth}")

            raise ex
        
        if not kpl_param.requires_grad:
            grad_kpl = None
        if not mlp_param.requires_grad:
            grad_mlp = None
        if not grid.requires_grad:
            grad_grid = None
        if not skip.requires_grad:
            grad_skip = None
        
        # print(kpl_param.shape, grad_kpl.shape)
        # print(grid.shape, grad_grid.shape)
        # print(skip.shape, grad_skip.shape)

        return grad_kpl, grad_mlp, grad_grid, grad_skip, \
            None, None, None, None, \
            None, None, None, None,