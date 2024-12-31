import os
import sys
import torch
import torch.nn as nn
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP

from typing import Callable
from functools import partial
from itertools import chain
import numpy as np

ROOT = os.path.abspath(os.path.join(__file__, "..", ".."))

def void_compile(model=None, **kwargs):
    if model is not None and callable(model):
        # torch.compile(model)
        return model
    else:
        # @torch.compile(kwargs0=xx, kwargs1=xxx)
        # def func(a, b)
        return lambda x:x
    
if not hasattr(torch, "compile"):
    print("[WARNING] 'torch.compile' is not supported")
    # torch.compile = lambda x,**kwargs:x

    torch.compile = void_compile

    torch_compile_or_jit = torch.jit.script
else:
    torch_compile_or_jit = torch.compile

    import torch._dynamo
    torch._dynamo.config.verbose = True
    torch._dynamo.config.suppress_errors = True


# torch_compile = partial(torch.compile, fullgraph=True, mode="reduce-overhead", 
# backend="eager")
torch_compile = torch.compile

compile_enabled = True
def compile(*args, **kwargs):
    if compile_enabled:
        return torch_compile(*args, **kwargs)
    else:
        return void_compile(*args, **kwargs)
    
class Timer(object):
    def __init__(self):
        super().__init__()

        self.event_list = []
        self.index      = 0
    
    def record(self):
        if self.index == len(self.event_list):
            self.event_list += [ torch.cuda.Event(enable_timing=True) for _ in range(4) ]

        self.event_list[self.index].record()
        self.index += 1
    
    def elapsed_ms(self):

        time_ms = [] # len(time_ms) == (self.index - 1)

        for e0, e1 in zip(self.event_list[:self.index-1], self.event_list[1:self.index]):
            time_ms.append(e0.elapsed_time(e1))
        return time_ms
    
    def synchronize(self):
        self.event_list[self.index-1].synchronize()
    
    def clear(self):
        self.index = 0

def make_deterministic(seed):
    import numpy as np
    import random

    np.random.seed(seed)
    random.seed(seed)
    torch.random.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    torch.backends.cudnn.deterministic = True
    torch.use_deterministic_algorithms(True, warn_only=True)

ddp_enabled = False

ori_stdout  = sys.stdout
def local_print(*args, _locals={}, **kwargs):
    print(f"Rank [{get_rank()}]", *args, **kwargs, file=ori_stdout)

    if "log_f" not in _locals:
        pid = os.getpid()
        fname = os.path.join(ROOT, "temp", "device_control", f"{pid}.log")
        os.makedirs(os.path.dirname(fname), exist_ok=True)
        saved_log = open(fname, "w")
        _locals["log_f"] = saved_log
    
    saved_log = _locals["log_f"]
    print(f"Rank [{get_rank()}]", *args, **kwargs, file=saved_log)
    saved_log.flush()

def init_device(enable_ddp=True):

    if enable_ddp:
        if "LOCAL_RANK" in os.environ:
            torch.cuda.set_device(int(os.environ["LOCAL_RANK"]))
            torch.cuda.empty_cache()
        # if "CUDA_VISIBLE_DEVICES" in os.environ:
        #     os.environ["CUDA_VISIBLE_DEVICES"]
        dist.init_process_group("nccl")
        # dist.init_process_group("gloo")
        rank       = dist.get_rank()
        device_id  = rank % torch.cuda.device_count()

        print(f"WORLD: {dist.get_world_size()} RANK: {rank} DEV: {device_id}")
        if rank != 0:
            print(f"RANK[{rank}] set stdout to NULL")
            sys.stdout.flush()
            sys.stdout = open(os.devnull, "w")
        else:
            sys.stdout.flush()
    else:
        device_id = 0

    device = torch.device(f'cuda:{device_id}' if torch.cuda.is_available() else 'cpu')
    torch.cuda.set_device(device)
    # torch.set_default_device(device)

    return device

def get_world_size():
    if dist.is_initialized():
        return dist.get_world_size()
    else:
        return 1

def get_rank():
    if dist.is_initialized():
        return dist.get_rank()
    else:
        return 0

class DDPWrapper(nn.Module):
    def __init__(self, **kwargs):
        super().__init__()

        def convert_nograd_params_to_buffers(module):
            ret = [[pn, prm] for pn, prm in module.named_parameters(recurse=False) if prm.requires_grad is False]
            for pn, prm in ret:
                delattr(module, pn)
                module.register_buffer(pn, prm.data.clone())
            for m in module.children():
                convert_nograd_params_to_buffers(m)

        for k, v in kwargs.items():
            if dist.is_initialized():
                v = torch.nn.SyncBatchNorm.convert_sync_batchnorm(v)
            convert_nograd_params_to_buffers(v)
            setattr(self, k, v)

        t_size = 0
        ddp_ignore = list()
        for n,ps in self.named_parameters():
            p_size = ps.numel()
            t_size += p_size
            if p_size == 0:
                ddp_ignore.append(n)
        self._ddp_params_and_buffers_to_ignore = ddp_ignore

        self.num_iterations = 0

    def forward(self, params):
        # self.num_iterations += 1
        # return next(self.parameters()) + x
        return torch.stack([p.mean() for p in params])

def send_model_to_device(model_or_dict, device, check_interval=1000):
    from contextlib import nullcontext, contextmanager

    mdc = None  # multi device context
    mpd = None  # main process decorator
    ret = model_or_dict

    if dist.is_initialized():

        if isinstance(model_or_dict, (dict,)):
            whole_model = DDPWrapper(**model_or_dict)

            whole_model.to(device)

            t_size = 0
            ddp_ignore = list()
            for n, ps in whole_model.named_parameters():
                p_size = ps.numel()
                t_size += p_size
                if p_size == 0:
                    ddp_ignore.append(n)

            whole_model._ddp_params_and_buffers_to_ignore = ddp_ignore

            # whole_model_ddp = DDP(whole_model, device_ids=[device.index], bucket_cap_mb=128, find_unused_parameters=True)
            # whole_model = whole_model_ddp.module

            sync_ddp()

            for n, p in whole_model.named_parameters():
                if p.is_contiguous() is False:
                    print(f"{n} is not contiguous")
            for n, p in whole_model.named_buffers():
                if p.is_contiguous() is False:
                    print(f"{n} is not contiguous")

            for n, p in whole_model.named_parameters():
                if p not in ddp_ignore:
                    broadcast_to_all(p, source=0)
            for n, b in whole_model.named_buffers():
                if b not in ddp_ignore:
                    broadcast_to_all(b, source=0)
            
            sync_ddp()

            sync_check(whole_model, check_buffer=True)
            sync_ddp()

            num_iterations = 0
            @contextmanager
            def ddp_forward(params):
                # whole_model_ddp(params)
                nonlocal num_iterations
                if check_interval is not None and ( num_iterations ) % check_interval == 0:
                    sync_check(whole_model)
                num_iterations += 1
                yield
                return
                # try:
                    # # whole_model_ddp(0)
                    # print("fake forward")
                    # # yield whole_model_ddp
                    # yield None
                    # print("after yield")
                    # nonlocal num_iterations
                    # if (num_iterations ) % 10 == 0:
                    #     sync_check(whole_model)
                    # num_iterations += 1
                    # return
                # except Exception as ex:
                    # import traceback
                    # print(traceback.format_exc())
            
            rank = dist.get_rank()

            if rank in [-1, 0]:
                def ddp_zeroonly(func):
                    return func
            else:
                def ddp_zeroonly(func):
                    def noop(*args, **kwargs):
                        return None
                    return noop

            mdc = ddp_forward
            mpd = ddp_zeroonly
            ret = {k:getattr(whole_model, k) for k in model_or_dict}
    else:
        mdc = nullcontext
        mpd = (lambda x:x)
        if isinstance(model_or_dict, (dict,)):
            ret = {k:v.to(device) for k,v in model_or_dict.items()}
    return mdc, mpd, ret

def sync_ddp():
    if dist.is_initialized():
        dist.barrier()

@torch.no_grad()
def broadcast_to_all(value, source=0):
    dist.broadcast(value, src=source)
    return value

# name2op = {
#     "mean": dist.ReduceOp.AVG,
#     "sum":  dist.ReduceOp.SUM
# }
def reduce_to_one(value, reduction="mean", 
    name2op={"mean": dist.ReduceOp.AVG, "sum": dist.ReduceOp.SUM}, target="all"):
    reduce_op = name2op[reduction]
    if target == "all":
        async_worker = dist.all_reduce(value, op=reduce_op)
    elif isinstance(target, (int,)):
        async_worker = dist.reduce(value, target, op=reduce_op)
    return value

@torch.no_grad()
def deepspeed_param_setting(params, zero="0"):

    total_params = params

    if zero != "0" and dist.is_initialized():

        # for k, v in model_dict:
            # total_params = list(model.parameters())

        total_p = np.sum([p.numel() for p in total_params])
        local_p = 0
        local_i = 0
        rank_ps = [list() for _ in range(get_world_size())]
        for p in total_params:
            rank_ps[local_i].append(p)
            local_p += p.numel()

            if local_p > total_p * (local_i+1)/len(rank_ps):
                local_i += 1
    else:
        # for k, v in model_dict:
        #     total_params = list(model.parameters())
        #     rank_ps = [set(total_params) for _ in range(get_world_size())]

        #     settings[k] = rank_ps
        rank_ps = [list(total_params) for _ in range(get_world_size())]
    
    rank_ps = [list(ps) for ps in rank_ps]
    
    return rank_ps

@torch.no_grad()
def broadcast_optimizer(optimizer, rank_ps=None, set_to_none=False):
    params  = []
    for pg in optimizer.param_groups:
        for p in pg["params"]:
            if p.grad is not None:
                params.append(p)

    if dist.is_initialized():
        workers = []
        
        if rank_ps is not None \
            and len(rank_ps[0]) != len(set(rank_ps[0]+rank_ps[-1])):

            world_size = get_world_size()

            active_ps = []
            for i in range(world_size):
                active_ps.append([p for p in rank_ps[i] if p.grad is not None])

            total_bucket = [None]*world_size
            for i in range(world_size):
                if len(active_ps[i]) > 0:
                    # total = torch.cat([p.grad.to(memory_format=torch.contiguous_format).flatten() for p in active_ps[i]])
                    total = torch.cat([p.grad.flatten() for p in active_ps[i]])
                    # local_print(f"[{i}] {len(active_ps[i])} {total.shape}")
                    workers.append(dist.reduce(total, i, op=dist.ReduceOp.AVG, async_op=True))
                    total_bucket[i] = total

            # for w in workers:
            #     w.wait()

            for i in range(world_size):
                if len(active_ps[i]) > 0:
                    workers.pop(0).wait()
                    # for p, grad in zip(params, torch.split(total_bucket[i], [np.prod(p.shape, dtype=np.int64, initial=1) for p in params if p.grad is not None])):
                    for p, grad in zip(active_ps[i], torch.split(total_bucket[i], [np.prod(p.shape, dtype=np.int64, initial=1) for p in active_ps[i]])):
                        p.grad = grad.reshape(p.grad.shape)

                # for p in rank_ps[i]:
                #     # local_print(i, p.shape)
                #     if p.grad is not None:
                #         workers.append(dist.reduce(p.grad, i, op=dist.ReduceOp.AVG, async_op=True))
        else:
            total = torch.cat([p.grad.to(memory_format=torch.contiguous_format).flatten() for p in params])
            dist.all_reduce(total, op=dist.ReduceOp.AVG, async_op=False)
            # workers.append(dist.all_reduce(total, op=dist.ReduceOp.AVG, async_op=True)) # the default behavoir of DDP is average

            for p, grad in zip(params, torch.split(total, [np.prod(p.shape, dtype=np.int64, initial=1) for p in params])):
                p.grad = grad.view(p.grad.shape)

            # for p in params:
            #     workers.append(dist.all_reduce(p.grad, op=dist.ReduceOp.AVG, async_op=True))
        
        for w in workers:
            w.wait()
        # dist.barrier()

    for p in params:
        grad = p.grad
        # if p.dtype != grad.dtype or p.device != grad.device or p.layout != grad.layout or p.stride() != grad.stride():
        if p.stride() != grad.stride():

            # p_fmts = [p.is_contiguous(memory_format=f) for f in [torch.contiguous_format, torch.channels_last]]
            # g_fmts = [grad.is_contiguous(memory_format=f) for f in [torch.contiguous_format, torch.channels_last]]

            # print(f"param {p.shape} {p.dtype} {p.device} {p.layout} {p.stride()} {p_fmts}")
            # print(f"grad  {grad.shape} {grad.dtype} {grad.device} {grad.layout} {grad.stride()} {g_fmts}")
        
            if p.is_contiguous(memory_format=torch.channels_last):
                p.grad = grad.to(dtype=p.dtype, memory_format=torch.channels_last)
            else:
                p.grad = grad.to(dtype=p.dtype)
            # print(f"p.grad  {p.grad.shape} {p.grad.dtype} {p.grad.device} {p.grad.layout} {p.grad.stride()}")

@torch.no_grad()
def broadcast_parameter(rank_ps=None):
    # print(len(rank_ps[0]), len(set(rank_ps[0]+rank_ps[-1])))

    if dist.is_initialized() \
        and rank_ps is not None \
        and len(rank_ps[0]) != len(set(rank_ps[0]+rank_ps[-1])):

        workers = []

        world_size = get_world_size()

        total_bucket = []
        for i in range(world_size):
            total = torch.cat([p.data.to(memory_format=torch.contiguous_format).flatten() for p in rank_ps[i]])
            workers.append(dist.broadcast(total, i, async_op=True))
            total_bucket.append(total)

        for i in range(world_size):
            workers[i].wait()

            for p, data in zip(rank_ps[i], torch.split(total_bucket[i], [np.prod(p.shape, dtype=np.int64, initial=1) for p in rank_ps[i]])):
                p.data.copy_(data.reshape(p.shape))

            # for p in rank_ps[i]:
            #     workers.append(dist.broadcast(p.data, i, async_op=True))

        for w in workers:
            w.wait()

def gather(value, target="all"):
    if target == "all":
        if isinstance(value, torch.Tensor):
            if value.is_cuda:
                value_o = torch.cat([value]*dist.get_world_size())
                dist.all_gather_into_tensor(value_o, value)
                ret = value_o
            else:
                tensor_list = [torch.zeros_like(value) for _ in range(dist.get_world_size())]
                dist.all_gather(tensor_list, value)
                return torch.cat(tensor_list)
        else:
            output = [None for _ in range(dist.get_world_size())]
            dist.all_gather_object(output, value)
            return output
    elif isinstance(target, int):
        if isinstance(value, torch.Tensor):
            gather_list = [torch.empty_like(value) for _ in dist.get_world_size()]
            dist.gather(value, gather_list, target)
            ret = torch.cat(gather_list)
        else:
            gather_list = [None for _ in range(dist.get_world_size())]
            dist.gather_object(value, gather_list, target)
            ret = gather_list
    # dist.barrier() 
    return ret

def sync_check(model, check_buffer=False):
    # print("ddp sync checking")

    if not dist.is_initialized():
        return
    
    sync_ddp()

    with torch.no_grad():
        for name, param in model.named_parameters():
            w = param.data.to(memory_format=torch.contiguous_format)

            tensor_list = [torch.zeros_like(w) for i in range(dist.get_world_size())]
            dist.all_gather(tensor_list, w)

            nl, nr = 0, 1
            tl, tr = tensor_list[nl], tensor_list[nr]
            close = torch.allclose(tl, tr)
            if not close:
                diff = (tl - tr).abs().mean()
                print(f"!!! Parameter Diff between DDP nodes {nl}/{nr} !!! Name:{name} Diff:{diff.item()}")

        if check_buffer is True:
            for name, param in model.named_buffers():
                w = param.data.to(memory_format=torch.contiguous_format)

                tensor_list = [torch.zeros_like(w) for i in range(dist.get_world_size())]
                dist.all_gather(tensor_list, w)

                nl, nr = 0, 1
                tl, tr = tensor_list[nl], tensor_list[nr]
                close = torch.allclose(tl, tr)
                if not close:
                    diff = (tl - tr).abs().mean()
                    print(f"!!! Buffer Diff between DDP nodes {nl}/{nr} !!! Name:{name} Diff:{diff.item()}")