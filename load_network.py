import os
import sys
import yaml

import torch

from collections import defaultdict, OrderedDict
from contextlib  import nullcontext

ROOT = os.path.abspath(os.path.join(__file__, ".."))

def get_class(type_name):
    import importlib
    mod_name = ".".join(type_name.split(".")[:-1])
    obj_name = type_name.split(".")[-1]

    sys.path.insert(0, ROOT)
    mod = importlib.import_module(mod_name)
    sys.path.pop(0)
    print(mod_name, mod)

    return getattr(mod, obj_name)

def load_GAN(gan_zip, config=None, verbose=True):

    if gan_zip.endswith(".zip"):
        import zipfile

        with zipfile.ZipFile(gan_zip) as zipf:
            with zipf.open("config.yaml") as f:
                gan_cfg = yaml.load(f, yaml.FullLoader)
        
                m_cfg = gan_cfg["model"]["G"]

            with zipf.open("param.pth") as f:
                state_dict = torch.load(f, map_location="cpu")
    elif gan_zip.endswith(".pth"):
        state_dict = torch.load(gan_zip, map_location="cpu")
        m_cfg      = config["model"]["G"]

    # init GAN
    with nullcontext("init GAN"):
        m_type   = m_cfg.get("type")
        m_args   = m_cfg.get("args",   list())
        m_kwargs = m_cfg.get("kwargs", dict())
        m_kwargs["fast_init"] = True

        mod = get_class(m_type)(*m_args, **m_kwargs)
        
    # load weight
    with nullcontext("load weight"):
        src_state   = state_dict["G_ema"]
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
                
                elif verbose:
                    print(f"[{k}] size mismatch {src_v.shape}->{tgt_v.shape}")

        missing, wrong = mod.load_state_dict(model_state, strict=False)
        print(f"load [G_ema] missig:{missing} wrong:{wrong} mismatch:{mismatch}")

    return mod