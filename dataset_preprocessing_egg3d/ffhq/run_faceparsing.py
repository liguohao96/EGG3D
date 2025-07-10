import os
import sys
import json
import torch
import torch.nn.functional as F
import numpy as np
import imageio
from PIL import Image, ImageOps

from glob import glob
from tqdm import tqdm
from natsort import natsorted
from functools import partial
from contextlib import nullcontext

import torchvision

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
SAVE_ROOT = os.path.join(ROOT, "temp", "seg_test")

os.makedirs(SAVE_ROOT, exist_ok=True)

sys.path.insert(0, ROOT)
# import models.BiSeNet as FACEP_mod
# from utils.face_2d import DlibLandmark, MediaPipeLandmark, GMMSkinSegmentation, PTNetRun
sys.path.pop(0)

def resize_2d(img_bhwc, H, W):
    b, h, w, c = img_bhwc.shape

    if (h, w) != (H, W):
        img_bhwc = F.interpolate(img_bhwc.permute(0, 3, 1, 2), (H, W)).permute(0, 2, 3, 1)
    return img_bhwc

def process_fn(dev_id, args, in_value, out_queue):
    import torch
    device = torch.device("cuda", dev_id)
    torch.cuda.set_device(device)
    with nullcontext("facer"):
        tform = torchvision.transforms.Compose([
            torchvision.transforms.Resize(448),
        ])

        annot_name = "celebm"
        # celebm/448
        annot = ['background', 'neck', 'face', 'cloth', 'r_ear', 
                 'l_ear', 'r_brow', 'l_brow', 'r_eye', 'l_eye', 
                 'nose',  'i_mouth', 'l_lip', 'u_lip', 'hair',
                 'eye_g', 'hat', 'ear_r', 'neck_l']

        # model = args.model
        model = torch.load(os.path.join(ROOT, "Data", "face_parsing.farl.celebm.main_ema_181500_jit.pt"))
        # print(model)
        model = model.eval().to(device)

        bg_cls   = ['background']
        head_cls = ['neck', 'face', 'r_ear', 'l_ear', 'r_brow', 'l_brow', 'r_eye', 'l_eye', 'nose', 'l_lip', 'u_lip']
        free_cls = ['cloth', 'i_mouth', 'hair', 'eye_g', 'hat', 'ear_r', 'neck_l']

        bg_inds   = np.array([annot.index(s) for s in   bg_cls])
        head_inds = np.array([annot.index(s) for s in head_cls])
        free_inds = np.array([annot.index(s) for s in free_cls])

        bg_clr   = np.array([0,   0,   0], dtype=np.uint8).reshape(1, 1, 3)  # black
        head_clr = np.array([0, 255,   0], dtype=np.uint8).reshape(1, 1, 3)  # green
        free_clr = np.array([0,   0, 255], dtype=np.uint8).reshape(1, 1, 3)  # blue

        f_bg_clr   = np.array([0,   0,   0], dtype=np.float32).reshape(1, 1, 3)  # black
        f_head_clr = np.array([0, 1.0,   0], dtype=np.float32).reshape(1, 1, 3)  # green
        f_free_clr = np.array([0,   0, 1.0], dtype=np.float32).reshape(1, 1, 3)  # blue

        def id2clr(id_img):
            bg_msk, head_msk, free_msk = map(lambda x:np.isin(id_img, x), (bg_inds, head_inds, free_inds))
            bg_alp, head_alp, free_alp = map(lambda x:x[:, :, None].astype(np.uint8), [bg_msk, head_msk, free_msk])
            color = bg_alp*bg_clr + head_alp*head_clr + free_alp*free_clr
            return color

        def pd2clr(pd_img):
            pd_img = pd_img.astype(np.float32)
            # pd_img = pd_img / (pd_img.sum(axis=0, keepaxis=True) - pd_img[bg_inds].sum(axis=0, keepaxis=True))
            pd_img = pd_img / (pd_img.sum(axis=0, keepdims=True))

            bg_alp   = pd_img[bg_inds].sum(axis=0)[:, :, None]
            head_alp = pd_img[head_inds].sum(axis=0)[:, :, None]
            free_alp = pd_img[free_inds].sum(axis=0)[:, :, None]

            color = bg_alp*f_bg_clr + head_alp*f_head_clr + free_alp*f_free_clr
            return (255*color.astype(np.float32)).astype(np.uint8)
    
    with nullcontext("TTA"):
        keep_origin = torchvision.transforms.Compose([
            torchvision.transforms.ToTensor(),
        ])
        gray_scale  = torchvision.transforms.Compose([
            torchvision.transforms.Grayscale(num_output_channels=3),
            torchvision.transforms.ToTensor(),
        ])
        colorj_l = [torchvision.transforms.Compose([
            torchvision.transforms.ColorJitter(
                brightness=0.4, 
                contrast=0.2,
                saturation=0.4, 
                hue=0.3),
            torchvision.transforms.ToTensor(),
        ]) for _ in range(3)]
        
        tta_callables = [keep_origin, gray_scale] + colorj_l
        
        def get_tta_sample(im: Image):
            image_list = []
            for aug in tta_callables:
                image_list.append(aug(im))
            
            def aggr_fn(logit):
                assert logit.size(0) == len(tta_callables), f"#tta={len(tta_callables)}, #ret={len(logit)}"
                return logit.sum(dim=0)
            
            return torch.stack(image_list, dim=0), aggr_fn

    running = True
    while running:

        if args.num_gpus > 1:
            value = in_value.get()
            if value is None:
                break
        
            def item_finished(code):
                out_queue.put((value, code))
        else:
            def item_finished(code):
                if code == 0:
                    pass
                else:
                    print(code)
            try:
                value = next(in_value)
            except StopIteration as ex:
                return

        im_path = args.dataset[value]
        err = 0

        try:
            name,ext= os.path.splitext(im_path)

            pth_fpath      = os.path.join(args.outdir, f"{name}.pth")
            pth_fpath_flip = os.path.join(args.outdir, f"{name}_mirror.pth")

            if args.skip_if_exist and os.path.exists(pth_fpath):
                continue

            im_flip = f"{name}_mirror{ext}"

            with torch.no_grad():
                im      = Image.open(os.path.join(args.indir, im_path)).convert('RGB')
                im_flip = Image.open(os.path.join(args.indir, im_flip)).convert('RGB') if im_flip is not None else ImageOps.mirror(im)

                inpt_im_0, aggr_fn_0 = get_tta_sample(im)      # T,C,H,W   Callable
                inpt_im_1, aggr_fn_1 = get_tta_sample(im_flip)

                inpt_im = torch.cat([inpt_im_0, inpt_im_1], axis=0)  # 2*T,C,H,W

                data_in = tform(inpt_im).to(device)
                logits  = model(data_in)[0].softmax(dim=1)
                
                # print(logits.sum(1).min(), logits.sum(1).max())

                if logits.shape[2:] != inpt_im.shape[2:]:
                    logits = F.interpolate(logits, inpt_im.shape[2:], mode="bilinear", antialias=True, align_corners=False)
                
                logit_0, logit_1 = logits.unflatten(0, (2, -1))

                logit_im   = aggr_fn_0(logit_0) + torch.flip(aggr_fn_1(logit_1), dims=(-1,))  # C,H,W
                logit_im_f = torch.flip(aggr_fn_0(logit_0), dims=(-1,)) + aggr_fn_1(logit_1)

            pd_img, pd_img_flip = map(lambda x:x.detach().cpu().half(), (logit_im, logit_im_f)) # C, H, W

            with open(pth_fpath, "wb") as f:
                torch.save({f"prob_{annot_name}": pd_img}, f)
        except Exception as ex:
            import traceback
            err = traceback.format_exc()
        finally:
            # this will always execute, even if continue
            item_finished(err)

    if args.num_gpus:
        print(f"process {dev_id} quit")

def progress_fn(total, out_queue):
    succ = count = 0
    pbar = tqdm(total=total)
    while True:
        value = out_queue.get()
        if value is None:
            break

        im_path, state = value
        if state == 0:
            succ += 1
        else:
            print(f"error dealing with '{im_path}', '{state}'")
        count += 1

        pbar.update()

    pbar.close()

@torch.no_grad()
def main(args):
    # device = torch.device("cuda")
    num_gpus = torch.cuda.device_count()
    print(f"#GPU={num_gpus}")
    assert num_gpus > 0

    with open(os.path.join(args.indir, 'dataset.json'), "r") as f:
        labels = json.load(f)["labels"]

        fnames = [im_path for (im_path, _) in labels if '_mirror' not in im_path]
        fnames = sorted(list(set(fnames)))
    
    if args.start_index is not None:
        fnames = fnames[args.start_index:]
    if args.num_samples is not None:
        fnames = fnames[:args.num_samples]
    
    args.num_gpus = num_gpus
    args.dataset  = fnames
            
    os.makedirs(args.outdir, exist_ok=True)

    if args.num_gpus > 1:
        mp = torch.multiprocessing
        mp.set_start_method("spawn")

        job_q = mp.Queue(num_gpus*10)
        out_q = mp.Queue(1024)

        p = None
        p = mp.Process(target=progress_fn, args=(len(fnames), out_q))
        p.start()
        proc_list = [p]

        p = mp.spawn(process_fn, args=(args, job_q, out_q), nprocs=num_gpus, join=False)
        proc_list.append(p)

        # for i, im_path in fnames:
        for i in range(len(fnames)):
            job_q.put(i)

        for p in range(num_gpus):
            job_q.put(None)
        
        import time
        while True:
            empty = job_q.empty()
            if empty:
                break
            else:
                time.sleep(5)
        job_q.close()

        out_q.put(None)
        while True:
            empty = out_q.empty()
            if empty:
                break
            else:
                time.sleep(5)
        out_q.close()

        for p in proc_list:
            if p is not None:
                p.join()
    else:
        # for i in tqdm(range(len(fnames))):
        process_fn(0, args, iter(tqdm(range(len(fnames)))), None)

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()

    parser.add_argument('--skip_if_exist', type=lambda x:eval(x), default=True)
    parser.add_argument('--indir',  type=str, required=True)
    parser.add_argument('--outdir', type=str, required=True)

    parser.add_argument('--start_index', type=int, default=None)
    parser.add_argument('--num_samples', type=int, default=None)
    args = parser.parse_args()

    main(args)