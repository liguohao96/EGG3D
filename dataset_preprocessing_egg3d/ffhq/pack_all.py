import os
import io
import sys
import json
import torch
import torch.nn.functional as F
import numpy as np
from PIL import Image, ImageOps

from glob import glob
from tqdm import tqdm
from natsort import natsorted
from functools import partial
from contextlib import nullcontext

import torchvision

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
SAVE_ROOT = os.path.join(ROOT, "temp", "fitting_debug")

os.makedirs(SAVE_ROOT, exist_ok=True)

sys.path.insert(0, ROOT)
# import models.BiSeNet as FACEP_mod
# from utils.face_2d import DlibLandmark, MediaPipeLandmark, GMMSkinSegmentation, PTNetRun
sys.path.pop(0)

@torch.no_grad()
def main(args):
    # device = torch.device("cuda")
    num_gpus = torch.cuda.device_count()
    print(f"#GPU={num_gpus}")
    assert num_gpus > 0

    with open(os.path.join(args.indir, 'dataset.json'), "r") as f:
        labels = json.load(f)["labels"]

        fnames = [im_path for (im_path, _) in labels]
        # fnames = [im_path for (im_path, _) in labels if '_mirror' not in im_path]
        fnames = sorted(list(set(fnames)))
    
    if args.start_index is not None:
        fnames = fnames[args.start_index:]
    if args.num_samples is not None:
        fnames = fnames[:args.num_samples]
    
    args.num_gpus = num_gpus
    args.dataset  = fnames
            
    # os.makedirs(args.fitdir, exist_ok=True)
    # os.makedirs(args.segdir, exist_ok=True)
    outext = os.path.splitext(args.outfile)[-1]

    if outext == ".zip":
        import zipfile
        zf = zipfile.ZipFile(file=args.outfile, mode="w", compression=zipfile.ZIP_STORED)

    labels = []

    for i, fname in enumerate(tqdm(fnames)):
        name,ext = os.path.splitext(fname)
        img_path = os.path.join(args.indir,  f"{name}.png")
        seg_path = os.path.join(args.segdir, f"{name}.png")
        fit_path = os.path.join(args.fitdir, f"{name}.pth")

        try:
            Image.open(img_path).verify()
            Image.open(seg_path).verify()
            img = Image.open(img_path)    # must reopen after Image.verify()
            seg = Image.open(seg_path)    # must reopen after Image.verify()
            data = torch.load(fit_path)

            label = data
            label["pose"] = torch.cat((torch.zeros_like(data["jaw_pose"]), data["jaw_pose"]), dim=-1)
            label["m2v"]  = data["m2v_matrix"]
            label["ndc"]  = data["ndc_matrix"]

            del label["m2v_matrix"]
            del label["ndc_matrix"]

            if outext == ".zip":

                img_bits = io.BytesIO()
                seg_bits = io.BytesIO()
                img.save(img_bits, format='png', compress_level=0, optimize=False)
                seg.save(seg_bits, format='png', compress_level=0, optimize=False)

                zf.writestr(os.path.join("final_crops", f"{name}.png"), img_bits.getbuffer())
                zf.writestr(os.path.join("segment_img", f"{name}.png"), seg_bits.getbuffer())

                labels.append((f"{name}.png", label))
            else:
                labels.append((fname, label))
        except Exception:
            import traceback
            print(traceback.format_exc())
            print(f"failed with {fname}")        

    print(f"dataset #sample: {len(labels)}")

    if outext == ".zip":
        pth_bits = io.BytesIO()
        torch.save({"labels": labels}, pth_bits)
        zf.writestr(f"lmkfit_dataset.pth", pth_bits.getbuffer())
        zf.close()
    else:
        with open(args.outfile, "wb") as f:
            torch.save({"labels": labels}, f)

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument('--indir',   type=str, required=True)
    parser.add_argument('--outfile', type=str, required=True)

    parser.add_argument('--fitdir', type=str, required=True)
    parser.add_argument('--segdir', type=str, required=True)

    parser.add_argument('--start_index', type=int, default=None)
    parser.add_argument('--num_samples', type=int, default=None)
    args = parser.parse_args()

    main(args)