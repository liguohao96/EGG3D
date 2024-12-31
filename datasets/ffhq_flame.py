import os
import numpy as np
import zipfile
import PIL.Image
import imageio
import json
import cv2
import pyspng
import torch
from collections import defaultdict

class FFHQFLAMEDataset(object):
    def __init__(self, ds_root, label_file, segment_root=None, zip_file=None, down_scale=1, image_type=None, augmentations="all", lazy_init=False, black_list=None):
        super().__init__()

        self._args_repr = f"ds_root='{ds_root}', label_file='{label_file}', segment_root='{segment_root}', down_scale={down_scale}, image_type={image_type}, augmentations={augmentations}"

        image_type = image_type if image_type is not None else "rgba"
        ret = []

        data_root  = ds_root
        param_file = label_file

        self.zip_file = None
        if zip_file is not None:
            self.zip_file = zipfile.ZipFile(zip_file)
            lazy_init     = True

        labels = torch.load(self._open_file(param_file))["labels"]

        if augmentations == "all":
            pass
        else:
            valid_sampleid = []
            valid_augments = augmentations.split(",")
            for i, (file_path, v) in enumerate(labels):
                # get_augment
                file_name = os.path.split(file_path)[-1] # ID_aug.png.png ID.png.png
                file_name = file_name.split(".")[0]      # ID_aug         ID
                splited   = file_name.split("_")
                aug_name  = "" if len(splited) == 1 else splited[-1]

                if len(aug_name) > 0 and aug_name not in valid_augments:
                    pass
                else:
                    valid_sampleid.append(i)
            
            labels = [labels[i] for i in valid_sampleid]

        if black_list is not None:
            with open(black_list, "r") as f:
                black_list = [l.rstrip() for l in f.readlines()]
            black_list = set(black_list)
            print(black_list)
            
            valid_id = []
            for i, (file_path, v) in enumerate(labels):
                if file_path in black_list:
                    print(f"delet {file_path}")
                    continue
                valid_id.append(i)

            labels = [labels[i] for i in valid_id]

        fpath_list  = []
        packed_dict = defaultdict(list)
        for indx, (file_path, data_dict) in enumerate(labels):
            fpath_list.append(file_path)
            for k, v in data_dict.items():
                packed_dict[k].append(v)

        self.mpf_fpaths = np.array(fpath_list)
        self.mpf_labels = {k:torch.stack(v) for k, v in packed_dict.items() if torch.is_tensor(v[0])}
        self.mpf_labels.update({k: np.array(v) for k, v in packed_dict.items() if not torch.is_tensor(v[0])})

        self.labels = labels

        self.data_root = data_root
        self.seg_root  = segment_root
        self.args      = (down_scale, image_type)

        if lazy_init is True:
            # close already opened
            if self.zip_file is not None:
                self.zip_file.close()

            self.zip_file = zip_file

    def _open_file(self, fname):
        if self.zip_file is not None:
            return self.zip_file.open(fname, "r")
        else:
            return open(fname, "rb")
    
    def __repr__(self):
        return f"FFHQFLAMEDataset({self._args_repr}) len={len(self)}"

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, index):
        if isinstance(self.zip_file, (str,)):
            print("lazy open zip file")
            self.zip_file = zipfile.ZipFile(self.zip_file)

        down_scale, image_type = self.args

        # file_path, v = self.labels[index]
        file_path = self.mpf_fpaths[index]

        img_path  = os.path.join(self.data_root, file_path)
        img_ext   = os.path.splitext(img_path)[-1]
        if len(img_ext) == 0:
            img_path += ".png"

        with self._open_file(img_path) as f:
            # image = np.array(PIL.Image.open(f)).astype(np.float32) / 255.
            image = pyspng.load(f.read())

            # assert image.dtype in [np.uint8], f"got error image type {image.dtype}"
            
            image = image.astype(np.float32) / 255.

        H, W, C   = image.shape

        if down_scale > 1:
            image = cv2.resize(image, None, fx=1/down_scale, fy=1/down_scale, interpolation=cv2.INTER_AREA)

        ret = {
            "img_path": img_path,
            "image":    image.astype(np.float32), 
            # "shape":    v["shape"].detach(),
            # "exp":      v["exp"].detach(),
            # "pose":     v["pose"].detach(),
            # "eye_pose": v["eye_pose"].detach(),
            # "tex":      v["tex"].detach(),
            # "light":    v["light"].detach(),
            # "m2v":      v["m2v"].detach(),
            # "ndc":      v["ndc"].detach(),
        }
        for k, v in self.mpf_labels.items():
            if torch.is_tensor(v):
                ret[k] = v[index].clone()
            else:
                ret[k] = v[index]

        if self.seg_root is not None:
            seg_path = os.path.join(self.seg_root,  file_path)

            with self._open_file(seg_path) as f:
                # segim = np.array(PIL.Image.open(f)).astype(np.float32) / 255.
                segim = pyspng.load(f.read()).astype(np.float32) / 255.
            if segim.shape[:2] != image.shape[:2]:
                segim = cv2.resize(segim, (image.shape[1], image.shape[0]), interpolation=cv2.INTER_AREA)
            
            ret["segim"] = segim.astype(np.float32)

            ret["glass_prob"] = np.array([segim[..., 0].mean() < 0.001]).astype(np.float32)
        return ret

    def close(self):
        try:
            if self.zip_file is not None:
                self.zip_file.close()
        finally:
            self.zip_file = None

class FFHQFLAMEDataset2(object):
    def __init__(self, ds_root, label_file, segment_root=None, zip_file=None, down_scale=1, image_type=None, augmentations="all", lazy_init=False, black_list=None):
        super().__init__()

        self._args_repr = f"ds_root='{ds_root}', label_file='{label_file}', segment_root='{segment_root}', down_scale={down_scale}, image_type={image_type}, augmentations={augmentations}"

        image_type = image_type if image_type is not None else "rgba"
        ret = []

        data_root  = ds_root
        param_file = label_file

        self.zip_file = None
        if zip_file is not None:
            self.zip_file = zipfile.ZipFile(zip_file)
            lazy_init     = True

        labels = torch.load(self._open_file(param_file))["labels"]

        if augmentations == "all":
            pass
        else:
            valid_sampleid = []
            valid_augments = augmentations.split(",")
            for i, (file_path, v) in enumerate(labels):
                # get_augment
                file_name = os.path.split(file_path)[-1] # ID_aug.png.png ID.png.png
                file_name = file_name.split(".")[0]      # ID_aug         ID
                splited   = file_name.split("_")
                aug_name  = "" if len(splited) == 1 else splited[-1]

                if len(aug_name) > 0 and aug_name not in valid_augments:
                    pass
                else:
                    valid_sampleid.append(i)
            
            labels = [labels[i] for i in valid_sampleid]

        if black_list is not None:
            with open(black_list, "r") as f:
                black_list = [l.rstrip() for l in f.readlines()]
            black_list = set(black_list)
            print(black_list)
            
            valid_id = []
            for i, (file_path, v) in enumerate(labels):
                if file_path in black_list:
                    print(f"delet {file_path}")
                    continue
                valid_id.append(i)

            labels = [labels[i] for i in valid_id]

        self.labels = labels

        self.data_root = data_root
        self.seg_root  = segment_root
        self.args      = (down_scale, image_type)

        if lazy_init is True:
            # close already opened
            if self.zip_file is not None:
                self.zip_file.close()

            self.zip_file = zip_file

        self.m2v_t3d_off = torch.as_tensor([0.0, 0.0, 0.05])
    
    def _open_file(self, fname):
        if self.zip_file is not None:
            return self.zip_file.open(fname, "r")
        else:
            return open(fname, "rb")
    
    def __repr__(self):
        return f"FFHQFLAMEDataset({self._args_repr}) len={len(self)}"

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, index):
        if isinstance(self.zip_file, (str,)):
            print("lazy open zip file")
            self.zip_file = zipfile.ZipFile(self.zip_file)

        down_scale, image_type = self.args

        file_path, v = self.labels[index]

        img_path  = os.path.join(self.data_root, file_path)
        img_ext   = os.path.splitext(img_path)[-1]
        if len(img_ext) == 0:
            img_path += ".png"

        with self._open_file(img_path) as f:
            # image = np.array(PIL.Image.open(f)).astype(np.float32) / 255.
            image = pyspng.load(f.read()).astype(np.float32) / 255.

        H, W, C   = image.shape

        if down_scale > 1:
            image = cv2.resize(image, None, fx=1/down_scale, fy=1/down_scale, interpolation=cv2.INTER_AREA)

        ret = {
            "img_path": img_path,
            "image":    image.astype(np.float32), 
            "shape":    v["shape"].detach(),
            "exp":      v["exp"].detach(),
            "pose":     v["pose"].detach(),
            "eye_pose": v["eye_pose"].detach(),
            "tex":      v["tex"].detach(),
            "light":    v["light"].detach(),
            "m2v":      v["m2v"].detach().clone(),
            "ndc":      v["ndc"].detach(),
        }

        ret["m2v"][:3, 3] += ret["m2v"][:3, :3] @ self.m2v_t3d_off

        if self.seg_root is not None:
            seg_path = os.path.join(self.seg_root,  file_path)

            with self._open_file(seg_path) as f:
                # segim = np.array(PIL.Image.open(f)).astype(np.float32) / 255.
                segim = pyspng.load(f.read()).astype(np.float32) / 255.
            if segim.shape[:2] != image.shape[:2]:
                segim = cv2.resize(segim, (image.shape[1], image.shape[0]), interpolation=cv2.INTER_AREA)
            
            ret["segim"] = segim.astype(np.float32)

            ret["glass_prob"] = np.array([segim[..., 0].mean() < 0.001]).astype(np.float32)
        return ret

    def close(self):
        try:
            if self.zip_file is not None:
                self.zip_file.close()
        finally:
            self.zip_file = None

if __name__ == "__main__":
    from tqdm import tqdm
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--ds_root",      type=str, default=None)
    parser.add_argument("--label_file",   type=str, default=None)
    parser.add_argument("--segment_root", type=str, default=None)
    parser.add_argument("--zip_file",     type=str, default=None)
    parser.add_argument("--black_list",   type=str, default=None)

    parser.add_argument("--num_workers",  type=int, default=0)
    parser.add_argument("--num_iter",     type=int, default=1000)

    args = parser.parse_args()
    
    dataset = FFHQFLAMEDataset(args.ds_root, args.label_file, args.segment_root, args.zip_file, black_list=args.black_list, lazy_init=True)

    print(dataset)

    for i in tqdm(range(len(dataset))):
        data = dataset[i]

        if i == 0:
            for k, v in data.items():
                print(k, type(v))
        
        if i >= 10:
            break

    dataset = FFHQFLAMEDataset(args.ds_root, args.label_file, args.segment_root, args.zip_file, lazy_init=True)
    train_loader  = torch.utils.data.DataLoader(dataset, batch_size=4, 
                                                shuffle=False, drop_last=False,
                                                num_workers=args.num_workers, pin_memory=False, persistent_workers=False)
    print("dataloader")
    for i, data in enumerate(tqdm(train_loader)):
        data = dataset[i]

        if i == 0:
            for k, v in data.items():
                print(k, type(v))
        
        if i >= args.num_iter:
            break