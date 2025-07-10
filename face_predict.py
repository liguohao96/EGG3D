import os
import torch
import torch.nn as nn
import torch.nn.functional as F

import os, sys
import cv2
import numpy as np
from time import time
from scipy.io import savemat
import argparse
from tqdm import tqdm
import torch

from easydict import EasyDict

ROOT = os.path.abspath(os.path.dirname(__file__))

ArcF_ROOT = os.path.join(ROOT, "3rdparty", "insightface", "recognition", "arcface_torch")
D3FR_ROOT = os.path.join(ROOT, "3rdparty", "Deep3DFaceRecon_pytorch")
DECA_ROOT = os.path.join(ROOT, "3rdparty", "DECA")

sys.path.insert(0, ArcF_ROOT)
from backbones import get_model
sys.path.pop(0)

sys.path.insert(0, D3FR_ROOT)
from models.facenet import ReconNetWrapper
# from util import util 
# import importlib
# importlib.invalidate_caches()
# d3fr_models = importlib.import_module("models")
# importlib.reload(d3fr_models)
# print(d3fr_models)
# d3fr_mod = importlib.import_module("models.facerecon_model")
# d3fr_model_cls = d3fr_mod.FaceReconModel
sys.path.pop(0)

sys.path.insert(0, DECA_ROOT)
from decalib.deca import DECA
from decalib.utils.config import cfg as deca_cfg
sys.path.pop(0)

def get_arcface_predictor(device="cpu", net_arch="r18", align=False):
    # run arcface
    net_name = net_arch.split("_")[0]
    net = get_model(net_name, dropout=0, fp16=False)
    state_dict = torch.load(os.path.join(ArcF_ROOT, f"{net_arch}_backbone.pth"), map_location="cpu")
    net.load_state_dict(state_dict)

    count_state = lambda sd: np.sum([v.numel() for k, v in sd.items()])

    total = count_state(net.state_dict())
    load  = count_state(state_dict)
    print(f"{net_arch} total:{total//2**20}MB loaded:{load//2**20}MB, ({load/total*100:.2f})%")

    import face_alignment

    if not hasattr(face_alignment.LandmarksType, "_2D"):
        face_alignment.LandmarksType._2D = face_alignment.LandmarksType.TWO_D

    class FAN(object):
        def __init__(self):
            import face_alignment
            self.model = face_alignment.FaceAlignment(face_alignment.LandmarksType._2D, flip_input=False)

        def run(self, image):
            '''
            image: 0-255, uint8, rgb, [h, w, 3]
            return: detected box list
            '''
            out = self.model.get_landmarks(image)
            return out

    class ArcFace_Predictor(nn.Module):
        def __init__(self, model, align, debug=True):
            super().__init__()

            self.model = model

            self.align = align

            src = np.array([
                [30.2946, 51.6963],
                [65.5318, 51.5014],
                [48.0252, 71.7366],
                [33.5493, 92.3655],
                [62.7299, 92.2041]], dtype=np.float32)
            src[:, 0] += 8.0
            self.src = src

            self.fan = FAN()

            self.debug       = debug
            self.debug_cache = {}
        
        def forward(self, image01_bhwc, lmk01_bk2=None):
            image_bchw = image01_bhwc.permute(0, 3, 1, 2)

            if self.align is True:
                from skimage import transform as trans
                BS = image01_bhwc.size(0)
                image_bchw = []
                for bi in range(BS):

                    rimg = image01_bhwc[bi].detach().cpu().numpy()

                    if lmk01_bk2 is not None:
                        landmark = lmk01_bk2[bi].detach().cpu().numpy()
                        landmark[:, 0] *= image01_bhwc.size(2) # x = x*W
                        landmark[:, 1] *= image01_bhwc.size(1) # y = y*H
                    else:
                        out = self.fan.run((255*rimg).astype(np.uint8))
                        landmark = out[0].squeeze()

                    landmark5 = np.zeros((5, 2), dtype=np.float32)
                    landmark5[0] = (landmark[36] + landmark[39]) / 2
                    landmark5[1] = (landmark[42] + landmark[45]) / 2
                    landmark5[2] = landmark[30]
                    landmark5[3] = landmark[48]
                    landmark5[4] = landmark[54]

                    tform = trans.SimilarityTransform()
                    tform.estimate(landmark5, self.src)
                    M = tform.params[0:2, :]
                    img = cv2.warpAffine(rimg,
                                        M, (112, 112),
                                        borderValue=0.0)
                    img = np.transpose(img, (2, 0, 1))  # 3*112*112, RGB

                    image_bchw.append(img)
                image_bchw = torch.as_tensor(image_bchw).to(device)
            else:
                image_bchw = F.interpolate(image_bchw, (112, 112), mode="bilinear", antialias=True)

            if self.debug:
                self.debug_cache["input_image"] = image_bchw.detach().clone().cpu()
            image_bchw = image_bchw.sub_(0.5).div_(0.5)
            return F.normalize(self.model(image_bchw), dim=-1, p=2)
    
    return ArcFace_Predictor(net, align).eval().to(device)

def get_d3fr_predictor(device="cpu", align=False):
    # run Deep3DFace
    # opt = EasyDict()
    # opt.model = "facerecon"

    # model = d3fr_model_cls(opt)
    # print(model.__class__)
    model = ReconNetWrapper(net_recon='resnet50', use_last_fc=False, init_path=None)

    load_path = os.path.join(ROOT, "dataset_preprocessing/ffhq/Deep3DFaceRecon_pytorch/checkpoints/pretrained", "epoch_20.pth")
    state_dict = torch.load(load_path, map_location="cpu")
    print('loading the model from %s' % load_path)
    model.load_state_dict(state_dict['net_recon'])
    model = model.eval().to(device)

    import face_alignment

    if not hasattr(face_alignment.LandmarksType, "_2D"):
        face_alignment.LandmarksType._2D = face_alignment.LandmarksType.TWO_D

    class FAN(object):
        def __init__(self):
            import face_alignment
            self.model = face_alignment.FaceAlignment(face_alignment.LandmarksType._2D, flip_input=False)

        def run(self, image):
            '''
            image: 0-255, uint8, rgb, [h, w, 3]
            return: detected box list
            '''
            out = self.model.get_landmarks(image)
            if out is None:
                return [0], 'kpt68'
            else:
                kpt = out[0].squeeze()
                left = np.min(kpt[:,0]); right = np.max(kpt[:,0]); 
                top = np.min(kpt[:,1]); bottom = np.max(kpt[:,1])
                bbox = [left,top, right, bottom]
                return bbox, 'kpt68'
    
    import torchvision
    from skimage.transform import estimate_transform, warp, resize, rescale
    class D3FR_Predictor(nn.Module):
        def __init__(self, d3fr, align, debug=True):
            super().__init__()

            crop_size = 224
            scale     = 1.5
            iscrop    = align
            face_detector = "fan"

            self.crop_size = crop_size
            self.scale = scale
            self.iscrop = iscrop
            self.resolution_inp = crop_size
            if face_detector == 'fan':
                self.face_detector = FAN()

            self.d3fr = d3fr

            self.debug       = debug
            self.debug_cache = {}

        def bbox2point(self, left, right, top, bottom, type='bbox'):
            ''' bbox from detector and landmarks are different
            '''
            if type=='kpt68':
                old_size = (right - left + bottom - top)/2*1.1
                center = np.array([right - (right - left) / 2.0, bottom - (bottom - top) / 2.0 ])
            elif type=='bbox':
                old_size = (right - left + bottom - top)/2
                center = np.array([right - (right - left) / 2.0, bottom - (bottom - top) / 2.0  + old_size*0.12])
            else:
                raise NotImplementedError
            return old_size, center
        
        def forward(self, image_bhwc):
            image_bchw = image_bhwc.permute(0, 3, 1, 2)

            BS = image_bchw.size(0)

            if self.iscrop:
                image_feed = []
                for bi in range(BS):
                    # image = torchvision.transforms.functional.to_pil_image(image_bchw[bi].detach().cpu())
                    image = (255*image_bhwc[bi]).detach().cpu().to(torch.uint8).numpy()

                    bbox, bbox_type = self.face_detector.run(image)
                    if len(bbox) < 4:
                        print('no face detected! run original image')
                        left = 0; right = h-1; top=0; bottom=w-1
                    else:
                        left = bbox[0]; right=bbox[2]
                        top = bbox[1]; bottom=bbox[3]
                    old_size, center = self.bbox2point(left, right, top, bottom, type=bbox_type)
                    size = int(old_size*self.scale)
                    src_pts = np.array([[center[0]-size/2, center[1]-size/2], [center[0] - size/2, center[1]+size/2], [center[0]+size/2, center[1]-size/2]])

                    DST_PTS = np.array([[0,0], [0,self.resolution_inp - 1], [self.resolution_inp - 1, 0]])
                    tform = estimate_transform('similarity', src_pts, DST_PTS)
                
                    image = image/255.

                    dst_image = warp(image, tform.inverse, output_shape=(self.resolution_inp, self.resolution_inp))
                    dst_image = dst_image.transpose(2,0,1)

                    image_feed.append(dst_image)
                
                image_bchw = torch.as_tensor(np.stack(image_feed, axis=0)).float().to(image_bchw.device)
            else:
                image_bchw = F.interpolate(image_bchw, (self.resolution_inp, self.resolution_inp), mode="bilinear", antialias=True)

            if self.debug:
                self.debug_cache["input_image"] = image_bchw.detach().clone().cpu()

            coeffs = self.d3fr(image_bchw)

            id_coeffs = coeffs[:, :80]
            exp_coeffs = coeffs[:, 80: 144]
            tex_coeffs = coeffs[:, 144: 224]
            angles = coeffs[:, 224: 227]
            gammas = coeffs[:, 227: 254]
            translations = coeffs[:, 254:]

            return {
                'shape': id_coeffs,
                'exp':   exp_coeffs,
                'tex':   tex_coeffs,
                'pose':  angles,
                'light': gammas.unflatten(-1, (-1, 3)),
                'trans': translations
            }

            # pred_coeffs = {k:v for k, v in self.d3fr.pred_coeffs_dict.items()}
            # pred_lm = self.pred_lm.cpu().numpy()
            # pred_lm = np.stack([pred_lm[:,:,0],self.input_img.shape[2]-1-pred_lm[:,:,1]],axis=2) # transfer to image coordinate
            # pred_coeffs['lm68'] = pred_lm
            # return pred_coeffs
    
    return D3FR_Predictor(model, align).eval().to(device)

def get_deca_predictor(device="cpu", align=False):
    # run DECA
    deca_cfg.model.use_tex     = False
    deca_cfg.rasterizer_type   = "pytorch3d"
    deca_cfg.model.extract_tex = False
    deca = DECA(config=deca_cfg, device=device)

    import face_alignment

    if not hasattr(face_alignment.LandmarksType, "_2D"):
        face_alignment.LandmarksType._2D = face_alignment.LandmarksType.TWO_D

    class FAN(object):
        def __init__(self):
            import face_alignment
            self.model = face_alignment.FaceAlignment(face_alignment.LandmarksType._2D, flip_input=False)

        def run(self, image):
            '''
            image: 0-255, uint8, rgb, [h, w, 3]
            return: detected box list
            '''
            out = self.model.get_landmarks(image)
            if out is None:
                return [0], 'kpt68'
            else:
                kpt = out[0].squeeze()
                left = np.min(kpt[:,0]); right = np.max(kpt[:,0]); 
                top = np.min(kpt[:,1]); bottom = np.max(kpt[:,1])
                bbox = [left,top, right, bottom]
                return bbox, 'kpt68'
    
    import torchvision
    from skimage.transform import estimate_transform, warp, resize, rescale
    class DECA_Predictor(nn.Module):
        def __init__(self, deca, align, debug=True):
            super().__init__()

            crop_size = 224
            scale     = 1.5
            iscrop    = align
            face_detector = "fan"

            self.crop_size = crop_size
            self.scale = scale
            self.iscrop = iscrop
            self.resolution_inp = crop_size
            if face_detector == 'fan':
                self.face_detector = FAN()

            self.deca = deca

            self.debug       = debug
            self.debug_cache = {}

        def bbox2point(self, left, right, top, bottom, type='bbox'):
            ''' bbox from detector and landmarks are different
            '''
            if type=='kpt68':
                old_size = (right - left + bottom - top)/2*1.1
                center = np.array([right - (right - left) / 2.0, bottom - (bottom - top) / 2.0 ])
            elif type=='bbox':
                old_size = (right - left + bottom - top)/2
                center = np.array([right - (right - left) / 2.0, bottom - (bottom - top) / 2.0  + old_size*0.12])
            else:
                raise NotImplementedError
            return old_size, center
        
        def forward(self, image_bhwc):
            image_bchw = image_bhwc.permute(0, 3, 1, 2)

            BS = image_bchw.size(0)

            if self.iscrop:
                image_feed = []
                for bi in range(BS):
                    # image = torchvision.transforms.functional.to_pil_image(image_bchw[bi].detach().cpu())
                    image = (255*image_bhwc[bi]).detach().cpu().to(torch.uint8).numpy()

                    bbox, bbox_type = self.face_detector.run(image)
                    if len(bbox) < 4:
                        print('no face detected! run original image')
                        left = 0; right = h-1; top=0; bottom=w-1
                    else:
                        left = bbox[0]; right=bbox[2]
                        top = bbox[1]; bottom=bbox[3]
                    old_size, center = self.bbox2point(left, right, top, bottom, type=bbox_type)
                    size = int(old_size*self.scale)
                    src_pts = np.array([[center[0]-size/2, center[1]-size/2], [center[0] - size/2, center[1]+size/2], [center[0]+size/2, center[1]-size/2]])

                    DST_PTS = np.array([[0,0], [0,self.resolution_inp - 1], [self.resolution_inp - 1, 0]])
                    tform = estimate_transform('similarity', src_pts, DST_PTS)
                
                    image = image/255.

                    dst_image = warp(image, tform.inverse, output_shape=(self.resolution_inp, self.resolution_inp))
                    dst_image = dst_image.transpose(2,0,1)

                    image_feed.append(dst_image)
                
                image_bchw = torch.as_tensor(np.stack(image_feed, axis=0)).float().to(image_bchw.device)
            else:
                image_bchw = F.interpolate(image_bchw, (self.resolution_inp, self.resolution_inp), mode="bilinear", antialias=True)

            if self.debug:
                self.debug_cache["input_image"] = image_bchw.detach().clone().cpu()
            return self.deca.encode(image_bchw)
    
    return DECA_Predictor(deca, align).eval().to(device)

def get_emoca_model():
    pass