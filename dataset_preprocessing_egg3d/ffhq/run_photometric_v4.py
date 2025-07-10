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
SAVE_ROOT = os.path.join(ROOT, "temp", "fitting_v4_debug")

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

def parse_fp_dict(fp_dict):
    name2label = {
        # celebm/448
        "celebm" : ['background', 'neck', 'face', 'cloth', 'r_ear', 
                'l_ear', 'r_brow', 'l_brow', 'r_eye', 'l_eye', 
                'nose',  'i_mouth', 'l_lip', 'u_lip', 'hair',
                'eye_g', 'hat', 'ear_r', 'neck_l']
    }

    img = lbl = key = None
    for k in name2label:
        mask_k = f"prob_{k}"
        if mask_k in fp_dict:
            img = fp_dict[mask_k]
            lbl = name2label[k]
            key = k
    
    assert img is not None, print(fp_dict.keys())

    return img, key

def batch_rodrigues(rot_vecs, epsilon=1e-8, dtype=torch.float32):
    ''' Calculates the rotation matrices for a batch of rotation vectors
        Parameters
        ----------
        rot_vecs: torch.tensor Nx3
            array of N axis-angle vectors
        Returns
        -------
        R: torch.tensor Nx3x3
            The rotation matrices for the given axis-angle parameters
    '''

    batch_size = rot_vecs.shape[0]
    device = rot_vecs.device

    angle = torch.norm(rot_vecs + 1e-8, dim=1, keepdim=True)
    rot_dir = rot_vecs / angle

    cos = torch.unsqueeze(torch.cos(angle), dim=1)
    sin = torch.unsqueeze(torch.sin(angle), dim=1)

    # Bx1 arrays
    rx, ry, rz = torch.split(rot_dir, 1, dim=1)
    K = torch.zeros((batch_size, 3, 3), dtype=dtype, device=device)

    zeros = torch.zeros((batch_size, 1), dtype=dtype, device=device)
    K = torch.cat([zeros, -rz, ry, rz, zeros, -rx, -ry, rx, zeros], dim=1) \
        .view((batch_size, 3, 3))

    ident = torch.eye(3, dtype=dtype, device=device).unsqueeze(dim=0)
    rot_mat = ident + sin * K + (1 - cos) * torch.bmm(K, K)
    return rot_mat

def face_normal(ver, tri):
    BS  = ver.size(0)

    faces   = ver[:, tri.flatten().long(), :].reshape(BS, -1, 3, 3)
    ori_face_nrm = torch.cross(faces[:,:,1]-faces[:,:,0], faces[:,:,2]-faces[:,:,0], dim=-1)
    return F.normalize(ori_face_nrm, dim=-1)

def vertex_normal(ver, tri):
    BS  = ver.size(0)

    tri = tri.long()

    ori_face_nrm = face_normal(ver, tri)

    v_nrm = torch.zeros_like(ver)
    v_nrm.scatter_add_(1, tri[None, :, 0, None].expand(BS, -1, 3), ori_face_nrm)
    v_nrm.scatter_add_(1, tri[None, :, 1, None].expand(BS, -1, 3), ori_face_nrm)
    v_nrm.scatter_add_(1, tri[None, :, 2, None].expand(BS, -1, 3), ori_face_nrm)
    return F.normalize(v_nrm, dim=-1)

def render_nvdr(m2v, prj, ver, tri, uv, uv_tri, tex, H=512, W=512, FOV=90, locals={}):
    import nvdiffrast.torch as dr
    BS     = ver.size(0)
    device = ver.device
    drctx = locals.get("drctx", dr.RasterizeCudaContext(device=device))

    # face_nrm = torch.cross(ver[:,tri[:,1]]-ver[:,tri[:,0]], ver[:,tri[:,2]]-ver[:,tri[:,0]], dim=-1)

    v_nrm = vertex_normal(ver, tri)

    tri_   = tri.int().contiguous()

    m2v_matx = m2v
    ndc_proj = prj

    ver_  = F.pad(ver, (0, 1), "constant", 1)
    ver_v = ver_  @ m2v_matx.transpose(1, 2)
    ver_n = ver_v @ ndc_proj.transpose(1, 2)
    tex_  = tex.contiguous()
    pos   = ver_n.contiguous()
    rast, rast_db = dr.rasterize(drctx, pos, tri_, resolution=[H, W])
    if tex_.ndim == 4:
        uv     = uv.float().contiguous()
        uv_tri = uv_tri.int().contiguous()
        im_uv,uv_da = dr.interpolate(uv, rast, uv_tri, rast_db, diff_attrs="all")
        # im_uv,uv_da = dr.interpolate(uv, rast, uv_tri)
        im_ndr      = dr.texture(tex_, im_uv, uv_da=uv_da)
    else:
        im_ndr, _   = dr.interpolate(tex_, rast, tri_, rast_db)
    im_ndr = dr.antialias(im_ndr, rast, pos, tri_)

    rot_nrm   = F.normalize(v_nrm @ m2v_matx[:,:3,:3].transpose(1, 2), dim=-1)     # B,Nf,3
    im_nrm, _ = dr.interpolate(rot_nrm.contiguous(), rast, tri_, rast_db)

    z_view    = ver_v[...,2:3] - ver_v[...,2:3].min(-2, keepdim=True).values.detach()
    im_dep, _ = dr.interpolate(z_view.contiguous(), rast, tri_, rast_db)

    im_ndr = dr.antialias(im_ndr, rast, pos, tri_)
    im_nrm = F.normalize(im_nrm, dim=-1)

    im_alp = (rast[:,:,:,-1].reshape(BS,H,W,1)>=1).float()
    im_dep = im_alp * (im_dep.reshape(BS,H,W,1))

    return im_ndr, im_alp, im_nrm, im_dep

def shading_SH2(normals, sh_coeff, locals={}):
    '''
        normals:  [bs, N, 3]
        sh_coeff: [bs, 9, 3]
    '''
    pi = np.pi
    constant_factor = torch.tensor(
    [1 / np.sqrt(4 * pi), ((2 * pi) / 3) * (np.sqrt(3 / (4 * pi))), ((2 * pi) / 3) * (np.sqrt(3 / (4 * pi))), \
     ((2 * pi) / 3) * (np.sqrt(3 / (4 * pi))), (pi / 4) * (3) * (np.sqrt(5 / (12 * pi))),
     (pi / 4) * (3) * (np.sqrt(5 / (12 * pi))), \
     (pi / 4) * (3) * (np.sqrt(5 / (12 * pi))), (pi / 4) * (3 / 2) * (np.sqrt(5 / (12 * pi))),
     (pi / 4) * (1 / 2) * (np.sqrt(5 / (4 * pi)))]).float()
    constant_factor = locals.get("constant_factor", constant_factor.to(normals.device))

    x, y, z = normals.unbind(-1)
    sh = torch.stack([
        x * 0. + 1., x, y, \
        z,  x * y, x * z,
        y * z, x ** 2 - y ** 2, 3 * (z ** 2) - 1
    ],
        1)  # [bs, 9, N]
    sh = sh * constant_factor[None, :, None]   # [bs, 9, N]
    # shading = torch.sum(sh_coeff[:, :, :, None, None] * sh[:, :, None, :, :], 1)  # [bz, 9, 3, h, w]
    shading = torch.einsum("bkc,bkn->bnc", sh_coeff, sh)
    return shading

import cv2
class PhotometricRunner(object):
    def __init__(self, device, args):
        super().__init__()

        self.flip_cam_yaw = args.flip_cam_yaw

        self.fp_label_to_mask = {}
        self.fp_label_to_part = {}
        with nullcontext("faceparsing"):
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

            # ONLY USED FOR Photometric Loss Weighting
            bg_cls   = ['background']
            head_cls = ['face', 'r_ear', 'l_ear', 'r_brow', 'l_brow', 'nose', 'l_lip', 'u_lip']
            free_cls = ['neck', 'cloth', 'i_mouth', 'hair', 'eye_g', 'hat', 'ear_r', 'neck_l']
            eyes_cls = ['r_eye', 'l_eye']
            ears_cls = ['r_ear', 'l_ear']

            assert len(bg_cls+head_cls+free_cls+eyes_cls) == len(annot)

            bg_inds   = np.array([annot.index(s) for s in   bg_cls])
            head_inds = np.array([annot.index(s) for s in head_cls])
            free_inds = np.array([annot.index(s) for s in free_cls])
            eyes_inds = np.array([annot.index(s) for s in eyes_cls])

            self.fp_label_to_mask[annot_name] = (torch.as_tensor(bg_inds), torch.as_tensor(head_inds), torch.as_tensor(free_inds), torch.as_tensor(eyes_inds))

            self.fp_label_to_part[annot_name] = {
                "eye": torch.as_tensor([annot.index(s) for s in ['r_eye', 'l_eye']]),
                "ear": torch.as_tensor([annot.index(s) for s in ['r_ear', 'l_ear']]),
                "face": torch.as_tensor([annot.index(s) for s in ['face', 'r_brow', 'l_brow', 'nose', 'l_lip', 'u_lip']]),
            }

        with nullcontext("face_3d_rec"):
            model_root = os.path.join(ROOT)
            sys.path.insert(0, model_root)
            import importlib
            importlib.invalidate_caches()
            MODELS_mod = importlib.import_module("models")
            importlib.reload(MODELS_mod)
            FLAME_mod = importlib.import_module("models.FLAME")
            # FACEP_mod = importlib.import_module("models.BiSeNet")
            # ARCF_mod  = importlib.import_module("models.ArcFace")
            # from utils.face_2d import DlibLandmark, MediaPipeLandmark, GMMSkinSegmentation, PTNetRun
            sys.path.pop(0)

            FLAME    = FLAME_mod.FLAME
            FLAMETex = FLAME_mod.FLAMETex

            from easydict import EasyDict

            self.device = device

            self.config = EasyDict({
                "flame_model_path":        os.path.join(ROOT, "Data", "FLAME2020", "generic_model.pkl"),
                "flame_lmk_embedding_path":os.path.join(ROOT, "Data", "FLAME2020", "landmark_embedding.npy"),
                "flame_tex_path":          os.path.join(ROOT, "Data", "FLAME2020", "FLAME_texture.npz"),
                "tex_path":                os.path.join(ROOT, "Data", "FLAME2020", "FLAME_albedo_from_BFM.npz"),
                "mask_path":               os.path.join(ROOT, "Data", "FLAME_masks", "FLAME_masks.pkl"),
                "flame_mediapipe_lmk_embedding_path": os.path.join(ROOT, "Data", "FLAME2020", "mediapipe_landmark_embedding.npz"),
                "flame_gaze_lmk_embedding_path":      os.path.join(ROOT, "Data", "FLAME2020", "gaze_landmark_embedding2.npz"),
                "n_shape": 100,
                "n_exp":   50,
                "n_tex":   50,
                "tex_type": 'BFM',
            })

            self.flame      = FLAME(self.config).to(device)
            self.flametex   = FLAMETex(self.config).to(device)

            flame_shp_f = os.path.join(".", "Data", "FLAME2020", "generic_model.pkl")
            flame_tex_f = os.path.join(".", "Data", "FLAME2020", "FLAME_texture.npz")
            flame_tim_f = os.path.join(".", "Data", "FLAME2020", "tex_mean.png")
            with open(flame_shp_f, "rb") as f:
                import pickle
                flame_shp = pickle.load(f, encoding="latin1")
        
            ver = torch.as_tensor(flame_shp["v_template"].reshape(-1, 3)).float().to(device)
            tri = torch.as_tensor(flame_shp["f"].reshape(-1, 3).astype(np.int32)).long().to(device)

            # rootj = [1.3331e-03, -1.4789e-01, -8.2942e-02]
            joints = torch.einsum('ik,ji->jk', [self.flame.v_template, self.flame.J_regressor])
            print("joints", joints)
            # rootj[0] = 0
            # print("root joint", rootj[0], rootj[1], rootj[2])
            self.offset = torch.as_tensor([0.0, 0.00, 0.05]).to(device)
        
            flame_tex = np.load(flame_tex_f)
            uv     = torch.as_tensor(flame_tex["vt"]).float()
            uv_tri = torch.as_tensor(flame_tex["ft"].astype(np.int32)).long()
            uv[:, 1] = 1 - uv[:, 1]
            uv, uv_tri = uv.to(device), uv_tri.to(device)

            tex = torch.as_tensor(imageio.imread(flame_tim_f)).to(device)/255.

            self.uv     = uv
            self.uv_tri = uv_tri
            self.tex    = tex

            with nullcontext("segment texture map"), torch.no_grad():
                with open(self.config.mask_path, "rb") as f:
                    data = pickle.load(f, encoding="latin1")
                
                # 'eye_region', 'neck', 'left_eyeball', 'right_eyeball', 'right_ear', 'right_eye_region', 'forehead', 'lips', 'nose', 'scalp', 'boundary', 'face', 'left_ear', 'left_eye_region'

                v_mask_l = []
                key_conf = [["right_eyeball", "left_eyeball"], ["face", "forehead", "scalp"], ["right_ear", "left_ear"]]
                for key_list in key_conf:
                    v_mask = torch.zeros(ver.size(0), dtype=torch.int32)
                    for k in key_list:
                        v_mask.scatter_add_(0, torch.as_tensor(data[k]), torch.ones(ver.size(0), dtype=torch.int32))
                    
                    v_mask_l.append( (v_mask > 0) )

                v_mask = torch.stack(v_mask_l, dim=1)  # V, C
                print(v_mask.shape)

                # https://pytorch.org/vision/master/_modules/torchvision/utils.html#draw_segmentation_masks
                def _generate_color_palette(num_objects):
                    palette = torch.tensor([2**25 - 1, 2**15 - 1, 2**21 - 1])
                    return [tuple((i * palette) % 255) for i in range(num_objects)]

                # palette = torch.as_tensor(_generate_color_palette(v_mask.size(1)+1)).float()[1:] / 255.

                palette = torch.as_tensor([
                    [1, 0, 0],
                    [0, 1, 0],
                    [0, 0, 1],
                ]).float()

                self.palette = {
                    "eye":  palette[0].to(device),
                    "face": palette[1].to(device),
                    "ear":  palette[2].to(device),
                }

                import nvdiffrast.torch as dr
                v_clr = torch.einsum("vk,kc->vc", v_mask.float(), palette).unsqueeze(0).to(device)
                drctx = dr.RasterizeCudaContext(device=device)

                uv, uv_tri    = self.uv.to(device), self.uv_tri.int().contiguous().to(device)

                print(v_clr.shape)

                rast, rast_db = dr.rasterize(drctx, F.pad(2*uv[None]-1, (0, 2), mode="constant", value=1), uv_tri, resolution=[512, 512])
                im_seg, _     = dr.interpolate(v_clr.contiguous(), rast, tri.int().contiguous().to(device), rast_db)

                self.seg_map = im_seg.clone().detach()

                del drctx

            lmk_embeddings_mediapipe = np.load(self.config.flame_mediapipe_lmk_embedding_path, 
                allow_pickle=True, encoding='latin1')
            self.lmk_faces_idx_mediapipe   = torch.tensor(lmk_embeddings_mediapipe['lmk_face_idx'].astype(np.int64), dtype=torch.long).to(device)
            self.lmk_bary_coords_mediapipe = torch.tensor(lmk_embeddings_mediapipe['lmk_b_coords'], dtype=torch.float32).to(device)

            lmk_embeddings_gaze = np.load(self.config.flame_gaze_lmk_embedding_path, 
                allow_pickle=True, encoding='latin1')

            self.lmk_faces_idx_gaze   = torch.tensor(lmk_embeddings_gaze['lmk_face_idx'].astype(np.int64), dtype=torch.long).to(device)
            self.lmk_bary_coords_gaze = torch.tensor(lmk_embeddings_gaze['lmk_b_coords'], dtype=torch.float32).to(device)

            # emoca/gdl/layers/losses/MediaPipeLandmarkLosses.py
            self.used_mediapipe_indices = torch.as_tensor([276, 282, 283, 285, 293, 295, 296, 300, 334, 336,  46,  52,  53,
                55,  63,  65,  66,  70, 105, 107, 249, 263, 362, 373, 374, 380,
                381, 382, 384, 385, 386, 387, 388, 390, 398, 466,   7,  33, 133,
                144, 145, 153, 154, 155, 157, 158, 159, 160, 161, 163, 173, 246,
                168,   6, 197, 195,   5,   4, 129,  98,  97,   2, 326, 327, 358,
                0,  13,  14,  17,  37,  39,  40,  61,  78,  80,  81,  82,  84,
                87,  88,  91,  95, 146, 178, 181, 185, 191, 267, 269, 270, 291,
                308, 310, 311, 312, 314, 317, 318, 321, 324, 375, 402, 405, 409,
                415]).to(device)

            # https://github.com/tensorflow/tfjs-models/blob/838611c02f51159afdd77469ce67f0e26b7bbb23/face-landmarks-detection/src/mediapipe-facemesh/keypoints.ts
            # rightEyeIris: [473, 474, 475, 476, 477],
            # leftEyeIris: [468, 469, 470, 471, 472],
            self.used_mediapipe_gz_indices = torch.as_tensor([468, 473]).to(device)

    @staticmethod
    def vertices2landmarks(vertices, faces, lmk_faces_idx, lmk_bary_coords):
        # Extract the indices of the vertices for each face
        # BxLx3
        batch_size, num_verts = vertices.shape[:2]
        num_lmks = lmk_faces_idx.numel() // batch_size
        device = vertices.device

        lmk_faces = torch.index_select(faces, 0, lmk_faces_idx.reshape(-1)).view(
            batch_size, -1, 3)

        lmk_faces += torch.arange(
            batch_size, dtype=torch.long, device=device).view(-1, 1, 1) * num_verts

        lmk_vertices = vertices.view(batch_size*num_verts, -1)[lmk_faces].view(
            batch_size, num_lmks, 3, -1)
        
        landmarks = torch.einsum('blfi,blf->bli', [lmk_vertices, lmk_bary_coords])
        return landmarks
    
    @staticmethod
    def project_vertex(vertex, m2v, ndc):
        homo_v = F.pad(vertex, (0, 1), "constant", 1)
        proj_v = (homo_v @ (m2v.transpose(-1, -2) @ ndc.transpose(-1, -2)))
        return proj_v
    
    @staticmethod
    def perspective_projection_matrix(fx, fy, near=0.1, far=100, sign=-1, device=None):
        pers_ndc = torch.eye(4)

        n, f, s = near, far, sign
        # https://www.songho.ca/opengl/gl_projectionmatrix.html
        # fx,  0, 0, 0
        #  0, fy, 0, 0
        #  0,  0, s*(f+n)/(f-n), -2*f*n/(f-n)
        #  0,  0, s, 0
        pers_ndc[0, 0] = fx
        pers_ndc[1, 1] = fy
        pers_ndc[2, 2] = s*(f+n)/(f-n)  # [opencv camera] map 'f' to 1, instead of '-f' to 1
        pers_ndc[2, 3] = -2*f*n/(f-n)
        pers_ndc[3, 2] = s
        pers_ndc[3, 3] = 0

        if device is not None:
            pers_ndc = pers_ndc.to(device)
        return pers_ndc

    def __call__(self, image, image_flip, fp_tuple, fr_param, fr_param_flip):
        '''
        image: pillow.Image
        '''

        device = self.device
        import torch.nn as nn

        # img_u8 = np.asarray(image)

        fp_image, fp_label = fp_tuple

        tri    = self.flame.faces_tensor
        uv     = self.uv
        uv_tri = self.uv_tri

        stacked_mp_lmk_478 = torch.stack([fr_param["mp_lmk_478"], fr_param_flip["mp_lmk_478"]], dim=0).to(device)

        gt_lmk_2d = stacked_mp_lmk_478[:, self.used_mediapipe_indices]     # Nf, K, 2
        gt_eye_2d = stacked_mp_lmk_478[:, self.used_mediapipe_gz_indices]  # Nf, K_eye, 2

        tforms_0, tforms_1 = fr_param["transform"], fr_param_flip["transform"]
        result_0, result_1 = fr_param["flamecode"], fr_param_flip["flamecode"]

        stack_fr_param = {"original_code": {}}
        for k in ["shapecode", "expcode", "posecode", "detailcode", "detailemocode",
                    "texcode", "lightcode", "cam"]:
            stack_fr_param[k] = torch.stack([data[k] for data in [result_0, result_1]], dim=0).to(device)
        for k in ["shape", "exp", "pose", 
                    "tex", "light", "cam"]:
            stack_fr_param["original_code"][k] = torch.stack([data["original_code"][k] for data in [result_0, result_1]], dim=0).to(device)
        
        
        Nf = stack_fr_param["shapecode"].size(0)

        with nullcontext("faceparsing"):
            fp_image = fp_image.float()
            fp_clsid = fp_image.argmax(0)

            fp_fmmsk = torch.isin(fp_clsid, self.fp_label_to_mask[fp_label][1]).to(device)
            fp_eymsk = torch.isin(fp_clsid, self.fp_label_to_mask[fp_label][3]).to(device)
            fp_fmmsk_flip = torch.flip(fp_fmmsk, dims=(-1,)).to(device)
            fp_eymsk_flip = torch.flip(fp_eymsk, dims=(-1,)).to(device)

            fp_famsk = torch.isin(fp_clsid, self.fp_label_to_part[fp_label]["face"]).to(device)
            fp_ermsk = torch.isin(fp_clsid, self.fp_label_to_part[fp_label]["ear"]).to(device)
            fp_famsk_flip = torch.flip(fp_famsk, dims=(-1,)).to(device)
            fp_ermsk_flip = torch.flip(fp_ermsk, dims=(-1,)).to(device)

            fp_weight      = (fp_fmmsk.float()      + 3*fp_eymsk.float()     ).unsqueeze(-1) # 
            fp_weight_flip = (fp_fmmsk_flip.float() + 3*fp_eymsk_flip.float()).unsqueeze(-1) # 

            fp_weight_used = torch.stack([fp_weight, fp_weight_flip], dim=0)

            # blur = torchvision.transforms.GaussianBlur(2*int(fp_weight_used.size(1)/5)+1)
            # fp_weight_used = blur(fp_weight_used.permute(0, 3, 1, 2)).permute(0, 2, 3, 1)

            dilate_size = 1+2*int(fp_weight_used.size(1)/80)
            fp_weight_used = F.pad(fp_weight_used.permute(0, 3, 1, 2), [(dilate_size-1)//2]*4, mode="reflect")
            fp_weight_used = F.max_pool2d(fp_weight_used, kernel_size=dilate_size, stride=1, padding=0).permute(0, 2, 3, 1)


            fp_segimg      = (self.palette["face"].reshape(1,1,-1)*fp_famsk[...,None].float()    + \
                              self.palette["eye"].reshape(1,1,-1) *fp_eymsk[...,None].float()    + \
                              self.palette["ear"].reshape(1,1,-1) *fp_ermsk[...,None].float()     ) # 
            fp_segimg_flip = (self.palette["face"].reshape(1,1,-1)*fp_famsk_flip[...,None].float() + \
                              self.palette["eye"].reshape(1,1,-1) *fp_eymsk_flip[...,None].float() + \
                              self.palette["ear"].reshape(1,1,-1) *fp_ermsk_flip[...,None].float()) # 
            fp_segimg_used = torch.stack([fp_segimg, fp_segimg_flip], dim=0)

        with nullcontext("original orthogonal"), torch.no_grad():
            model_pose = torch.zeros_like(stack_fr_param['posecode'])
            model_pose[:, 3:] = stack_fr_param['posecode'][:, 3:]      # set roo=0, keep jaw

            orth_m2w = torch.eye(4).unsqueeze(0).repeat(Nf, 1, 1).to(device)
            orth_m2w[:, :3, :3] *= stack_fr_param["cam"][:, 0, None, None]
            orth_m2w[:, 0, 3]    = stack_fr_param["cam"][:, 1] * stack_fr_param["cam"][:, 0]
            orth_m2w[:, 1, 3]    = stack_fr_param["cam"][:, 2] * stack_fr_param["cam"][:, 0]

            orth_m2w = orth_m2w# @ root_rot

            tform = torch.inverse(torch.stack([tforms_0, tforms_1], dim=0)).to(device)  # R | t
            t_2d  = torch.eye(3)[None, :, :].repeat(Nf, 1, 1).to(device)
            t_2d[:, 0, 0] = 0.5*224
            t_2d[:, 1, 1] = 0.5*224
            t_2d[:, 0, 2] = 0.5*224
            t_2d[:, 1, 2] = 0.5*224

            t_2d = tform@t_2d

            orth_t3d_l = []
            for bi in range(Nf):
                z_scale = torch.diagonal(t_2d[bi, :2,:2]).abs().mean()

                orth_t3d = torch.diag(torch.as_tensor([z_scale/512, z_scale/512, z_scale/512, 1]))
                orth_t3d[0,  3] = t_2d[bi, 0, 2]/512 - 0.5
                orth_t3d[1,  3] = 0.5 - t_2d[bi, 1, 2]/512   # 3D y-axis is pointing up, 2D y is pointing down
                orth_t3d_l.append(orth_t3d)
            orth_t3d = torch.stack(orth_t3d_l, dim=0)
            orth_m2w = orth_t3d.to(orth_m2w.device) @ orth_m2w

            orth_w2v = torch.eye(4).to(orth_m2w.device)
            orth_w2v[2,  3] = -3
            orth_w2v = orth_w2v[None].expand(Nf, -1, -1)

            orth_m2v = orth_w2v @ orth_m2w

            n, f = 0.001, 20
            orth_ndc = torch.eye(4).to(device)
            orth_ndc[0, 0] = 2
            orth_ndc[1, 1] = -2
            orth_ndc[2, 2] = -2/(f-n)  # [opencv camera] map 'f' to 1, instead of '-f' to 1
            orth_ndc[2, 3] = -(f+n)/(f-n)
            orth_ndc = orth_ndc[None].expand(Nf, -1, -1)

            shape, exp = stack_fr_param["shapecode"], stack_fr_param["expcode"]
            pose, eyep = stack_fr_param["posecode"], None
            tex        = stack_fr_param["texcode"]
            verts, landmarks2d, landmarks3d = self.flame(
                shape_params=shape, expression_params=exp, 
                pose_params=pose, eye_pose_params=eyep)
            
            orth_v = self.project_vertex(verts, orth_m2v, orth_ndc)
            orth_d = orth_v / orth_v[..., 3:]

        # perspective
        with nullcontext("perspective"):
            fx = fy = (1015 * 300 / 102 / 700) # focal_length * new_rescale / old_rescale / crop_res
            n, f = 0.1, 10
            s = -1

            # 1/np.tan(fov/2) = focal_length / (width/2)
            pers_ndc = self.perspective_projection_matrix(2*fx, 2*fy, near=n, far=f, sign=s, device=device)

            # print(pers_ndc)
            # tensor([[ 8.5294,  0.0000,  0.0000,  0.0000],
            #         [ 0.0000,  8.5294,  0.0000,  0.0000],
            #         [ 0.0000,  0.0000, -1.0202, -0.2020],
            #         [ 0.0000,  0.0000, -1.0000,  0.0000]], device='cuda:0')

        with nullcontext("params"):
            shape_param  = nn.Parameter(stack_fr_param['shapecode'].clone().to(device))
            # shape_param  = nn.Parameter(stack_fr_param['shapecode'].mean(dim=0, keepdim=True).clone().to(device))
            exp_param    = nn.Parameter(stack_fr_param['expcode'].clone().to(device))
            tex_param    = nn.Parameter(stack_fr_param["texcode"].clone().to(device))
            # tex_param    = nn.Parameter(stack_fr_param["texcode"].mean(dim=0, keepdim=True).clone().to(device))
            jaw_param    = nn.Parameter(stack_fr_param["posecode"][:, 3:].to(device))
            eyep_param   = nn.Parameter(torch.zeros(Nf, 6).to(device))
            lights_param = nn.Parameter(stack_fr_param["lightcode"].clone().to(device))

            if self.flip_cam_yaw:
                cam_rot = nn.Parameter(stack_fr_param["posecode"][:1, :3].float().to(device))
                t3d_vec = nn.Parameter(torch.as_tensor([[0.0, 0.0, -1.0]]).float().to(device))
            else:
                cam_rot = nn.Parameter(stack_fr_param["posecode"][:, :3].float().to(device))
                t3d_vec = nn.Parameter(torch.as_tensor([[0.0, 0.0, -1.0]]).expand(Nf, -1).float().to(device))

            def get_flame_param():
                return shape_param.expand(Nf, -1), exp_param, torch.cat([torch.zeros_like(jaw_param), jaw_param], dim=1), eyep_param, tex_param.expand(Nf, -1), lights_param
            
            rot_44_last_row = torch.as_tensor([0,0,0,1]).reshape(1, 1, 4).to(device)
            flip_44 = torch.ones(4, 4)
            flip_44[0, 1] *= -1
            flip_44[0, 2] *= -1
            flip_44[1, 0] *= -1
            flip_44[2, 0] *= -1
            flip_44[0, 3] *= -1
            flip_44 = flip_44.reshape(1, 4, 4).to(device)

            def get_camera_param():
                rot_33 = batch_rodrigues(cam_rot)                            # ...,3,3
                rot_34 = torch.cat([rot_33, t3d_vec.unsqueeze(-1)], dim=-1)  # ...,3,4
                rot_44 = torch.cat([rot_34, rot_44_last_row], dim=-2)
                if self.flip_cam_yaw:
                    rot_44 = torch.cat([rot_44, rot_44*flip_44], dim=0)
                return rot_44.expand(Nf, -1, -1)

            opt_pose = torch.optim.Adam(
                [cam_rot, t3d_vec],
                lr=0.01,
                weight_decay=0.0001
            )
            shd_pose = torch.optim.lr_scheduler.StepLR(opt_pose, 1, (0.5**0.1)) # half by 10 step
            opt_coarse = torch.optim.Adam(
                [cam_rot, t3d_vec, shape_param, exp_param, jaw_param],
                lr=0.005,
                weight_decay=0.0001
            )
            shd_coarse = torch.optim.lr_scheduler.StepLR(opt_coarse, 1, (0.5**0.01)) # half by 100 step
            
            opt_eye = torch.optim.Adam(
                [eyep_param],
                lr=0.01,
            )
            shd_eye = torch.optim.lr_scheduler.StepLR(opt_eye, 1, (0.5**0.01)) # half by 100 step

            opt_fine = torch.optim.Adam(
                [cam_rot, t3d_vec] + \
                [shape_param, exp_param, jaw_param, eyep_param] + \
                [tex_param, lights_param],
                lr=0.010,
            )
            shd_fine = torch.optim.lr_scheduler.StepLR(opt_fine, 1, (0.5**0.01)) # half by 100 step

        # batch_size = 1

        with torch.enable_grad():
            # for i in range(20):
            # tex    = self.tex
            gt_im = torch.stack([torch.as_tensor(np.array(image)), torch.as_tensor(np.array(image_flip))], dim=0)
            gt_im = (gt_im.float()/255.).to(device)

            # projected 2D point
            w_lmk, w_eye = 50, 50
            w_prj = torch.ones_like(orth_d[..., 0:1])*10
            # # image space
            # w_pho, w_vgg, w_dpt = 0.1, 1, 1
            # # regularize
            # w_shp_reg, w_exp_reg, w_pos_reg = 1e-4, 1e-4, 1e-3
            def get_landmark(proj_v, tri):
                batch_size = proj_v.size(0)
                lmk_faces_idx_mp = self.lmk_faces_idx_mediapipe.unsqueeze(dim=0).expand(batch_size, -1)
                lmk_bary_coords_mp = self.lmk_bary_coords_mediapipe.unsqueeze(dim=0).expand(batch_size, -1, -1)

                landmark_4d_mp = self.vertices2landmarks(proj_v, tri,
                                        lmk_faces_idx_mp, lmk_bary_coords_mp)
                landmark_2d_mp = landmark_4d_mp[..., :2] / landmark_4d_mp[..., 3:]
                return 0.5*landmark_2d_mp+0.5
            def get_gaze_lmk(proj_v, tri):
                batch_size = proj_v.size(0)
                lmk_faces_idx_gz = self.lmk_faces_idx_gaze.unsqueeze(dim=0).expand(batch_size, -1)
                lmk_bary_coords_gz = self.lmk_bary_coords_gaze.unsqueeze(dim=0).expand(batch_size, -1, -1)

                landmark_4d_gz = self.vertices2landmarks(proj_v, tri,
                                        lmk_faces_idx_gz, lmk_bary_coords_gz)
                landmark_2d_gz = landmark_4d_gz[..., :2] / landmark_4d_gz[..., 3:]
                return 0.5*landmark_2d_gz+0.5
            
            reverse_y = torch.diag(torch.as_tensor([1,-1,1,1]).float().to(device))

            h = w = 256
            gt_im = F.interpolate(gt_im.permute(0, 3, 1, 2), (h, w), mode="bilinear", align_corners=False, antialias=True).permute(0, 2, 3, 1)
            gt_sg = F.interpolate(fp_segimg_used.permute(0, 3, 1, 2), (h, w), mode="bilinear", align_corners=False, antialias=True).permute(0, 2, 3, 1)
            wt_im = F.interpolate(fp_weight_used.permute(0, 3, 1, 2), (h, w), mode="bilinear", align_corners=False, antialias=True).permute(0, 2, 3, 1)

            NUM_POSE   = 20
            NUM_COARSE = 50
            NUM_EYE    = 30
            NUM_FINE   = 100
            NUM_TOTAL  = NUM_POSE + NUM_COARSE + NUM_EYE + NUM_FINE
            for i in range(NUM_TOTAL):
                (shape, exp, pose, eyep, tex, light), pers_m2v = get_flame_param(), get_camera_param()

                verts, landmarks2d, landmarks3d = self.flame(
                    shape_params=shape, expression_params=exp, 
                    pose_params=pose, eye_pose_params=eyep)
                
                verts = verts + self.offset.reshape(1, 1, 3)
                
                m2v_matx = pers_m2v
                ndc_matx = pers_ndc @ reverse_y

                proj_v = self.project_vertex(verts, m2v_matx, ndc_matx)
                proj_d = proj_v / proj_v[..., 3:]

                pd_lmk_2d = get_landmark(proj_v, tri)
                pd_eye_2d = get_gaze_lmk(proj_v, tri)
                loss_lmk = F.mse_loss(gt_lmk_2d, pd_lmk_2d) * w_lmk
                loss_eye = F.mse_loss(gt_eye_2d, pd_eye_2d) * w_eye
                loss_prj = (w_prj * F.mse_loss(proj_d[..., :2], orth_d[..., :2], reduction="none")).mean()

                if i < NUM_POSE:
                    loss = loss_lmk + loss_prj
                    optm = opt_pose
                    schd = shd_pose
                elif i < NUM_POSE + NUM_COARSE:
                    loss = loss_lmk + loss_prj
                    optm = opt_coarse
                    schd = shd_coarse
                elif i < NUM_POSE + NUM_COARSE + NUM_EYE:
                    loss = loss_eye
                    optm = opt_eye
                    schd = shd_eye
                elif i < NUM_POSE + NUM_COARSE + NUM_EYE + NUM_FINE:
                    tex_map = self.flametex(tex)

                    m2v_matx.register_hook(lambda g:g*0.2)
                    verts.register_hook(lambda g:g*0.2)

                    pd_im, alpha, normal, depth = render_nvdr(m2v_matx, ndc_matx.unsqueeze(0), verts, tri, uv, uv_tri, 
                        torch.cat([tex_map.permute(0, 2, 3, 1), self.seg_map.expand(tex_map.size(0), -1, -1, -1)], dim=-1), H=h, W=w)

                    pd_im, pd_sg = torch.split(pd_im, [tex_map.size(1), self.seg_map.size(-1)], dim=-1)

                    shade = shading_SH2(normal.flatten(1, 2), light).unflatten(1, (normal.shape[1:3]))
                    pd_im = pd_im*shade

                    loss_img = ((gt_im - pd_im).square() * wt_im).mean()
                    loss_seg = ((gt_sg - pd_sg).square() * wt_im).mean()

                    loss = loss_lmk + loss_eye + loss_img + loss_seg
                    optm = opt_fine
                    schd = shd_fine
                
                l_id_sym = 1e-3*(shape_param.std(dim=0).mean() + tex_param.std(dim=0).mean())
                l_norm   = 1e-5*(shape_param.norm(p=2, dim=-1).mean() + exp_param.norm(p=2, dim=-1).mean() + tex_param.norm(p=2, dim=-1).mean())
                loss = loss + l_id_sym + l_norm

                optm.zero_grad()
                loss.backward()
                optm.step()
                schd.step()

        (shape, exp, pose, eyep, tex, light), pers_m2v = get_flame_param(), get_camera_param()

        result_0 = {
            "shape":      shape[0],
            "exp":        exp[0],
            "jaw_pose":   jaw_param[0],
            "eye_pose":   eyep[0],
            "tex":        tex[0],
            "light":      light[0],
            "m2v_matrix": pers_m2v[0],
            "ndc_matrix": pers_ndc,
            "img_weight": fp_weight_used[0],
            "seg_gt":     gt_sg[0],
            "seg_pd":     pd_sg[0]
        }
        result_1 = {
            "shape":      shape[1],
            "exp":        exp[1],
            "jaw_pose":   jaw_param[1],
            "eye_pose":   eyep[1],
            "tex":        tex[1],
            "light":      light[1],
            "m2v_matrix": pers_m2v[1],
            "ndc_matrix": pers_ndc,
            "img_weight": fp_weight_used[1],
            "seg_gt":     gt_sg[1],
            "seg_pd":     pd_sg[1]
        }
        return result_0, result_1

class Faceparsing2Segment(object):
    def __init__(self, device):
        super().__init__()

        self.f_bg_clr   = torch.as_tensor([0,   0,   0]).float().reshape(1, 1, 3).to(device)  # black
        self.f_eyes_clr = torch.as_tensor([1.0, 0,   0]).float().reshape(1, 1, 3).to(device)  # read
        self.f_head_clr = torch.as_tensor([0, 1.0,   0]).float().reshape(1, 1, 3).to(device)  # green
        self.f_free_clr = torch.as_tensor([0,   0, 1.0]).float().reshape(1, 1, 3).to(device)  # blue

        label2inds = {}

        with nullcontext("celebm/448"):
            annot    = ['background', 'neck', 'face', 'cloth', 'r_ear', 
                        'l_ear', 'r_brow', 'l_brow', 'r_eye', 'l_eye', 
                        'nose',  'i_mouth', 'l_lip', 'u_lip', 'hair',
                        'eye_g', 'hat', 'ear_r', 'neck_l']
            bg_cls   = ['background']
            head_cls = ['face', 'r_ear', 'l_ear', 'r_brow', 'l_brow', 'nose', 'l_lip', 'u_lip']
            free_cls = ['neck', 'cloth', 'i_mouth', 'hair', 'eye_g', 'hat', 'ear_r', 'neck_l']
            eyes_cls = ['r_eye', 'l_eye']

            assert len(bg_cls+head_cls+free_cls+eyes_cls) == len(annot)

            bg_inds   = torch.as_tensor([annot.index(s) for s in   bg_cls]).long().to(device)
            head_inds = torch.as_tensor([annot.index(s) for s in head_cls]).long().to(device)
            free_inds = torch.as_tensor([annot.index(s) for s in free_cls]).long().to(device)
            eyes_inds = torch.as_tensor([annot.index(s) for s in eyes_cls]).long().to(device)

            label2inds["celebm"] = (bg_inds, head_inds, free_inds, eyes_inds)

        self.label2inds = label2inds
    
    @torch.no_grad()
    def __call__(self, fp_image, fp_label):
        bg_inds, head_inds, free_inds, eyes_inds = self.label2inds[fp_label]
        return self.pd2clr(fp_image, bg_inds, head_inds, free_inds, eyes_inds)

    def pd2clr(self, pd_img, bg_inds, head_inds, free_inds, eyes_inds):
        f_bg_clr, f_head_clr, f_free_clr, f_eyes_clr = self.f_bg_clr, self.f_head_clr, self.f_free_clr, self.f_eyes_clr

        # pd_img = pd_img.astype(np.float32)
        # pd_img = pd_img / (pd_img.sum(axis=0, keepdims=True))
        pd_img = pd_img.float()
        pd_img = pd_img / (pd_img.sum(dim=0, keepdim=True))

        bg_alp   = pd_img[bg_inds].sum(axis=0)[:, :, None]
        head_alp = pd_img[head_inds].sum(axis=0)[:, :, None]
        free_alp = pd_img[free_inds].sum(axis=0)[:, :, None]
        eyes_alp = pd_img[eyes_inds].sum(axis=0)[:, :, None]

        color = bg_alp*f_bg_clr + head_alp*f_head_clr + free_alp*f_free_clr + eyes_alp*f_eyes_clr
        return color
        # return (255*color.astype(np.float32)).astype(np.uint8)

def process_fn(dev_id, args, in_value, out_queue):
    import torch
    device = torch.device("cuda", dev_id)
    torch.cuda.set_device(device)

    try:
        with nullcontext("compacibility"):
            import numpy as np
            old_type = ["bool", "int", "float", "complex", "object", "unicode", "str"]
            for t in old_type:
                if not hasattr(np, t):
                    setattr(np, t, getattr(np, f"{t}_"))

            import face_alignment
            face_alignment.LandmarksType._2D = face_alignment.LandmarksType.TWO_D

        with nullcontext("render"):

            model_root = os.path.join(ROOT)
            sys.path.insert(0, model_root)
            import importlib
            importlib.invalidate_caches()
            MODELS_mod = importlib.import_module("models")
            importlib.reload(MODELS_mod)
            FLAME_mod = importlib.import_module("models.FLAME")
            sys.path.pop(0)
            from easydict import EasyDict

            FLAME    = FLAME_mod.FLAME
            FLAMETex = FLAME_mod.FLAMETex

            config = EasyDict({
                "flame_model_path":        os.path.join(ROOT, "Data", "FLAME2020", "generic_model.pkl"),
                "flame_lmk_embedding_path":os.path.join(ROOT, "Data", "FLAME2020", "landmark_embedding.npy"),
                "flame_tex_path":          os.path.join(ROOT, "Data", "FLAME2020", "FLAME_texture.npz"),
                "tex_path":                os.path.join(ROOT, "Data", "FLAME2020", "FLAME_albedo_from_BFM.npz"),
                "flame_mediapipe_lmk_embedding_path": os.path.join(ROOT, "Data", "FLAME2020", "mediapipe_landmark_embedding.npz"),
                "flame_gaze_lmk_embedding_path":      os.path.join(ROOT, "Data", "FLAME2020", "gaze_landmark_embedding2.npz"),
                "n_shape": 100,
                "n_exp":   50,
                "n_tex":   50,
                "tex_type": 'BFM',
            })

            flame      = FLAME(config).to(device)
            flametex   = FLAMETex(config).to(device)

            offset  = torch.as_tensor([0.0, 0.00, 0.05]).to(device)

            flame_tex_f = os.path.join(ROOT, "Data", "FLAME2020", "FLAME_texture.npz")
            flame_tex = np.load(flame_tex_f)
            uv     = torch.as_tensor(flame_tex["vt"]).float()
            uv_tri = torch.as_tensor(flame_tex["ft"].astype(np.int32)).long()
            uv[:, 1] = 1 - uv[:, 1]
            uv, uv_tri = uv.to(device), uv_tri.to(device)

        with nullcontext("model"):
            model = PhotometricRunner(device, args)
            fp2seg= Faceparsing2Segment(torch.device("cpu"))
    except:
        import traceback
        print(traceback.format_exc())

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
            name,ext       = os.path.splitext(im_path)
            pth_fpath      = os.path.join(args.fitdir, f"{name}.pth")
            pth_fpath_flip = os.path.join(args.fitdir, f"{name}_mirror.pth")
            png_fpath      = os.path.join(args.segdir, f"{name}.png")
            png_fpath_flip = os.path.join(args.segdir, f"{name}_mirror.png")

            if args.skip_if_exist:
                if os.path.exists(pth_fpath) and os.path.exists(pth_fpath_flip):
                    if os.path.exists(png_fpath) and os.path.exists(png_fpath_flip):
                        continue

            im_flip = f"{name}_mirror{ext}"
            
            fp_full_path      = os.path.join(args.fpdir, f"{name}.pth")
            fr_full_path      = os.path.join(args.frdir, f"{name}.pth")
            fr_full_path_flip = os.path.join(args.frdir, f"{name}_mirror.pth")

            im      = Image.open(os.path.join(args.indir, im_path)).convert('RGB')
            im_flip = Image.open(os.path.join(args.indir, im_flip)).convert('RGB') if im_flip is not None else ImageOps.mirror(im)

            fp_saved = torch.load(fp_full_path)                          # dict(mask_xxx=Tensor[C, H, W])
            fp_image, fp_label = parse_fp_dict(fp_saved)

            seg_img, seg_img_flip = fp2seg(fp_image, fp_label), fp2seg(torch.flip(fp_image, dims=(-1,)), fp_label)

            fr_param      = torch.load(fr_full_path)                 # dict(...)
            fr_param_flip = torch.load(fr_full_path_flip)            # dict(...)

            result, result_flip = model(im, im_flip, (fp_image, fp_label), fr_param, fr_param_flip)

            if args.debug is True:
                # flame, flametex = model.flame, model.flametex

                result_all = {}
                for k in ["shape", "exp", "jaw_pose", "eye_pose", 
                          "tex", "light", "m2v_matrix", "ndc_matrix"]:
                    result_all[k] = torch.stack([data[k] for data in (result, result_flip)], dim=0)

                RES = 256
                BS  = 2
                img_by_row = []

                img_weight = torch.cat([result["img_weight"], result_flip["img_weight"]], dim=1).expand(-1, -1, 3).to(device)
                seg_gt     = resize_2d(torch.cat([result["seg_gt"], result_flip["seg_gt"]], dim=1).unsqueeze(0).to(device), img_weight.size(0), img_weight.size(1)).squeeze(0)
                seg_pd     = resize_2d(torch.cat([result["seg_pd"], result_flip["seg_pd"]], dim=1).unsqueeze(0).to(device), img_weight.size(0), img_weight.size(1)).squeeze(0)
                img_seg    = torch.cat([seg_img, seg_img_flip], dim=1).expand(-1, -1, 3).to(device)

                img_origin = np.concatenate([np.asarray(im), np.asarray(im_flip)], axis=1)/255.
                img_origin = cv2.resize(img_origin.astype(np.float32),  (RES*BS, RES), interpolation=cv2.INTER_AREA)
                img_by_row.append(img_origin)

                is_neck  = torch.isin(fp_image.argmax(0), torch.as_tensor([1]))            # neck, cloth
                img_neck = torch.cat([is_neck, torch.flip(is_neck, dims=(-1,))], dim=-1).unsqueeze(-1).expand(-1,-1,3).cpu().numpy()
                img_neck = cv2.resize(img_neck.astype(np.float32),  (RES*BS, RES), interpolation=cv2.INTER_AREA)
                # img_by_row.append(img_neck)

                shape, exp = result_all["shape"], result_all["exp"]
                pose, eyep = torch.cat([torch.zeros_like(result_all["jaw_pose"]),  result_all["jaw_pose"]], dim=-1), result_all["eye_pose"]
                tex        = result_all["tex"]
                verts, landmarks2d, landmarks3d = flame(
                    shape_params=shape, expression_params=exp, 
                    pose_params=pose, eye_pose_params=eyep)
                verts = verts + offset.reshape(1, 1, 3)
                
                tex_map = flametex(tex)

                tri = flame.faces_tensor

                with nullcontext("draw perspective"):
                    pers_m2v = result_all["m2v_matrix"]
                    pers_ndc = result_all["ndc_matrix"]
                    reverse_y = torch.diag(torch.as_tensor([1,-1,1,1]).float().to(device))

                    ndc_mtx_= pers_ndc @ reverse_y.unsqueeze(0)

                    image, alpha, normal, depth = render_nvdr(pers_m2v, ndc_mtx_, verts, tri, uv, uv_tri, tex_map.permute(0, 2, 3, 1))

                    shade = shading_SH2(normal.flatten(1, 2), result_all["light"]).unflatten(1, (normal.shape[1:3]))
                    image = image*shade

                    image = image.transpose(0, 1).flatten(1, 2)  # H, K*W, 3
                    normal= normal.transpose(0, 1).flatten(1, 2) # H, K*W, 3
                    alpha = alpha.transpose(0, 1).flatten(1, 2)  # H, K*W, 3

                    input = torch.as_tensor(np.concatenate([np.asarray(im), np.asarray(im_flip)], axis=1)/255.).to(device)

                    # image = torch.cat([img_weight, image, 0.5+0.5*normal, 0.8*input+0.4*(0.5+0.5*normal)*alpha], dim=0)
                    image = torch.cat([image, 0.5+0.5*normal, 0.8*input+0.4*(0.5+0.5*normal)*alpha, img_weight, img_seg, 0.4*input+0.6*img_seg, seg_gt, seg_pd], dim=0)

                    row_of_output = image.detach().cpu().numpy()

                img_output = cv2.resize(row_of_output.astype(np.float32),  (RES*BS, 8*RES), interpolation=cv2.INTER_AREA)
                img_by_row.append(img_output)

                save_img = (255*np.concatenate(img_by_row, axis=0).clip(0, 1)).astype(np.uint8).copy()

                cv2.imwrite(os.path.join(SAVE_ROOT, f"{name}.png"), cv2.cvtColor(save_img, cv2.COLOR_RGB2BGR))

            if "img_weight" in result:
                del result["img_weight"]
                del result_flip["img_weight"]

            if "seg_gt" in result:
                del result["seg_gt"]
                del result_flip["seg_gt"]

            if "seg_pd" in result:
                del result["seg_pd"]
                del result_flip["seg_pd"]

            result      = {k: v.detach().cpu() for k,v in result.items()}
            result_flip = {k: v.detach().cpu() for k,v in result_flip.items()}

            seg_img, seg_img_flip = map(lambda th_im: (255*th_im.clamp(0,1).detach().cpu().numpy()).astype(np.uint8), [seg_img, seg_img_flip])
            seg_img      = Image.fromarray(seg_img.astype('uint8')).convert('RGB')
            seg_img_flip = Image.fromarray(seg_img_flip.astype('uint8')).convert('RGB')
            # imageio.imwrite(png_fpath,      seg_img)
            # imageio.imwrite(png_fpath_flip, seg_img_flip)

            seg_img.save(png_fpath,           compress_level=0, optimize=False)
            seg_img_flip.save(png_fpath_flip, compress_level=0, optimize=False)

            with open(pth_fpath, "wb") as f:
                torch.save(result, f)
            with open(pth_fpath_flip, "wb") as f:
                torch.save(result_flip, f)
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

        # fnames = [im_path for (im_path, _) in labels]
        fnames = [im_path for (im_path, _) in labels if '_mirror' not in im_path]
        fnames = sorted(list(set(fnames)))
    
    if args.start_index is not None:
        fnames = fnames[args.start_index:]
    if args.num_samples is not None:
        fnames = fnames[:args.num_samples]
    
    args.num_gpus = num_gpus
    args.dataset  = fnames
            
    os.makedirs(args.fitdir, exist_ok=True)
    os.makedirs(args.segdir, exist_ok=True)

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
    parser.add_argument('--fpdir',  type=str, required=True)
    parser.add_argument('--frdir',  type=str, required=True)
    parser.add_argument('--fitdir', type=str, required=True)
    parser.add_argument('--segdir', type=str, required=True)

    parser.add_argument('--debug', action="store_true")

    parser.add_argument("--flip_cam_yaw", type=lambda x:eval(x), default=True)

    parser.add_argument('--start_index', type=int, default=None)
    parser.add_argument('--num_samples', type=int, default=None)
    args = parser.parse_args()

    main(args)