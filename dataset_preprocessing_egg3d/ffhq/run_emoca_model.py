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
SAVE_ROOT = os.path.join(ROOT, "temp", "emoca_debug")

os.makedirs(SAVE_ROOT, exist_ok=True)

sys.path.insert(0, ROOT)
import utils
# import models.BiSeNet as FACEP_mod
# from utils.face_2d import DlibLandmark, MediaPipeLandmark, GMMSkinSegmentation, PTNetRun
sys.path.pop(0)

def resize_2d(img_bhwc, H, W):
    b, h, w, c = img_bhwc.shape

    if (h, w) != (H, W):
        img_bhwc = F.interpolate(img_bhwc.permute(0, 3, 1, 2), (H, W)).permute(0, 2, 3, 1)
    return img_bhwc

import cv2
import numpy as np
import scipy
import torch
from skimage.io import imread
from skimage.transform import rescale, estimate_transform, warp
from torch.utils.data import Dataset
class EMOCATestData(object):
    def __init__(self, testpath, iscrop=True, crop_size=224, scale=1.25, face_detector='fan',
                 scaling_factor=1.0, max_detection=None):
        self.max_detection = max_detection
        if isinstance(testpath, list):
            self.imagepath_list = testpath
        elif os.path.isdir(testpath):
            self.imagepath_list = glob(testpath + '/*.jpg') + glob(testpath + '/*.png') + glob(testpath + '/*.bmp')
        elif os.path.isfile(testpath) and (testpath[-3:] in ['jpg', 'png', 'bmp']):
            self.imagepath_list = [testpath]
        elif os.path.isfile(testpath) and (testpath[-3:] in ['mp4', 'csv', 'vid', 'ebm']):
            self.imagepath_list = video2sequence(testpath)
        else:
            print(f'please check the test path: {testpath}')
            exit()
        print('total {} images'.format(len(self.imagepath_list)))
        self.imagepath_list = sorted(self.imagepath_list)
        self.scaling_factor = scaling_factor
        self.crop_size = crop_size
        self.scale = scale
        self.iscrop = iscrop
        self.resolution_inp = crop_size
        # add_pretrained_deca_to_path()
        # from decalib.datasets import detectors
        if face_detector == 'fan':
            from gdl.utils.FaceDetector import FAN
            self.face_detector = FAN()
        # elif face_detector == 'mtcnn':
        #     self.face_detector = detectors.MTCNN()
        else:
            print(f'please check the detector: {face_detector}')
            exit()

    def __len__(self):
        return len(self.imagepath_list)

    def __getitem__(self, index):
        imagepath = str(self.imagepath_list[index])
        imagename = imagepath.split('/')[-1].split('.')[0]

        image = np.array(imread(imagepath))

        return self.process(image)
    
    def process(self, image, kpt=None, kpt_type="kpt68"):
        from gdl.datasets.ImageDatasetHelpers import bbox2point
        if len(image.shape) == 2:
            image = image[:, :, None].repeat(1, 1, 3)
        if len(image.shape) == 3 and image.shape[2] > 3:
            image = image[:, :, :3]

        if self.scaling_factor != 1.:
            image = rescale(image, (self.scaling_factor, self.scaling_factor, 1))*255.

        h, w, _ = image.shape
        if self.iscrop:
            # # provide kpt as txt file, or mat file (for AFLW2000)
            # kpt_matpath = imagepath.replace('.jpg', '.mat').replace('.png', '.mat')
            # kpt_txtpath = imagepath.replace('.jpg', '.txt').replace('.png', '.txt')
            # if os.path.exists(kpt_matpath):
            #     kpt = scipy.io.loadmat(kpt_matpath)['pt3d_68'].T
            #     left = np.min(kpt[:, 0])
            #     right = np.max(kpt[:, 0])
            #     top = np.min(kpt[:, 1])
            #     bottom = np.max(kpt[:, 1])
            #     old_size, center = bbox2point(left, right, top, bottom, type='kpt68')
            # elif os.path.exists(kpt_txtpath):
            #     kpt = np.loadtxt(kpt_txtpath)
            #     left = np.min(kpt[:, 0])
            #     right = np.max(kpt[:, 0])
            #     top = np.min(kpt[:, 1])
            #     bottom = np.max(kpt[:, 1])
            #     old_size, center = bbox2point(left, right, top, bottom, type='kpt68')
            if kpt is not None:
                left = np.min(kpt[:, 0])
                right = np.max(kpt[:, 0])
                top = np.min(kpt[:, 1])
                bottom = np.max(kpt[:, 1])
                old_size, center = bbox2point(left, right, top, bottom, type=kpt_type)
            else:
                # bbox, bbox_type, landmarks = self.face_detector.run(image)
                bbox, bbox_type = self.face_detector.run(image)
                if len(bbox) < 1:
                    print('no face detected! run original image')
                    left = 0
                    right = h - 1
                    top = 0
                    bottom = w - 1
                    old_size, center = bbox2point(left, right, top, bottom, type=bbox_type)
                else:
                    if self.max_detection is None:
                        bbox = bbox[0]
                        left = bbox[0]
                        right = bbox[2]
                        top = bbox[1]
                        bottom = bbox[3]
                        old_size, center = bbox2point(left, right, top, bottom, type=bbox_type)
                    else: 
                        old_size, center = [], []
                        num_det = min(self.max_detection, len(bbox))
                        for bbi in range(num_det):
                            bb = bbox[0]
                            left = bb[0]
                            right = bb[2]
                            top = bb[1]
                            bottom = bb[3]
                            osz, c = bbox2point(left, right, top, bottom, type=bbox_type)
                        old_size += [osz]
                        center += [c]
            
            if isinstance(old_size, list):
                size = []
                src_pts = []
                for i in range(len(old_size)):
                    size += [int(old_size[i] * self.scale)]
                    src_pts += [np.array(
                        [[center[i][0] - size[i] / 2, center[i][1] - size[i] / 2], [center[i][0] - size[i] / 2, center[i][1] + size[i] / 2],
                        [center[i][0] + size[i] / 2, center[i][1] - size[i] / 2]])]
            else:
                size = int(old_size * self.scale)
                src_pts = np.array(
                    [[center[0] - size / 2, center[1] - size / 2], [center[0] - size / 2, center[1] + size / 2],
                    [center[0] + size / 2, center[1] - size / 2]])
        else:
            src_pts = np.array([[0, 0], [0, h - 1], [w - 1, 0]])
        
        image = image / 255.
        if not isinstance(src_pts, list):
            DST_PTS = np.array([[0, 0], [0, self.resolution_inp - 1], [self.resolution_inp - 1, 0]])
            tform = estimate_transform('similarity', src_pts, DST_PTS)
            dst_image = warp(image, tform.inverse, output_shape=(self.resolution_inp, self.resolution_inp))
            dst_image = dst_image.transpose(2, 0, 1)
            return {'image': torch.tensor(dst_image).float(),
                    # 'image_name': imagename,
                    # 'image_path': imagepath,
                    'tform': torch.as_tensor(tform.params).float(),
                    # 'original_image': torch.tensor(image.transpose(2,0,1)).float(),
                    }
        else:
            DST_PTS = np.array([[0, 0], [0, self.resolution_inp - 1], [self.resolution_inp - 1, 0]])
            dst_images = []
            for i in range(len(src_pts)):
                tform = estimate_transform('similarity', src_pts[i], DST_PTS)
                dst_image = warp(image, tform.inverse, output_shape=(self.resolution_inp, self.resolution_inp))
                dst_image = dst_image.transpose(2, 0, 1)
                dst_images += [dst_image]
            dst_images = np.stack(dst_images, axis=0)
            
            # imagenames = [imagename + f"{j:02d}" for j in range(dst_images.shape[0])]
            # imagepaths = [imagepath]* dst_images.shape[0]
            return {'image': torch.tensor(dst_images).float(),
                    # 'image_name': imagenames,
                    # 'image_path': imagepaths,
                    'tform': torch.as_tensor(tform.params).float(),
                    # 'original_image': torch.tensor(image.transpose(2,0,1)).float(),
                    }


def process_fn(dev_id, args, in_value, out_queue):
    import torch
    device = torch.device("cuda", dev_id)
    torch.cuda.set_device(device)
    with nullcontext("compacibility"):
        import numpy as np
        old_type = ["bool", "int", "float", "complex", "object", "unicode", "str"]
        for t in old_type:
            if not hasattr(np, t):
                setattr(np, t, getattr(np, f"{t}_"))

        import face_alignment
        if not hasattr(face_alignment.LandmarksType, "_2D"):
            face_alignment.LandmarksType._2D = face_alignment.LandmarksType.TWO_D

    with nullcontext("landmark"):
        model_root = os.path.join(ROOT)
        # from utils.face_2d import DlibLandmark, MediaPipeLandmark, GMMSkinSegmentation, PTNetRun
        sys.path.pop(0)
        # mp_lmk = MediaPipeLandmark()

        import mediapipe as mp

        face_tracker = mp.solutions.face_mesh.FaceMesh(
            static_image_mode=True,
            refine_landmarks=True,
            max_num_faces=1,
            min_detection_confidence=0.5
        )

        def mp_lmk(np_rgb):
            results = face_tracker.process(np_rgb)
            lmk_list = []
            for idx, res in enumerate(results.multi_face_landmarks):
                # print(res, dir(res))
                lmk  = [[p.x, p.y] for p in res.landmark]
                lmk_list.append(lmk)
            return np.array(lmk_list, dtype=np.float32)

        pairs = [(3, 248), (7, 249), (20, 461), (21, 251), (22, 252), (23, 253), (24, 254), (25, 255), (26, 256), (27, 257), (28, 258), (29, 259), (30, 260), (31, 261), (32, 262), (33, 263), (34, 264), (35, 265), (36, 266), (37, 267), (38, 268), (39, 269), (40, 270), (41, 271), (42, 272), (43, 273), (44, 274), (45, 275), (46, 276), (47, 277), (48, 278), (49, 279), (50, 280), (51, 281), (52, 282), (53, 283), (54, 284), (55, 285), (56, 286), (57, 287), (58, 288), (59, 392), (60, 290), (61, 291), (62, 292), (63, 293), (64, 294), (65, 295), (66, 296), (67, 297), (68, 298), (69, 299), (70, 300), (71, 301), (72, 302), (73, 303), (74, 304), (75, 305), (76, 306), (77, 307), (78, 308), (79, 309), (80, 310), (81, 311), (82, 317), (83, 313), (84, 314), (85, 315), (86, 316), (87, 312), (88, 310), (89, 319), (90, 320), (91, 321), (92, 322), (93, 323), (95, 415), (96, 325), (97, 326), (98, 460), (99, 328), (100, 329), (101, 330), (102, 331), (103, 332), (104, 333), (105, 334), (106, 335), (107, 336), (108, 337), (109, 338), (110, 339), (111, 340), (112, 341), (113, 342), (114, 343), (115, 344), (116, 345), (117, 346), (118, 347), (119, 348), (120, 349), (121, 350), (122, 351), (123, 352), (124, 353), (125, 354), (126, 355), (127, 356), (128, 357), (129, 331), (130, 359), (131, 360), (132, 361), (133, 362), (134, 363), (135, 364), (136, 365), (137, 366), (138, 367), (139, 368), (140, 369), (141, 370), (142, 371), (143, 372), (144, 373), (145, 374), (146, 375), (147, 376), (148, 377), (149, 378), (150, 379), (153, 380), (154, 381), (155, 382), (156, 383), (157, 384), (158, 385), (159, 386), (160, 387), (161, 388), (162, 389), (163, 390), (165, 391), (166, 438), (167, 393), (169, 394), (170, 395), (171, 396), (172, 397), (173, 398), (174, 399), (176, 400), (177, 401), (178, 311), (179, 403), (180, 404), (181, 405), (182, 406), (183, 407), (184, 408), (185, 409), (186, 410), (187, 411), (188, 412), (189, 413), (190, 414), (191, 415), (192, 416), (193, 417), (194, 418), (196, 419), (198, 420), (201, 421), (202, 422), (203, 423), (204, 424), (205, 425), (206, 426), (207, 427), (208, 428), (209, 429), (210, 430), (211, 431), (212, 432), (213, 433), (214, 434), (215, 435), (216, 436), (217, 437), (218, 438), (219, 439), (220, 440), (221, 441), (222, 442), (223, 443), (224, 444), (225, 445), (226, 446), (227, 447), (228, 448), (229, 449), (230, 450), (231, 451), (232, 452), (233, 453), (234, 454), (235, 439), (236, 456), (237, 457), (238, 461), (239, 457), (240, 289), (241, 354), (242, 354), (243, 463), (244, 464), (245, 465), (246, 466), (247, 467), (238, 250), (102, 278), (166, 289), (76, 292), (59, 305), (61, 306), (62, 308), (218, 309), (82, 312), (80, 318), (191, 324), (99, 326), (98, 327), (60, 328), (34, 356), (129, 358), (125, 370), (166, 392), (81, 402), (235, 455), (238, 458), (237, 459), (240, 460), (20, 462), (244, 465), (468, 473), (469, 476), (470, 475), (471, 474), (472, 477)]
        mid = [0, 1, 2, 4, 5, 6, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 94, 151, 152, 164, 168, 175, 195, 197, 199, 200]

        a_indx = np.array([p[0] for p in pairs])
        b_indx = np.array([p[1] for p in pairs])
        m_indx = np.array(mid)

        def mirror_lmk(lmk):
            new = np.zeros_like(lmk)

            new[..., 1] = lmk[..., 1]

            new[m_indx, 0] = 1 - lmk[m_indx, 0]
            new[a_indx, 0] = 1 - lmk[b_indx, 0]
            new[b_indx, 0] = 1 - lmk[a_indx, 0]
            return new

        MP_LMK68_INDEX = np.array([276, 282, 283, 285, 293, 295, 296, 300, 334, 336,  46,  52,  53,
            55,  63,  65,  66,  70, 105, 107, 249, 263, 362, 373, 374, 380,
            381, 382, 384, 385, 386, 387, 388, 390, 398, 466,   7,  33, 133,
            144, 145, 153, 154, 155, 157, 158, 159, 160, 161, 163, 173, 246,
            168,   6, 197, 195,   5,   4, 129,  98,  97,   2, 326, 327, 358,
              0,  13,  14,  17,  37,  39,  40,  61,  78,  80,  81,  82,  84,
            87,  88,  91,  95, 146, 178, 181, 185, 191, 267, 269, 270, 291,
            308, 310, 311, 312, 314, 317, 318, 321, 324, 375, 402, 405, 409,
            415])

    with nullcontext("emoca"):

        emoca_root = os.path.join(ROOT, "3rdparty", "emoca", "gdl_apps", "EMOCA")
        sys.path.insert(0, emoca_root)
        from gdl_apps.EMOCA.utils.load import load_model
        # from gdl.datasets.ImageTestDataset import TestData
        sys.path.pop(0)

        path_to_models = os.path.join(ROOT, "3rdparty", "emoca", "assets", "EMOCA", "models")
        model_name     = "EMOCA_v2_lr_mse_20"
        stage          = "detail"

        # emoca/gdl/models/DECA.py DecaModule
        emoca_model, conf = load_model(path_to_models, model_name, stage)
        emoca_model.to(device)
        emoca_model.eval()

        emoca_model = emoca_model
        emoca_test_data = EMOCATestData([os.devnull], face_detector="fan", max_detection=20)
    
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
            "flame_gaze_lmk_embedding_path":      os.path.join(ROOT, "Data", "FLAME2020", "gaze_landmark_embedding.npz"),
            "n_shape": 100,
            "n_exp":   50,
            "n_tex":   50,
            "tex_type": 'BFM',
        })

        flame      = FLAME(config).to(device)
        flametex   = FLAMETex(config).to(device)

        flame_tex_f = os.path.join(ROOT, "Data", "FLAME2020", "FLAME_texture.npz")
        flame_tex = np.load(flame_tex_f)
        uv     = torch.as_tensor(flame_tex["vt"]).float()
        uv_tri = torch.as_tensor(flame_tex["ft"].astype(np.int32)).long()
        uv[:, 1] = 1 - uv[:, 1]
        uv, uv_tri = uv.to(device), uv_tri.to(device)

        # flame_uv = object()
        # flame_uv.uv     = uv
        # flame_uv.uv_tri = uv_tri

        def render_nvdr(m2v, prj, ver, tri, uv, uv_tri, tex, H=512, W=512, FOV=90, locals={}):
            import nvdiffrast.torch as dr
            BS     = ver.size(0)
            device = ver.device
            drctx = locals.get("drctx", dr.RasterizeCudaContext(device=device))

            face_nrm = torch.cross(ver[:,tri[:,1]]-ver[:,tri[:,0]], ver[:,tri[:,2]]-ver[:,tri[:,0]], dim=-1)

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

            # cv2gl   = torch.diag(torch.as_tensor([1,-1,-1]).float()).to(face_nrm.device)  # fix opencv camera axis y z
            rot_nrm = F.normalize(face_nrm @ m2v_matx[:,:3,:3].transpose(1, 2), dim=-1)     # B,Nf,3
            # rot_nrm = rot_nrm @ cv2gl
            # print("nvdr face normal", face_nrm[0, :5])
            # print("nvdr rot", m2v_matx[:,:3,:3].transpose(1, 2))
            im_nrm  = torch.zeros((BS,H,W,3), device=rot_nrm.device)
            for bi in range(BS):
                tri_ind = rast[bi,...,-1]
                pix_msk = tri_ind >= 1           # H,W
                tri_ind = tri_ind[tri_ind>=1]-1  # N
                im_nrm[bi,pix_msk] = rot_nrm[bi, tri_ind.long()]

            z_view    = ver_v[...,2:3] - ver_v[...,2:3].min(-2, keepdim=True).values.detach()
            im_dep, _ = dr.interpolate(z_view.contiguous(), rast, tri_, rast_db)

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

    with nullcontext("TTA"):
        to_nparray = torchvision.transforms.Lambda(lambda pil_im: np.array(pil_im))
        keep_origin = torchvision.transforms.Compose([
            to_nparray
        ])
        gray_scale  = torchvision.transforms.Compose([
            torchvision.transforms.Grayscale(num_output_channels=3),
            to_nparray
        ])

        brightness_scale = [0.8, 1.2, 1.4]
        gamma_value      = [0.8, 1.2, 1.4]
        rgbspace_l       = [torchvision.transforms.Compose([
            torchvision.transforms.Lambda(
                # lambda pil_im, f=f : torchvision.transforms.functional.adjust_brightness(pil_im, f)
                lambda pil_im, f=f : torchvision.transforms.functional.adjust_gamma(pil_im, f)
            ),
            to_nparray
        ]) for f in gamma_value]
        
        tta_callables = [keep_origin, gray_scale] + rgbspace_l
        
        def get_tta_sample(im: Image):
            image_list = []
            for aug in tta_callables:
                image_list.append(aug(im))
            
            def aggr_fn(result):
                shapecode     = result["shapecode"]
                expcode       = result["expcode"]
                detailcode    = result["detailcode"]
                detailemocode = result["detailemocode"]
                posecode      = result["posecode"]
                texcode       = result["texcode"]
                lightcode     = result["lightcode"]
                cam           = result["cam"]

                original_code = result["original_code"]
                ori_shape     = original_code["shape"]
                ori_exp       = original_code["exp"]
                ori_pose      = original_code["pose"]
                ori_tex       = original_code["tex"]
                ori_light     = original_code["light"]
                ori_cam       = original_code["cam"]

                # simple aggr
                a_shapecode     = torch.mean(shapecode,     dim=0, keepdims=True)
                a_expcode       = torch.mean(expcode,       dim=0, keepdims=True)
                a_detailcode    = torch.mean(detailcode,    dim=0, keepdims=True)
                a_detailemocode = torch.mean(detailemocode, dim=0, keepdims=True)
                a_posecode      = torch.mean(posecode,      dim=0, keepdims=True)
                a_cam           = torch.mean(cam, dim=0, keepdims=True)
                # harder aggr, just using [0]
                a_texcode       = texcode[:1]
                a_lightcode     = lightcode[:1]

                # simple aggr
                a_ori_shape    = torch.mean(ori_shape,    dim=0, keepdims=True)
                a_ori_exp      = torch.mean(ori_exp,      dim=0, keepdims=True)
                a_ori_pose     = torch.mean(ori_pose,     dim=0, keepdims=True)
                a_ori_cam      = torch.mean(ori_cam, dim=0, keepdims=True)
                # harder aggr, just using [0]
                a_ori_tex      = ori_tex[:1]
                a_ori_light    = ori_light[:1]

                return {
                    "shapecode":     a_shapecode   ,
                    "expcode":       a_expcode     ,
                    "detailcode":    a_detailcode  ,
                    "detailemocode": a_detailemocode,
                    "posecode":      a_posecode    ,
                    "texcode":       a_texcode     ,
                    "lightcode":     a_lightcode   ,
                    "cam":           a_cam         ,

                    "original_code": {
                        "shape": a_ori_shape,
                        "exp":   a_ori_exp  ,
                        "pose":  a_ori_pose ,
                        "tex":   a_ori_tex  ,
                        "light": a_ori_light,
                        "cam":   a_ori_cam  ,
                    }
                }

            return image_list, aggr_fn
        
        def aggr_origin_and_mirror(aggr_ori, aggr_mir):
            return aggr_ori, aggr_mir
        
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
            pth_fpath      = os.path.join(args.outdir, f"{name}.pth")
            pth_fpath_flip = os.path.join(args.outdir, f"{name}_mirror.pth")

            if args.skip_if_exist and os.path.exists(pth_fpath):
                continue

            im_flip = f"{name}_mirror{ext}"
            
            with torch.no_grad():
                im      = Image.open(os.path.join(args.indir, im_path)).convert('RGB')
                im_flip = Image.open(os.path.join(args.indir, im_flip)).convert('RGB') if im_flip is not None else ImageOps.mirror(im)

                inpt_im_0, aggr_fn_0 = get_tta_sample(im)      # List[np.array[H,W,C]] Callable
                inpt_im_1, aggr_fn_1 = get_tta_sample(im_flip)

                img_total = inpt_im_0 + inpt_im_1

                def safe_mp_lmk(im):
                    try:
                        return mp_lmk(im)[0]
                    except Exception:
                        # import traceback
                        # print(traceback.format_exc())
                        return None

                # debugging
                if args.debug is True:
                    pass
                    RES = 128

                    inpt_total    = inpt_im_0 + inpt_im_1

                    row_of_origin = np.concatenate(inpt_total, axis=1) / 255.
                    img_origin = cv2.resize(row_of_origin.astype(np.float32), (RES*len(inpt_total), RES), interpolation=cv2.INTER_AREA)
                    # row_of_input  = torch.cat([d["image"].squeeze(0) for d in data_list], dim=2).permute(1, 2, 0).detach().cpu().numpy()

                    save_img = (255*(img_origin).clip(0, 1)).astype(np.uint8).copy()

                    cv2.imwrite(os.path.join(SAVE_ROOT, f"{name}.png"), cv2.cvtColor(save_img, cv2.COLOR_RGB2BGR))

                
                # infer mediapipe
                kpt_mp_0  = [safe_mp_lmk(im) for im in inpt_im_0] # List[np.array[478,2]]
                kpt_mp_1  = [safe_mp_lmk(im) for im in inpt_im_1]

                kpt_mp_all= [kpt for kpt in kpt_mp_0 if kpt is not None] + [mirror_lmk(kpt) for kpt in kpt_mp_1 if kpt is not None]

                kpt_aggr  = np.stack(kpt_mp_all, axis=0).mean(axis=0)  # 478,2

                # kpt_mp_0  = np.concatenate([kpt for kpt in kpt_mp_0 if kpt is not None], axis=0)   # T,K,2
                # kpt_mp_1  = np.concatenate([kpt for kpt in kpt_mp_1 if kpt is not None], axis=0)   # T,K,2

                # # aggr mediapipe
                # h, w = inpt_im_0[0].shape[:2]
                # # kpt_aggr  = 0.5*(kpt_mp_0.mean(axis=0) + kpt_mp_1_.mean(axis=0))
                # kpt_aggr_0= kpt_mp_0.mean(axis=0)  # use separete aggrated keypoint, due to semantic change
                # kpt_aggr_1= kpt_mp_1.mean(axis=0)
                # kpt_total = [ np.stack((w*kpt_aggr_0[:, 0], h*kpt_aggr_0[:, 1]), axis=-1) for _ in inpt_im_0 ] + \
                #             [ np.stack((w*kpt_aggr_1[:, 0], h*kpt_aggr_1[:, 1]), axis=-1) for _ in inpt_im_1 ]

                h, w = inpt_im_0[0].shape[:2]
                kpt_aggr_0= kpt_aggr
                kpt_aggr_1= mirror_lmk(kpt_aggr)
                kpt_total = [ np.stack((w*kpt_aggr_0[:, 0], h*kpt_aggr_0[:, 1]), axis=-1) for _ in inpt_im_0 ] + \
                            [ np.stack((w*kpt_aggr_1[:, 0], h*kpt_aggr_1[:, 1]), axis=-1) for _ in inpt_im_1 ]
                data_list = [emoca_test_data.process(im, kpt, kpt_type="mediapipe") for im, kpt in zip(img_total, kpt_total)]
                
                # images   = testdata['image'].to(device)
                images_0 = torch.stack([d["image"] for d in data_list[:len(inpt_im_0)]], dim=0).to(device)
                images_1 = torch.stack([d["image"] for d in data_list[len(inpt_im_0):]], dim=0).to(device)
                tforms_0 = torch.stack([d["tform"] for d in data_list[:len(inpt_im_0)]], dim=0).to(device)
                tforms_1 = torch.stack([d["tform"] for d in data_list[len(inpt_im_0):]], dim=0).to(device)

                codedict_0 = emoca_model.encode({"image": images_0}, training=False)
                codedict_1 = emoca_model.encode({"image": images_1}, training=False)

                result_0, result_1 = aggr_origin_and_mirror(aggr_fn_0(codedict_0), aggr_fn_1(codedict_1)) # len(origin), len(mirror)

                result, result_flip = {"original_code":{}}, {"original_code":{}}

                result_all = {}
                origin_all = {}
                for k in ["shapecode", "expcode", "posecode", "detailcode", "detailemocode",
                            "texcode", "lightcode", "cam"]:
                    result_all[k] = torch.cat([data[k] for data in [result_0]*len(inpt_im_0)+[result_1]*len(inpt_im_1)], dim=0)
                    # print(k, result_all[k].shape)

                    result[k]      = result_all[k][0].detach().cpu()
                    result_flip[k] = result_all[k][len(inpt_im_0)].detach().cpu()
                for k in ["shape", "exp", "pose", 
                            "tex", "light", "cam"]:
                    origin_all[k] = torch.cat([data[k] for data in [result_0["original_code"]]*len(inpt_im_0)+[result_1["original_code"]]*len(inpt_im_1)], dim=0)
                    # print(k, origin_all[k].shape)

                    result["original_code"][k]      = origin_all[k][0].detach().cpu()
                    result_flip["original_code"][k] = origin_all[k][len(inpt_im_0)].detach().cpu()
                result_all["original_code"] = origin_all

                result = {
                    "mp_lmk_478": torch.as_tensor(kpt_aggr_0),
                    "transform": tforms_0[0].cpu(),
                    "flamecode": result,
                }
                result_flip = {
                    "mp_lmk_478": torch.as_tensor(kpt_aggr_1),
                    "transform": tforms_1[0].cpu(),
                    "flamecode": result_flip,
                }

                if args.debug is True:
                    RES = 128

                    for k,v in codedict_0.items():
                        print("emoca", k, getattr(v, "shape", type(v)))
                    for k,v in codedict_0["original_code"].items():
                        print("original", k, getattr(v, "shape", type(v)))

                    inpt_total    = inpt_im_0 + inpt_im_1

                    for i in range(len(inpt_total)):
                        img_with_kpt  = inpt_total[i].copy()
                        for ki in range(len(kpt_total[i])):
                            x, y = (kpt_total[i][ki]).astype(np.int32).tolist()
                            cv2.circle(img_with_kpt, (x, y), 3, (220, 220, 20), -1)
                        inpt_total[i] = img_with_kpt

                    row_of_origin = np.concatenate(inpt_total, axis=1) / 255.
                    row_of_input  = torch.cat([d["image"].squeeze(0) for d in data_list], dim=2).permute(1, 2, 0).detach().cpu().numpy()

                    img_origin = cv2.resize(row_of_origin.astype(np.float32), (RES*len(inpt_total), RES), interpolation=cv2.INTER_AREA)
                    img_input  = cv2.resize(row_of_input.astype(np.float32),  (RES*len(inpt_total), RES), interpolation=cv2.INTER_AREA)

                    img_output = cv2.resize(row_of_input.astype(np.float32),  (RES*len(inpt_total), RES), interpolation=cv2.INTER_AREA)

                    BS = result_all["shapecode"].size(0)
                    
                    shape, exp = result_all["shapecode"], result_all["expcode"]
                    pose, eyep = result_all["posecode"], None
                    tex        = result_all["texcode"]
                    verts, landmarks2d, landmarks3d = flame(
                        shape_params=shape, expression_params=exp, 
                        pose_params=pose, eye_pose_params=eyep)
                    tex_map = flametex(tex)

                    print("tex_map", tex_map.shape)

                    tri = flame.faces_tensor

                    with nullcontext("draw orthogonal"):
                        model_pose = torch.zeros_like(result_all['posecode'])
                        model_pose[:, 3:] = result_all['posecode'][:, 3:]

                        orth_m2w = torch.eye(4).unsqueeze(0).repeat(BS, 1, 1).to(device)
                        orth_m2w[:, :3, :3] *= result_all["cam"][:, 0, None, None]
                        orth_m2w[:, 0, 3]    = result_all["cam"][:, 1] * result_all["cam"][:, 0]
                        orth_m2w[:, 1, 3]    = result_all["cam"][:, 2] * result_all["cam"][:, 0]

                        # orth_m2c = orth_w2c @ orth_m2w @ root_rot
                        orth_m2w = orth_m2w# @ root_rot

                        # images_1 = torch.stack([d["image"] for d in data_list[len(inpt_im_0):]], dim=0).to(device)

                        tform = torch.inverse(torch.cat([tforms_0, tforms_1]))  # R | t
                        t_2d  = torch.eye(3)[None, :, :].repeat(BS, 1, 1).to(device)
                        t_2d[:, 0, 0] = 0.5*224
                        t_2d[:, 1, 1] = 0.5*224
                        t_2d[:, 0, 2] = 0.5*224
                        t_2d[:, 1, 2] = 0.5*224

                        t_2d = tform@t_2d

                        orth_t3d_l = []
                        for bi in range(BS):
                            z_scale = torch.diagonal(t_2d[bi, :2,:2]).abs().mean()

                            orth_t3d = torch.diag(torch.as_tensor([z_scale/512, z_scale/512, z_scale/512, 1]))
                            orth_t3d[0,  3] = t_2d[bi, 0, 2]/512 - 0.5
                            orth_t3d[1,  3] = 0.5 - t_2d[bi, 1, 2]/512   # 3D y-axis is pointing up, 2D y is pointing down
                            orth_t3d_l.append(orth_t3d)
                        orth_t3d = torch.stack(orth_t3d_l, dim=0)
                        orth_m2w = orth_t3d.to(orth_m2w.device) @ orth_m2w

                        orth_w2v = torch.eye(4).to(orth_m2w.device)
                        orth_w2v[2,  3] = -3
                        orth_w2v = orth_w2v[None].expand(BS, -1, -1)

                        orth_m2v = orth_w2v @ orth_m2w

                        n, f = 0.001, 20
                        orth_ndc = torch.eye(4).to(device)
                        orth_ndc[0, 0] = 2
                        orth_ndc[1, 1] = -2
                        orth_ndc[2, 2] = -2/(f-n)  # [opencv camera] map 'f' to 1, instead of '-f' to 1
                        orth_ndc[2, 3] = -(f+n)/(f-n)
                        orth_ndc = orth_ndc[None].expand(BS, -1, -1)

                        image, alpha, normal, depth = render_nvdr(orth_m2v, orth_ndc, verts, tri, uv, uv_tri, tex_map.permute(0, 2, 3, 1))

                        shade = shading_SH2(normal.flatten(1, 2), result_all["lightcode"]).unflatten(1, (normal.shape[1:3]))
                        image = image*shade

                        image = image.transpose(0, 1).flatten(1, 2) # H, K*W, 3

                        row_of_output = image.detach().cpu().numpy()

                    img_output = cv2.resize(row_of_output.astype(np.float32),  (RES*len(inpt_total), RES), interpolation=cv2.INTER_AREA)

                    save_img = (255*np.concatenate([img_origin, img_input, img_output], axis=0).clip(0, 1)).astype(np.uint8).copy()

                    cv2.imwrite(os.path.join(SAVE_ROOT, f"{name}.png"), cv2.cvtColor(save_img, cv2.COLOR_RGB2BGR))

                    # os._exit(1)

            with open(pth_fpath, "wb") as f:
                torch.save(result, f)
            with open(pth_fpath_flip, "wb") as f:
                torch.save(result_flip, f)
        except Exception as ex:
            import traceback
            err = traceback.format_exc() + f"\nerror dealing with {im_path}"
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

    parser.add_argument('--debug', action="store_true")

    parser.add_argument('--start_index', type=int, default=None)
    parser.add_argument('--num_samples', type=int, default=None)
    args = parser.parse_args()

    main(args)