import os
import sys
import numpy as np
import torch
import torch.nn.functional as F

from tqdm import tqdm

import pickle
from easydict import EasyDict

from contextlib import nullcontext

ROOT = os.path.abspath(os.path.dirname(__file__))

sys.path.insert(0, ROOT)
from utils import device_control
from utils import chunk_fn
sys.path.pop(0)

class FeatureStats:
    def __init__(self, capture_all=False, capture_mean_cov=False, max_items=None):
        self.capture_all = capture_all
        self.capture_mean_cov = capture_mean_cov
        self.max_items = max_items
        self.num_items = 0
        self.num_features = None
        self.all_features = None
        self.raw_mean = None
        self.raw_cov = None

    def set_num_features(self, num_features):
        if self.num_features is not None:
            assert num_features == self.num_features
        else:
            self.num_features = num_features
            self.all_features = []
            self.raw_mean = np.zeros([num_features], dtype=np.float64)
            self.raw_cov = np.zeros([num_features, num_features], dtype=np.float64)

    def is_full(self):
        return (self.max_items is not None) and (self.num_items >= self.max_items)

    def append(self, x):
        x = np.asarray(x, dtype=np.float32)
        assert x.ndim == 2
        if (self.max_items is not None) and (self.num_items + x.shape[0] > self.max_items):
            if self.num_items >= self.max_items:
                return
            x = x[:self.max_items - self.num_items]

        self.set_num_features(x.shape[1])
        self.num_items += x.shape[0]
        if self.capture_all:
            self.all_features.append(x)
        if self.capture_mean_cov:
            x64 = x.astype(np.float64)
            self.raw_mean += x64.sum(axis=0)
            self.raw_cov += x64.T @ x64

    def get_all(self):
        assert self.capture_all
        return np.concatenate(self.all_features, axis=0)

    def get_mean_cov(self):
        assert self.capture_mean_cov
        mean = self.raw_mean / self.num_items
        cov = self.raw_cov / self.num_items
        cov = cov - np.outer(mean, mean)
        return mean, cov

    def save(self, pkl_file):
        with open(pkl_file, 'wb') as f:
            pickle.dump(self.__dict__, f)

    @staticmethod
    def load(pkl_file):
        with open(pkl_file, 'rb') as f:
            s = EasyDict(pickle.load(f))
        obj = FeatureStats(capture_all=s.capture_all, max_items=s.max_items)
        obj.__dict__.update(s)
        return obj

@torch.no_grad()
def get_fid(data_loader, gen_sample_fn, device, num_generate=50*1000, num_image=None, H=3, W=4, K=5, verbose=False,
    locals={}):
    import pickle
    import hashlib
    import scipy
    from easydict import EasyDict
    from scipy.stats import multivariate_normal

    if "inception-v3" not in locals:
        with open(os.path.join(ROOT, "Data", "inception-2015-12-05.pkl"), "rb") as f:
            model = pickle.load(f).eval().to(device)
        
        locals["inception-v3"] = model
    
    device_control.sync_ddp()
    model = locals["inception-v3"]

    @torch.no_grad()
    def get_pred(images):
        pred = model((255*images.permute(0,3,1,2)).clamp(0, 255).to(torch.uint8), return_features=True)

        if device_control.ddp_enabled:
            pred = device_control.gather(pred)

        pred = pred.cpu().numpy()
        return pred
    
    def compute_fid(mu_real, sigma_real, mu_gen, sigma_gen):
        m = np.square(mu_gen - mu_real).sum()
        s, _ = scipy.linalg.sqrtm(np.dot(sigma_gen, sigma_real), disp=False) # pylint: disable=no-member
        fid = np.real(m + np.trace(sigma_gen + sigma_real - s * 2))
        return float(fid), (m, np.real(np.trace(sigma_gen + sigma_real - s * 2)))

    img_fs = FeatureStats(max_items=num_image,    capture_all=False, capture_mean_cov=True)
    gen_fs = FeatureStats(max_items=num_generate, capture_all=False, capture_mean_cov=True)
    img_list, gen_list, gen_pred = [], [], []

    dataset = data_loader.dataset
    assert dataset is not None
    md5     = hashlib.md5((repr(dataset) + os.path.join(ROOT, "Data", "inception-2015-12-05.pkl")).encode('utf-8'))
    result_tag  = f'{md5.hexdigest()}-{num_image}'
    result_file = os.path.join(ROOT, "Data", f"fid50k_full-{result_tag}.pkl")

    if verbose:
        print((repr(dataset) + os.path.join(ROOT, "Data", "inception-2015-12-05.pkl")).encode('utf-8'))
        print(f"{result_file} exist:{os.path.exists(result_file)}")
        print(f"RANK: {device_control.get_rank()}")

    if result_tag not in locals:
        flag = os.path.exists(result_file)
        if not flag:
            for i, data in enumerate(tqdm(data_loader, disable=verbose is False)):
                images    = data["image"].to(device)
                img_feats = get_pred(images)
                img_fs.append(img_feats)

                if img_fs.is_full():
                    break
            if device_control.get_rank() == 0:
                os.makedirs(os.path.dirname(result_file), exist_ok=True)
                img_fs.save(result_file)

                if verbose:
                    print(f"after saving, {result_file} exist:{os.path.exists(result_file)}")
        else:
            img_fs = FeatureStats.load(result_file)

        if verbose:
            print(f"received: {img_fs.num_items} | total: {len(dataset)}")

        locals[result_tag] = img_fs

    img_fs = locals[result_tag]

    device_control.sync_ddp()

    num_current = 0
    pbar = tqdm(total=num_generate, disable=not verbose)
    # for i, data in enumerate(data_loader):
    iter_loader = iter(data_loader)
    while True:
        try:
            data      = next(iter_loader)
        except StopIteration:
            iter_loader = iter(data_loader)
            data        = next(iter_loader)
        images    = data["image"]
        
        gen_images = gen_sample_fn(data)
        gen_feats  = get_pred(gen_images)  # already joined across device, return value is np.ndarray[Ngpu*Nbatch, Nfeature]

        if len(img_list) < H*W * (K**2):
            if device_control.ddp_enabled:
                images     = device_control.gather(images.to(device))  # ddp gather
                gen_images = device_control.gather(gen_images)

            img_list.extend(images.detach().cpu()[:gen_images.size(0)])
            gen_list.extend(gen_images.detach().cpu())
            gen_pred.extend(gen_feats)
        
        gen_fs.append(gen_feats)
        num_current += gen_feats.shape[0]

        pbar.update(gen_feats.shape[0])
        if num_current >= num_generate:
            break
    pbar.close()

    assert gen_fs.is_full()

    # gen_pred = gen_pred[:num_total]

    mu_img, sigma_img = img_fs.get_mean_cov()
    mu_gen, sigma_gen = gen_fs.get_mean_cov()

    fid, fid_detail = compute_fid(mu_img, sigma_img, mu_gen, sigma_gen)

    img_list, gen_list = map(lambda x:torch.stack(x, dim=0)[:H*W*K*K], [img_list, gen_list]) # H*K,W*K,h,w,c

    gen_list_pred_f = np.asarray(gen_pred[:H*W*K*K])  # K, c

    real_multivar_g = multivariate_normal(mu_img, np.real(sigma_img), allow_singular=True)
    gen_list_score  = real_multivar_g.logpdf(gen_list_pred_f)
    if verbose:
        print(gen_list_score)
        print(np.isfinite(gen_list_pred_f).sum()/np.prod(gen_list_pred_f.shape))
        print("img", np.isfinite(mu_img).sum(), np.isfinite(sigma_img).sum())
        print("gen", np.isfinite(mu_gen).sum(), np.isfinite(sigma_gen).sum())
        print("fid", fid)
    gen_list_argsort= np.argsort(gen_list_score)
    gen_list_sorted = gen_list[torch.as_tensor(gen_list_argsort)]
    gen_list_sorted = gen_list_sorted.unflatten(0, (H*K, W*K)).transpose(1, 2).flatten(0, 1).flatten(1, 2)

    img_list, gen_list = map(lambda x:x.unflatten(0, (H*K, W*K)), [img_list, gen_list]) # H*K,W*K,h,w,c
    img_list, gen_list = map(lambda x:x.transpose(1, 2).flatten(0, 1).flatten(1, 2), [img_list, gen_list])  # H*K*h,W*K*w,c

    img = (255*img_list.detach().cpu().numpy()).astype(np.uint8).copy()
    gen = (255*gen_list.detach().cpu().numpy()).astype(np.uint8).copy()

    gen_sorted = (255*gen_list_sorted.detach().cpu().numpy()).astype(np.uint8).copy()

    return fid, (img, gen, gen_sorted)

@torch.no_grad()
def get_ID(dataset, get_label, gen_sample_fn, device, dim_z, num_test=1024, seed=210, N=10, verbose=False, metric_log_buff=sys.stdout, id_align=False,
    locals={}):
    '''
    compute ID metric with 'Efficient Geometry-aware 3D Generative Adversarial Networks' based on decription of 'Multi-view consistency'
    details at https://nvlabs.github.io/eg3d/media/eg3d.pdf
    '''

    NUM_ID  = num_test
    NUM_NEW = NUM_ID*2

    ARCF_NET = "r50"
    ID_ALIGN = id_align

    rng = np.random.default_rng(seed=seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)

    cond_sample0_i = rng.choice(np.arange(len(dataset)), (NUM_ID),  replace=False)
    cond_sample1_i = rng.choice(np.arange(len(dataset)), (NUM_NEW), replace=False)

    cond_sample_id  = torch.utils.data.default_collate([get_label(dataset, i) for i in cond_sample0_i])
    # cond_data_new   = [get_label(dataset, i) for i in cond_sample1_i]

    sys.path.insert(0, ROOT)
    from face_predict import get_deca_predictor, get_arcface_predictor, get_d3fr_predictor
    sys.path.pop(0)
    arcf_predictor = get_arcface_predictor(device, ARCF_NET, align=ID_ALIGN)
    arcf_predictor.debug = True
    print(f"2d face recognition align: {ID_ALIGN}", file=metric_log_buff)
    print(f"2d face recognition with {arcf_predictor.__class__}", file=metric_log_buff)

    id_image, id_debug = [], []
    id_similarity = []
    keys = ["m2v", "ndc"]
    print(f"ID consistency keys: {keys} #IDx#Other={NUM_ID}x2", file=metric_log_buff)
    for i in tqdm(range(NUM_ID), desc="ID"):
        noise      = torch.randn((1, dim_z), device=device).expand(2, -1)

        data_other = torch.utils.data.default_collate([get_label(dataset, si) for si in cond_sample1_i[i*2:i*2+2]])

        def get_code(key):
            if key in keys:
                code = data_other[key].to(device)
            else:
                # code = cond_sample_id[key][i:i+1].expand(2, -1).to(device)
                code = torch.stack([ cond_sample_id[key][i] ]*2, dim=0).to(device)
            return code

        m2v        = get_code("m2v")
        ndc        = get_code("ndc")
        id_code    = get_code("shape")
        exp_code   = get_code("exp")
        pose_code  = get_code("pose")
        eyep_code  = get_code("eye_pose")
        light_code = get_code("light")

        image, lmk68 = gen_sample_fn(noise, m2v, ndc, id_code, exp_code, pose_code, eyep_code, light_code)

        feat = arcf_predictor(image, lmk68) # 2, C

        id_sim = F.cosine_similarity(feat[0], feat[1], dim=0)*0.5+0.5 # [-1, 1] -> [0, 1]

        id_similarity.append(id_sim.item())

        if i < N:
            log_image = image.transpose(0, 1).flatten(1, 2) # N,H,W,3 -> H,N*W,3
            log_image = (255*log_image.detach().cpu().numpy()).astype(np.uint8)

            # H, W  = image.size(1), image.size(2)
            # lmknp = lmk68.detach().cpu().numpy()
            # for bi in range(image.size(0)):
            #     for xy in lmknp[bi]:
            #         x, y = int(xy[0]*W+bi*W), int(xy[1]*H)
            #         cv2.circle(log_image, (x, y), 3, (220, 220, 20), -1)
            id_image.append(log_image)

            dbg_image = arcf_predictor.debug_cache["input_image"].permute(2, 0, 3, 1).flatten(1, 2)
            dbg_image = (255*dbg_image.detach().cpu().numpy()).astype(np.uint8)
            id_debug.append(dbg_image)
        elif i == N:
            im_origin = np.concatenate(id_image, axis=0)
            im_debug = np.concatenate(id_debug, axis=0)

            arcf_predictor.debug = False

    arcf_predictor.debug = False

    del cond_sample_id
    del arcf_predictor

    id_score = np.mean(id_similarity)

    return id_score, (im_origin, im_debug)

@torch.no_grad()
def get_AXD(dataset, get_label, gen_sample_fn, device, dim_z, num_test=500, seed=210, N=10, verbose=False, metric_log_buff=sys.stdout,
    estimator="d3fr", 
    locals={}):

    NUM_ID  = num_test
    NUM_NEW = NUM_ID*20

    rng = np.random.default_rng(seed=seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)

    cond_sample0_i = rng.choice(np.arange(len(dataset)), (NUM_ID),  replace=False)
    cond_sample1_i = rng.choice(np.arange(len(dataset)), (NUM_NEW), replace=False)

    cond_sample_id  = torch.utils.data.default_collate([get_label(dataset, i) for i in cond_sample0_i])

    sys.path.insert(0, ROOT)
    from face_predict import get_deca_predictor, get_arcface_predictor, get_d3fr_predictor
    sys.path.pop(0)

    if estimator == "d3fr":
        frec_predictor = get_d3fr_predictor(device, align=False)
    elif estimator == "deca":
        frec_predictor = get_deca_predictor(device, align=False)
    frec_predictor.debug = True
    print(f"3d face reconstruction with {frec_predictor.__class__}", file=metric_log_buff)

    with nullcontext("AED_APD"):
        gen_image, net_debug = [], []

        fake_exps, fake_poses = [], []
        real_exps, real_poses = [], []

        keys = ["m2v", "exp", "pose"]
        print(f"AED_APD keys: {keys} #IDx#Other={NUM_ID}x20", file=metric_log_buff)
        for i in tqdm(range(NUM_ID), desc="AED_APD"):
            noise      = torch.randn((1, dim_z), device=device).expand(20, -1)

            data_other = torch.utils.data.default_collate([get_label(dataset, si) for si in cond_sample1_i[i*20:i*20+20]])

            def get_code(key):
                if key in keys:
                    code = data_other[key].to(device)
                else:
                    # code = cond_sample_id[key][i:i+1].expand(2, -1).to(device)
                    code = torch.stack([ cond_sample_id[key][i] ]*20, dim=0).to(device)
                return code

            m2v        = get_code("m2v")
            ndc        = get_code("ndc")
            id_code    = get_code("shape")
            exp_code   = get_code("exp")
            pose_code  = get_code("pose")
            eyep_code  = get_code("eye_pose")
            light_code = get_code("light")

            image, lmk68 = gen_sample_fn(noise, m2v, ndc, id_code, exp_code, pose_code, eyep_code, light_code)

            cond_image = data_other["image"].to(device) # N,H,W,3

            cond_pred = frec_predictor(cond_image)
            fake_pred = frec_predictor(image)

            real_exps.append(cond_pred["exp"].detach().cpu())
            fake_exps.append(fake_pred["exp"].detach().cpu())
            real_poses.append(cond_pred["pose"].detach().cpu())
            fake_poses.append(fake_pred["pose"].detach().cpu())

            if i < N:
                log_image  = torch.cat([cond_image.detach().cpu(), image.detach().cpu()], dim=2).flatten(0, 1) # N*H,2*W,3
                log_image  = (255*log_image.numpy()).astype(np.uint8)

                dbg_image = []
                frec_predictor(cond_image)
                dbg_image.append(frec_predictor.debug_cache["input_image"].permute(0, 2, 3, 1)) # b,h,w,3

                frec_predictor(image)
                dbg_image.append(frec_predictor.debug_cache["input_image"].permute(0, 2, 3, 1)) # b,h,w,3
                
                dbg_image = torch.cat(dbg_image, dim=2).flatten(0, 1)

                dbg_image = (255*dbg_image.detach().cpu().numpy()).astype(np.uint8)

                gen_image.append(log_image)
                net_debug.append(dbg_image)
            elif i == N:

                im_origin = np.concatenate(gen_image, axis=1)
                im_debug  = np.concatenate(net_debug, axis=1)

                frec_predictor.debug = False

    with nullcontext("AID"):
        gen_image, net_debug = [], []
        fake_light, real_light = [], []

        frec_predictor.debug = True

        num_other = NUM_NEW // NUM_ID

        # keys = []
        keys = ["light"]
        print(f"AXD keys: {keys} #IDx#Other={NUM_ID}", file=metric_log_buff)
        for i in tqdm(range(NUM_ID), desc="AID"):
            noise      = torch.randn((1, dim_z), device=device).expand(num_other, -1)

            # data_other = {}
            data_other = torch.utils.data.default_collate([get_label(dataset, si) for si in cond_sample1_i[i*num_other:i*num_other+num_other]])

            def get_code(key):
                if key in keys:
                    code = data_other[key].to(device)
                else:
                    # code = cond_sample_id[key][i:i+1].expand(2, -1).to(device)
                    code = torch.stack([ cond_sample_id[key][i] ]*num_other, dim=0).to(device)
                return code

            m2v        = get_code("m2v")
            ndc        = get_code("ndc")
            id_code    = get_code("shape")
            exp_code   = get_code("exp")
            pose_code  = get_code("pose")
            eyep_code  = get_code("eye_pose")
            light_code = get_code("light")

            image, lmk68 = chunk_fn(gen_sample_fn, 4, [noise, m2v, ndc, id_code, exp_code, pose_code, eyep_code, light_code])

            # cond_image = cond_sample_id["image"][i:i+1].to(device) # N,H,W,3
            cond_image = data_other["image"].to(device) # N,H,W,3

            cond_pred = frec_predictor(cond_image)
            fake_pred = frec_predictor(image)

            real_light.append(cond_pred["light"].flatten(1).detach().cpu())
            fake_light.append(fake_pred["light"].flatten(1).detach().cpu())

            if i < N:
                log_image  = torch.cat([cond_image.detach().cpu(), image.detach().cpu()], dim=2).flatten(0, 1) # N*H,2*W,3
                log_image  = (255*log_image.numpy()).astype(np.uint8)

                dbg_image = []
                frec_predictor(cond_image)
                dbg_image.append(frec_predictor.debug_cache["input_image"].permute(0, 2, 3, 1)) # b,h,w,3

                frec_predictor(image)
                dbg_image.append(frec_predictor.debug_cache["input_image"].permute(0, 2, 3, 1)) # b,h,w,3
                
                dbg_image = torch.cat(dbg_image, dim=2).flatten(0, 1)

                dbg_image = (255*dbg_image.detach().cpu().numpy()).astype(np.uint8)

                gen_image.append(log_image)
                net_debug.append(dbg_image)
            elif i == N:

                im_origin = np.concatenate([im_origin]+gen_image, axis=1)
                im_debug  = np.concatenate([im_debug ]+net_debug, axis=1)

                frec_predictor.debug = False

    del cond_sample_id
    del frec_predictor

    fake_exps, fake_poses = np.concatenate(fake_exps, axis=0), np.concatenate(fake_poses, axis=0)
    real_exps, real_poses = np.concatenate(real_exps, axis=0), np.concatenate(real_poses, axis=0)

    fake_light, real_light = np.concatenate(fake_light, axis=0), np.concatenate(real_light, axis=0)

    if estimator == "deca":
        import math
        import sklearn.metrics
        AED = math.sqrt(sklearn.metrics.mean_squared_error(real_exps, fake_exps))
        APD = math.sqrt(sklearn.metrics.mean_squared_error(real_poses[:, :3], fake_poses[:, :3]))
        AID = math.sqrt(sklearn.metrics.mean_squared_error(real_light, fake_light))
        print(f"AXD compute math.sqrt(sklearn.metrics.mean_squared_error)", file=metric_log_buff)
    else:
        AED = F.mse_loss(torch.as_tensor(fake_exps),  torch.as_tensor(real_exps) ).item()
        APD = F.mse_loss(torch.as_tensor(fake_poses[:, :3]), torch.as_tensor(real_poses[:, :3])).item()
        AID = F.mse_loss(torch.as_tensor(fake_light), torch.as_tensor(real_light)).item()
        print(f"AXD compute F.mse_loss", file=metric_log_buff)

    return (AED, APD, AID), (im_origin, im_debug)
            