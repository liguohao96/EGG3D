import torch


def chunk_fn(fn, chunk_size, input_tensor_list, *args, **kwargs):
    input_chunk_list = [torch.split(t, chunk_size, dim=0) for t in input_tensor_list]

    all_ret = []
    for i, input_tuple in enumerate(zip(*input_chunk_list)):
        r = fn(*input_tuple, *args, **kwargs)
        all_ret.append(r)
    
    r = all_ret[0]
    if len(all_ret) == 1:
        return r

    if isinstance(r, (list, tuple)):
        # ret = list(map(lambda i: torch.cat([ one[i] for one in all_ret], dim=0), range(len(r)) ))
        ret = [ torch.cat([one[i] for one in all_ret], dim=0) for i in range(len(r)) ]
    elif isinstance(r, (dict,)):
        # ret = dict(map(lambda k: (k,torch.cat([ one[k] for one in all_ret], dim=0)), r.keys() ))
        ret = { k: torch.cat([one[k] for one in all_ret], dim=0) for k in r.keys() }
    else:
        ret = torch.cat(all_ret, dim=0)
    return ret

def near_far_from_sphere(ray_origins: torch.Tensor, ray_directions: torch.Tensor, r = 1.0, keepdim=True):
    """
    NOTE: modified from https://github.com/Totoro97/NeuS
    ray_origins: camera center's coordinate
    ray_directions: camera rays' directions. already normalized.
    """
    # rayso_norm_square = torch.sum(ray_origins**2, dim=-1, keepdim=True)
    # NOTE: (minus) the length of the line projected from [the line from camera to sphere center] to [the line of camera rays]
    ray_cam_dot = torch.sum(ray_origins * ray_directions, dim=-1, keepdim=keepdim)
    mid = -ray_cam_dot
    # NOTE: a convservative approximation of the half chord length from ray intersections with the sphere.
    #       all half chord length < r
    near = mid - r
    far = mid + r
    
    near = near.clamp_min(0.0)
    far = far.clamp_min(r)  # NOTE: instead of clamp_min(0.0), just some trick.
    
    return torch.stack([near, far], dim=-1)

def near_far_from_value(ray_origins, ray_directions, near, far):
    """
    NOTE: modified from https://github.com/Totoro97/NeuS
    ray_origins: camera center's coordinate
    ray_directions: camera rays' directions. already normalized.
    """
    # rayso_norm_square = torch.sum(ray_origins**2, dim=-1, keepdim=True)
    # NOTE: (minus) the length of the line projected from [the line from camera to sphere center] to [the line of camera rays]
    Nr = max(ray_origins.size(0), ray_directions.size(0))

    device = ray_origins.device

    near = torch.full((Nr,), near, device=device)
    far  = torch.full((Nr,), far , device=device)
    
    return torch.stack([near, far], dim=-1)

def near_far_from_aabb(ray_origins, ray_directions, aabb):
    Nr = max(ray_origins.size(0), ray_directions.size(0))

    device = ray_origins.device

    dir_fraction = 1.0 / (ray_directions + 1e-6)

    rate_a = (aabb[1] - ray_origins) * dir_fraction
    rate_b = (aabb[0] - ray_origins) * dir_fraction

    near = torch.minimum(rate_a, rate_b).amax(-1)
    far  = torch.maximum(rate_a, rate_b).amin(-1)

    near = torch.clamp(near, min=0)
    far  = torch.maximum(far, near + 1e-6)

    return torch.stack([near, far], dim=-1)

def near_far_from_aabb_old(ray_origins, ray_directions, aabb):
    """
    NOTE: modified from https://github.com/Totoro97/NeuS
    ray_origins: camera center's coordinate
    ray_directions: camera rays' directions. already normalized.
    """
    # rayso_norm_square = torch.sum(ray_origins**2, dim=-1, keepdim=True)
    # NOTE: (minus) the length of the line projected from [the line from camera to sphere center] to [the line of camera rays]
    Nr = max(ray_origins.size(0), ray_directions.size(0))

    device = ray_origins.device

    dir_fraction = 1.0 / (ray_directions + 1e-6)

    # x
    t1 = (aabb[0, 0] - ray_origins[:, 0:1]) * dir_fraction[:, 0:1]
    t2 = (aabb[1, 0] - ray_origins[:, 0:1]) * dir_fraction[:, 0:1]
    # y
    t3 = (aabb[0, 1] - ray_origins[:, 1:2]) * dir_fraction[:, 1:2]
    t4 = (aabb[1, 1] - ray_origins[:, 1:2]) * dir_fraction[:, 1:2]
    # z
    t5 = (aabb[0, 2] - ray_origins[:, 2:3]) * dir_fraction[:, 2:3]
    t6 = (aabb[1, 2] - ray_origins[:, 2:3]) * dir_fraction[:, 2:3]

    near = torch.max(
        torch.cat([torch.minimum(t1, t2), torch.minimum(t3, t4), torch.minimum(t5, t6)], dim=1), dim=1
    ).values
    far = torch.min(
        torch.cat([torch.maximum(t1, t2), torch.maximum(t3, t4), torch.maximum(t5, t6)], dim=1), dim=1
    ).values

    # clamp to near plane
    near = torch.clamp(near, min=0)
    far  = torch.maximum(far, near + 1e-6)

    return torch.stack([near, far], dim=-1)