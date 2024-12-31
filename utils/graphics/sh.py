# COPY FROM https://github.com/LuniumLuk/AnalyticSHAreaLight/blob/main/utils/spherical_harmonics.py

import os
import sys
import math
import numpy as np
import torch

from scipy.special import factorial, lpmv

'''
Spherical harmonics
Reference:
[1] "An efficient representation for irradiance environment maps" by Ravi Ramamoorthi, Pat Hanrahan, SIGGRAPH 2001
[2] "Sparse Zonal Harmonic Factorization for Efficient SH Rotation" by Derek Nowrouzezahrai et al., SIGGRAPH 2012
[3] "spherical-harmonics" https://github.com/google/spherical-harmonics
[4] "SphericalHarmonics" https://github.com/chalmersgit/SphericalHarmonics
we replaced the hand written Legendre polynomial evaluation with scipy.special.lpmv
and added Al(l) evaluation for rendering
'''

def get_index(l, m):
    return l * (l + 1) + m

def eval_sh(l, m, phi, theta):
    assert l >= 0
    assert -l <= m and m <= l

    x, y, z = sph2cart(phi, theta)
    if l == 0:
        return 0.282095 * np.ones_like(x)
    elif l == 1:
        if m == -1:
            return -0.488603 * y
        elif m == 0:
            return 0.488603 * z
        elif m == 1:
            return -0.488603 * x
    elif l == 2:
        if m == -2:
            return 1.092548 * x * y
        elif m == -1:
            return -1.092548 * y * z
        elif m == 0:
            # in the original paper, this is 3z^2 - 1, which assumes that
            # the input cartesian coordinates are normalized
            return 0.315392 * (-x * x - y * y + 2 * z * z)
        elif m == 1:
            return -1.092548 * x * z
        elif m == 2:
            return 0.546274 * (x * x - y * y)

    kml = math.sqrt(
        (2.0 * l + 1) * factorial(l - abs(m)) /
        (4.0 * math.pi * factorial(l + abs(m)))
    )

    if m > 0:
        return math.sqrt(2.0) * kml * np.cos(m * phi) * lpmv(m, l, np.cos(theta))
    elif m < 0:
        return math.sqrt(2.0) * kml * np.sin(-m * phi) * lpmv(-m, l, np.cos(theta))
    else:
        return kml * lpmv(0, l, np.cos(theta))

def project_envmap(envmap, order=2, double_precision=True):
    assert order >= 0

    FLOAT = np.float64 if double_precision else np.float32

    h, w, c = envmap.shape
    envmap = envmap.astype(FLOAT)

    x, y = np.meshgrid(
        np.linspace(0, w-1, w, dtype=FLOAT),
        np.linspace(0, h-1, h, dtype=FLOAT),
    )
    x = (x + 0.5) / w
    y = (y + 0.5) / h

    phi, theta = equirectangular_project(x, y)

    weight = calc_solid_angle(theta, w, h)

    buffer = np.zeros((h, w, c, (order + 1) ** 2), dtype=FLOAT)
    for l in range(order + 1):
        for m in range(-l, l + 1):
            i = get_index(l, m)
            sh = eval_sh(l, m, phi, theta)
            buffer[..., i] += sh[..., None] * weight[..., None] * envmap

    coeffs = np.sum(buffer, axis=(0, 1))
    return coeffs.T

# reference: [3]
def near_by_margin(actual : float, expected : float):
    diff = abs(actual - expected)
    # 5 bits of error in mantissa (source of '32 *')
    return diff < 32 * sys.float_info.epsilon

def kronecker_delta(i : int, j : int):
    return 1.0 if i == j else 0.0

def get_centered_element(r : np.ndarray, i : int, j : int):
    offset = int((r.shape[0] - 1) / 2)
    return r[i + offset, j + offset]

def P(i : int, a : int, b : int, l : int, r : list[np.ndarray]):
    if b == l:
        return get_centered_element(r[1], i, 1) *          \
               get_centered_element(r[l - 1], a, l - 1) -  \
               get_centered_element(r[1], i, -1) *         \
               get_centered_element(r[l - 1], a, -l + 1)
    elif b == -l:
        return get_centered_element(r[1], i, 1) *          \
               get_centered_element(r[l - 1], a, -l + 1) + \
               get_centered_element(r[1], i, -1) *         \
               get_centered_element(r[l - 1], a, l - 1)
    else:
        return get_centered_element(r[1], i, 0) * get_centered_element(r[l - 1], a, b)

def U(m : int, n : int, l : int, r : list[np.ndarray]):
    return P(0, m, n, l, r)

def V(m : int, n : int, l : int, r : list[np.ndarray]):
    if m == 0:
        return P(1, 1, n, l, r) + P(-1, -1, n, l, r)
    elif m > 0:
        return P(1, m - 1, n, l, r) * math.sqrt(1 + kronecker_delta(m, 1)) - \
            P(-1, -m + 1, n, l, r) * (1 - kronecker_delta(m, 1))
    else:
        return P(1, m + 1, n, l, r) * (1 - kronecker_delta(m, -1)) + \
            P(-1, -m - 1, n, l, r) * math.sqrt(1 + kronecker_delta(m, -1))

def W(m : int, n : int, l : int, r : list[np.ndarray]):
    if (m == 0):
        return 0.0
    elif m > 0:
        return P(1, m + 1, n, l, r) + P(-1, -m - 1, n, l, r)
    else:
        return P(1, m - 1, n, l, r) - P(-1, -m + 1, n, l, r)

def cmpute_uvw_coeff(m : int, n : int, l : int):
    d = 1.0 if m == 0 else 0.0
    denom = 2.0 * l * (2.0 * l - 1) if abs(n) == l else (l + n) * (l - n)

    u = math.sqrt((l + m) * (l - m) / denom)
    v = 0.5 * math.sqrt((1 + d) * (l + abs(m) - 1.0) * (l + abs(m)) / denom) * (1 - 2 * d)
    w = -0.5 * math.sqrt((l - abs(m) - 1) * (l - abs(m)) / denom) * (1 - d)

    return u, v, w

def calculate_band_rotation(l : int, band_rotation : list[np.ndarray]):
    assert len(band_rotation) == l

    r = np.identity(2 * l + 1)

    for m in range(-l, l + 1):
        for n in range(-l, l + 1):
            u, v, w = cmpute_uvw_coeff(m, n, l)

            if not near_by_margin(u, 0.0):
                u *= U(m, n, l, band_rotation)
            if not near_by_margin(v, 0.0):
                v *= V(m, n, l, band_rotation)
            if not near_by_margin(w, 0.0):
                w *= W(m, n, l, band_rotation)

            r[m + l, n + l] = u + v + w

    return r

def rotate_single_channel(coeffs : np.ndarray, rotation):
    assert coeffs.ndim == 1

    order = int(math.sqrt(coeffs.shape[0])) - 1

    band_rotations = []

    # order 0 (first band) is simply the 1x1 identity matrix
    r = np.identity(1)
    band_rotations.append(r)

    mat = rotation
    r = np.identity(3)
    r[0, 0] =  mat[1, 1]
    r[0, 1] = -mat[1, 2]
    r[0, 2] =  mat[1, 0]
    r[1, 0] = -mat[2, 1]
    r[1, 1] =  mat[2, 2]
    r[1, 2] = -mat[2, 0]
    r[2, 0] =  mat[0, 1]
    r[2, 1] = -mat[0, 2]
    r[2, 2] =  mat[0, 0]
    band_rotations.append(r)

    for l in range(2, order + 1):
        # print(l)
        r = calculate_band_rotation(l, band_rotations)
        band_rotations.append(r)

    # apply rotations
    rotated_coeffs = np.zeros_like(coeffs)
    for l in range(order + 1):
        band_coeffs = np.zeros((2 * l + 1))

        for m in range(-l, l + 1):
            band_coeffs[m + l] = coeffs[get_index(l, m)]

        band_coeffs = np.matmul(band_rotations[l], band_coeffs)

        for m in range(-l, l + 1):
            rotated_coeffs[get_index(l, m)] = band_coeffs[m + l]
    
    return rotated_coeffs


def rotate_sh(coeffs, rotation):
    '''
        coeffs:   [B, N, 3]
        rotation: [B, 3, 3]
    '''

    BS = coeffs.shape[0]

    order = int(math.sqrt(coeffs.shape[1])) - 1

    all_mat = []
    for bi in range(BS):
        big_mat = np.zeros((coeffs.shape[1], coeffs.shape[1]), dtype=np.float32)

        band_rotations = []

        # order 0 (first band) is simply the 1x1 identity matrix
        r = np.identity(1)
        band_rotations.append(r)

        mat = rotation[bi].detach().cpu().numpy()
        r = np.identity(3)
        r[0, 0] =  mat[1, 1]
        r[0, 1] = -mat[1, 2]
        r[0, 2] =  mat[1, 0]
        r[1, 0] = -mat[2, 1]
        r[1, 1] =  mat[2, 2]
        r[1, 2] = -mat[2, 0]
        r[2, 0] =  mat[0, 1]
        r[2, 1] = -mat[0, 2]
        r[2, 2] =  mat[0, 0]
        band_rotations.append(r)

        for l in range(2, order + 1):
            r = calculate_band_rotation(l, band_rotations)
            band_rotations.append(r)

        base = 0
        for l in range(order + 1):
            # print(base, 2*l+1, band_rotations[l].shape)
            big_mat[base:base+2*l+1, base:base+2*l+1] = band_rotations[l]

            base += 2*l+1
        
        # print(big_mat)
        
        all_mat.append(big_mat)
    
    all_mat = torch.as_tensor(all_mat, dtype=torch.float32, device=coeffs.device)
        
    rotated_coeffs = torch.bmm(all_mat, coeffs)
    return rotated_coeffs

def car2sph(x, y, z):
    r = np.sqrt(x * x + y * y + z * z)
    phi = np.arctan2(y, x)
    theta = np.arccos(z / r)
    return phi, theta

def sph2cart(phi, theta):
    x = np.sin(theta) * np.cos(phi)
    y = np.sin(theta) * np.sin(phi)
    z = np.cos(theta)
    return x, y, z

def equirectangular_project(x, y):
    return (1.0 - x) * (2.0 * np.pi), y * np.pi

def calc_solid_angle(theta, w, h):
    pixel_size_x = 2.0 * np.pi / w
    pixel_size_y = np.pi / h

    # sin(theta) * d(theta) * d(phi) -> -d(cos(theta)) * d(phi)
    return pixel_size_x * abs(np.cos(theta - (pixel_size_y / 2.0)) - np.cos(theta + (pixel_size_y / 2.0)))


if __name__ == '__main__':

    # SH rotation

    coeffs = np.random.rand(9)

    r = np.deg2rad(30)
    rot = np.array([
        [ np.cos(r), 0, np.sin(r),    0],
        [         0, 1,         0,    0],
        [-np.sin(r), 0, np.cos(r), -1.2],
        [         0, 0,         0,    1],
    ])

    r_coeffs = rotate_single_channel(coeffs, rot)

    print(coeffs)
    print(r_coeffs)