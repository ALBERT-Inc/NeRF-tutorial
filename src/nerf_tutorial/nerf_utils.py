# codes are almost from https://github.com/ALBERT-Inc/blog_nerf/blob/master/NeRF.ipynb

import math
import torch
import numpy as np
import pandas as pd
from pyntcloud import  PyntCloud


def camera_parameters_to_rays(
    w, h, cx, cy, fx, fy, pose, device=None, **kwargs):
    """transform image pixels to camera origins and rays.
    this function do the same process
    with Intrinsic class and Extrinsic class.

    Args:
        w (int): width of rendering image.
        h (int): height of rendering image.
        cx (float): x value of image center for rendering.
        cx (float): y value of image center for rendering.
        fx (float): x value of focal length for rendering.
        fx (float): y value of focal length for rendering.
        pose (torch.tensor): pose matrix for rendering.
                this takes (1, 4, 4) shape.
    Returns:
        o (torch.tensor): camera origins in world coordinate.
                this takes (W*H, 3) shape.
        d (torch.tensor): camera directions in world coordinate.
                this takes (W*H, 3) shape.
    """

    # intrinsic part
    v, u = np.mgrid[:h, :w].astype(np.float32)
    u = torch.tensor(u, dtype=torch.float32, device=device)
    v = torch.tensor(v, dtype=torch.float32, device=device)

    _x = (u - cx) / fx
    _y = (v - cy) / fy
    _z = torch.ones_like(_x, device=device)
    _w = torch.ones_like(_x, device=device)

    xyzw = torch.stack([_x, _y, _z, _w], dim=2)
    xyzw = xyzw.reshape(-1, 4)

    # extrinsic part
    _o = np.zeros((len(xyzw), 4), dtype=np.float32)
    _o[:, 3] = 1.
    _o = torch.tensor(_o, dtype=torch.float32, device=device)

    pose = pose.repeat(len(xyzw), 1, 1)
    d = torch.bmm(pose, xyzw[..., None])[:, :, 0][:, :3]
    o = torch.bmm(pose, _o[..., None])[:, :, 0][:, :3]
    d = d - o
    d = d / torch.norm(d, dim=1, keepdim=True)
    return o, d


@torch.no_grad()
def render_nerf(
    nerf, camera_parameters, bsz_eval=1024, only_coarse=False):
    """rendering function with nerf.

    Args:
        nerf (torch.nn.Module): nerf model.
        camera_parameters (dict): dictionary of camera parameters.
        bsz_eval (int): batch size for nerf inference.
        only_coarse (bool): only_coarse (bool): whether to use 
                            coarse network for rendering.
    Returns:
        Cs (numpy.array): rendered image array in (H, W, 3) shape.
    """
    
    Cs = []

    nerf.eval()
    o, d = camera_parameters_to_rays(**camera_parameters)

    num_data = len(o)
    if num_data % bsz_eval == 0:
        num_iter = num_data // bsz_eval
    else:
        num_iter = num_data // bsz_eval + 1

    for i in range(num_iter):
        start = i * bsz_eval
        end = min((i+1)*bsz_eval, num_data)

        C = nerf(
            o[start:end], d[start:end], only_coarse=only_coarse)[-1]
        Cs.append(C)

    Cs = torch.clamp(torch.cat(Cs), 0., 1.)
    Cs = Cs.reshape(
        camera_parameters["h"], camera_parameters["w"], 3)
    return Cs.cpu().detach().numpy()


def _gen_3d_grid(size, xlim=(-1, 1), ylim=(-1, 1), zlim=(-1, 1)):
    x_num, y_num, z_num = size
    z, y, x = torch.meshgrid(
        torch.linspace(-1, 1, z_num, dtype=torch.float32),
        torch.linspace(-1, 1, y_num, dtype=torch.float32),
        torch.linspace(-1, 1, x_num, dtype=torch.float32),
    )
    grid = torch.stack([z, y, x], dim=3)
    return grid


@torch.no_grad()
def extract_pointcloud(
    nerf, num_grid_edge=100, bsz_eval=1024, 
    sigma_threshold=5.0, device=None):
    """extract point cloud in PyntCloud format from nerf.

    Args:
        nerf (torch.nn.Module): nerf model.
        num_grid_edge (int): the number of points in a grid edge.
        bsz_eval (int): batch size for nerf inference.
        sigma_threshold (float): threshold for density screening.
        device (torch.device): device to use in inference.
    Returns:
        cloud (pyntcloud.PyntCloud): pointcloud extracted.
    """
    
    grid = _gen_3d_grid(
        (num_grid_edge, num_grid_edge, num_grid_edge)).reshape(-1, 3)
    
    grid_size = len(grid)
    if grid_size % bsz_eval == 0:
        iter_num = grid_size // bsz_eval
    else:
        iter_num = grid_size // bsz_eval + 1

    sigma = []
    color = []
    nerf.eval()
    for i in range(iter_num):
        start = i * bsz_eval
        end = min((i+1)*bsz_eval, grid_size)
        
        x_batch = grid[start:end].to(device)
        d_batch = - torch.ones_like(x_batch) / np.sqrt(3)
        d_batch[:, 1] = 0.
        c_batch, sigma_batch = \
            nerf.radiance_field(x_batch, d_batch, network="fine")
        sigma.append(sigma_batch.cpu().detach().numpy())
        color.append(c_batch.cpu().detach().numpy())

    sigma = np.concatenate(sigma)
    color = np.concatenate(color)
    
    cond = (sigma >= sigma_threshold)[:, 0]
    _g = grid[cond].numpy()
    _c = color[cond] * 255.
    cloud = PyntCloud(
        pd.DataFrame(np.concatenate([_g, _c], axis=1),
        columns=['x', 'y', 'z', 'red', 'green', 'blue'])
    )
    cloud.points[['red', 'green', 'blue']] = \
        cloud.points[['red', 'green', 'blue']].astype(np.uint8)
    return cloud


def position_encode(p, L):
    """Encode positions.
    Args:
        p (ndarray, [batch_size, dim]): Position.
        L (int): encoding param.
    Returns:
        ndarray [batch_size, dim * L]: Encoded position.
    """
    # normalization.
    p = torch.tanh(p)

    batch_size = p.shape[0]
    i = torch.arange(L, dtype=torch.float32, device=p.device)
    a = (2. ** i[None, None]) * math.pi * p[:, :, None]
    s = torch.sin(a)
    c = torch.cos(a)
    e = torch.cat([s, c], axis=2).view(batch_size, -1)
    return e


def position_encode_barf(p, L, alpha=0.):
    """Encode positions (BARF version).
    Args:
        p (ndarray, [batch_size, dim]): Position.
        L (int): encoding param.
        alpha (float): hy-pass rate.
    Returns:
        ndarray [batch_size, dim + dim * L]: Encoded position.
    """
    # normalization.
    p = torch.tanh(p)
    alpha = L * alpha

    batch_size = p.shape[0]
    i = torch.arange(L, dtype=torch.float32, device=p.device)

    filter_mask = (alpha < i).repeat(2)
    all_path_mask = (alpha - i >= 1.).repeat(2)
    part_path_mask = filter_mask == all_path_mask
    w_part = (1 - torch.cos((alpha-i) * math.pi)) * 0.5
    w_part = w_part.repeat(2)

    a = (2. ** i[None, None]) * math.pi * p[:, :, None]
    s = torch.sin(a)
    c = torch.cos(a)
    e = torch.cat([s, c], axis=2)
    e = torch.where(filter_mask, torch.zeros_like(e), e)
    e = torch.where(part_path_mask, e*w_part, e)
    e = torch.cat([p, e.reshape(batch_size, -1)], axis=1)
    return e


def split_ray(t_n, t_f, N, batch_size):
    """Split the ray into N partitions.
    partition: [t_n, t_n + (1 / N) * (t_f - t_n), ..., t_f]
    Args:
        t_n (float): t_near. Start point of split.
        t_f (float): t_far. End point of split.
        N (int): Num of partitions.
        batch_size (int): Batch size.
    Returns:
        ndarray, [batch_size, N]: A partition.
    """
    partitions = np.linspace(t_n, t_f, N+1, dtype=np.float32)
    return np.repeat(partitions[None], repeats=batch_size, axis=0)


def sample_coarse(partitions):
    """Sample ``t_i`` from partitions for ``coarse`` network.
    t_i ~ U[t_n + ((i - 1) / N) * (t_f - t_n), t_n + (i / N) * (t_f - t_n)]
    Args:
        partitions (ndarray, [batch_size, N+1]): Outputs of ``split_ray``.
    Return:
        ndarray, [batch_size, N]: Sampled t.
    """
    t = np.random.uniform(
        partitions[:, :-1], partitions[:, 1:]).astype(np.float32)
    return t


def _pcpdf(partitions, weights, N_s):
    """Sample from piecewise-constant probability density function.
    Args:
        partitions (ndarray, [batch_size, N_p+1]): N_p Partitions.
        weights (ndarray, [batch_size, N_p]): The ratio of sampling from each
            partition.
        N_s (int): Num of samples.
    Returns:
        numpy.ndarray, [batch_size, N_s]: Samples.
    """
    batch_size, N_p = weights.shape

    # normalize weights.
    weights[weights < 1e-16] = 1e-16
    weights /= weights.sum(axis=1, keepdims=True)

    _sample = np.random.uniform(
        0, 1, size=(batch_size, N_s)).astype(np.float32)
    _sample = np.sort(_sample, axis=1)

    # Slopes of a piecewise linear function.
    a = (partitions[:, 1:] - partitions[:, :-1]) / weights

    # Intercepts of a piecewise linear function.
    cum_weights = np.cumsum(weights, axis=1)
    cum_weights = np.pad(cum_weights, ((0, 0), (1, 0)),
                         mode='constant')
    b = partitions[:, :-1] - a * cum_weights[:, :-1]

    sample = np.zeros_like(_sample)
    for j in range(N_p):
        min_j = cum_weights[:, j:j+1]
        max_j = cum_weights[:, j+1:j+2]
        a_j = a[:, j:j+1]
        b_j = b[:, j:j+1]
        mask = ((min_j <= _sample) & (_sample < max_j)).astype(np.float32)
        sample += (a_j * _sample + b_j) * mask

    return sample


def sample_fine(partitions, weights, t_c, N_f):
    """Sample ``t_i`` from partitions for ``fine`` network.
    Sampling from each partition according to given weights.
    Args:
        partitions (ndarray, [batch_size, N_c+1]): Outputs of ``split_ray``.
        weights (ndarray, [batch_size, N_c]):
            T_i * (1 - exp(- sigma_i * delta_i)).
        t_c (ndarray, [batch_size, N_c]): ``t`` of coarse rendering.
        N_f (int): num of sampling.
    Return:
        ndarray, [batch_size, N_c+N_f]: Sampled t.
    """
    t_f = _pcpdf(partitions, weights, N_f)
    t = np.concatenate([t_c, t_f], axis=1)
    t = np.sort(t, axis=1)
    return t


def ray(o, d, t):
    """Returns points on the ray.
    Args:
        o (ndarray, [batch_size, 3]): Start points of the ray.
        d (ndarray, [batch_size, 3]): Directions of the ray.
        t (ndarray, [batch_size, N]): Sampled t.
    Returns:
        ndarray, [batch_size, N, 3]: Points on the ray.
    """
    return o[:, None] + t[..., None] * d[:, None]
