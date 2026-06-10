import os, sys
import numpy as np
import imageio
import json
import random
import time
import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm, trange
from pytorch_msssim import ssim, ms_ssim, SSIM, MS_SSIM
import lpips

import matplotlib.pyplot as plt

from run_nerf_helpers import *

from load_llff import load_llff_data
from load_deepvoxels import load_dv_data
from load_blender import load_blender_data
from load_LINEMOD import load_LINEMOD_data
import nni

import torch_pruning as tp

from backend.haq import haq_simu, ptq

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
np.random.seed(0)
DEBUG = False

def batchify(fn, chunk):
    """Constructs a version of 'fn' that applies to smaller batches.
    """
    if chunk is None:
        return fn
    def ret(inputs):
        return torch.cat([fn(inputs[i:i+chunk]) for i in range(0, inputs.shape[0], chunk)], 0)
    return ret

def run_network(inputs, viewdirs, fn, embed_fn, embeddirs_fn, netchunk=1024*64):
    """Prepares inputs and applies network 'fn'.
    """
    inputs_flat = torch.reshape(inputs, [-1, inputs.shape[-1]])
    embedded = embed_fn(inputs_flat)

    if viewdirs is not None:
        input_dirs = viewdirs[:,None].expand(inputs.shape)
        input_dirs_flat = torch.reshape(input_dirs, [-1, input_dirs.shape[-1]])
        embedded_dirs = embeddirs_fn(input_dirs_flat)
        embedded = torch.cat([embedded, embedded_dirs], -1)

    outputs_flat = batchify(fn, netchunk)(embedded)
    outputs = torch.reshape(outputs_flat, list(inputs.shape[:-1]) + [outputs_flat.shape[-1]])
    return outputs


def batchify_rays(rays_flat, chunk=1024*32, **kwargs):
    """Render rays in smaller minibatches to avoid OOM.
    """
    all_ret = {}
    for i in range(0, rays_flat.shape[0], chunk):
        ret = render_rays(rays_flat[i:i+chunk], **kwargs)
        for k in ret:
            if k not in all_ret:
                all_ret[k] = []
            all_ret[k].append(ret[k])

    all_ret = {k : torch.cat(all_ret[k], 0) for k in all_ret}
    return all_ret

def render(H, W, K, chunk=1024*32, rays=None, c2w=None, ndc=True,
                  near=0., far=1.,
                  use_viewdirs=False, c2w_staticcam=None,
                  **kwargs):
    """Render rays
    Args:
      H: int. Height of image in pixels.
      W: int. Width of image in pixels.
      focal: float. Focal length of pinhole camera.
      chunk: int. Maximum number of rays to process simultaneously. Used to
        control maximum memory usage. Does not affect final results.
      rays: array of shape [2, batch_size, 3]. Ray origin and direction for
        each example in batch.
      c2w: array of shape [3, 4]. Camera-to-world transformation matrix.
      ndc: bool. If True, represent ray origin, direction in NDC coordinates.
      near: float or array of shape [batch_size]. Nearest distance for a ray.
      far: float or array of shape [batch_size]. Farthest distance for a ray.
      use_viewdirs: bool. If True, use viewing direction of a point in space in model.
      c2w_staticcam: array of shape [3, 4]. If not None, use this transformation matrix for 
       camera while using other c2w argument for viewing directions.
    Returns:
      rgb_map: [batch_size, 3]. Predicted RGB values for rays.
      disp_map: [batch_size]. Disparity map. Inverse of depth.
      acc_map: [batch_size]. Accumulated opacity (alpha) along a ray.
      extras: dict with everything returned by render_rays().
    """
    if c2w is not None:
        # special case to render full image
        rays_o, rays_d = get_rays(H, W, K, c2w)
    else:
        # use provided ray batch
        rays_o, rays_d = rays

    if use_viewdirs:
        # provide ray directions as input
        viewdirs = rays_d
        if c2w_staticcam is not None:
            # special case to visualize effect of viewdirs
            rays_o, rays_d = get_rays(H, W, K, c2w_staticcam)
        viewdirs = viewdirs / torch.norm(viewdirs, dim=-1, keepdim=True)
        viewdirs = torch.reshape(viewdirs, [-1,3]).float()

    sh = rays_d.shape # [..., 3]
    if ndc:
        # for forward facing scenes
        rays_o, rays_d = ndc_rays(H, W, K[0][0], 1., rays_o, rays_d)

    # Create ray batch
    rays_o = torch.reshape(rays_o, [-1,3]).float()
    rays_d = torch.reshape(rays_d, [-1,3]).float()

    near, far = near * torch.ones_like(rays_d[...,:1]), far * torch.ones_like(rays_d[...,:1])
    rays = torch.cat([rays_o, rays_d, near, far], -1)
    if use_viewdirs:
        rays = torch.cat([rays, viewdirs], -1)

    # Render and reshape
    all_ret = batchify_rays(rays, chunk, **kwargs)
    for k in all_ret:
        k_sh = list(sh[:-1]) + list(all_ret[k].shape[1:])
        all_ret[k] = torch.reshape(all_ret[k], k_sh)

    k_extract = ['rgb_map', 'disp_map', 'acc_map']
    ret_list = [all_ret[k] for k in k_extract]
    ret_dict = {k : all_ret[k] for k in all_ret if k not in k_extract}
    return ret_list + [ret_dict]


def render_path(render_poses, hwf, K, chunk, render_kwargs, gt_imgs=None, savedir=None, render_factor=0):

    H, W, focal = hwf

    if render_factor!=0:
        # Render downsampled for speed
        H = H//render_factor
        W = W//render_factor
        focal = focal/render_factor

    rgbs = []
    disps = []

    t = time.time()
    for i, c2w in enumerate(tqdm(render_poses)):
        print(i, time.time() - t)
        t = time.time()
        rgb, disp, acc, _ = render(H, W, K, chunk=chunk, c2w=c2w[:3,:4], **render_kwargs)
        rgbs.append(rgb.cpu().numpy())
        disps.append(disp.cpu().numpy())
        if i==0:
            print(rgb.shape, disp.shape)

        """
        if gt_imgs is not None and render_factor==0:
            p = -10. * np.log10(np.mean(np.square(rgb.cpu().numpy() - gt_imgs[i])))
            print(p)
        """

        # # calculate PSNR
        # if gt_imgs is not None and render_factor==0:
        #     mse = np.mean(np.square(rgb.cpu().numpy() - gt_imgs[i]))
        #     psnr = 10. * np.log10(1. / mse)
        #     print(f"PSNR: {psnr:.4f} dB")
        #     # save PSNR
        #     with open(os.path.join(savedir, 'psnr.txt'), 'a') as f:
        #         f.write(f"pic {i}: {psnr:.4f} dB\n")

        # calculate PSNR
        if gt_imgs is not None and render_factor==0:
            print("savedir:", savedir)
            mse = np.mean(np.square(rgb.cpu().numpy() - gt_imgs[i]))
            psnr = 10. * np.log10(1. / mse)
            print(f"PSNR: {psnr:.4f} dB")
            # save PSNR
            with open(os.path.join(savedir, 'psnr.txt'), 'a') as f:
                f.write(f"pic {i}: {psnr:.4f} dB\n")

            rgb_tensor = rgb.permute(2, 0, 1).unsqueeze(0).float()
            gt_tensor = torch.from_numpy(gt_imgs[i]).permute(2, 0, 1).unsqueeze(0).float()

            rgb_tensor = rgb_tensor.to("cuda" if torch.cuda.is_available() else "cpu")
            gt_tensor = gt_tensor.to("cuda" if torch.cuda.is_available() else "cpu")

            # SSIM and LPIPS computation (commented out for speed)
            #     f.write(f"pic {i}: {lpips_dist_vgg.item():.4f}\n")

        if savedir is not None:
            rgb8 = to8b(rgbs[-1])
            filename = os.path.join(savedir, '{:03d}.png'.format(i))
            imageio.imwrite(filename, rgb8)


    rgbs = np.stack(rgbs, 0)
    disps = np.stack(disps, 0)

    return rgbs, disps


def create_nerf(args):
    """Instantiate NeRF's MLP model.
    """
    embed_fn, input_ch = get_embedder(args.multires, args.i_embed)

    input_ch_views = 0
    embeddirs_fn = None
    if args.use_viewdirs:
        embeddirs_fn, input_ch_views = get_embedder(args.multires_views, args.i_embed)
    output_ch = 5 if args.N_importance > 0 else 4
    skips = [3]
    model = NeRF(D=args.netdepth, W=args.netwidth, r=int(args.rank),
                 input_ch=input_ch, output_ch=output_ch, skips=skips,
                 input_ch_views=input_ch_views, use_viewdirs=args.use_viewdirs).to(device)
    grad_vars = list(model.parameters())

    model_fine = None
    if args.N_importance > 0:
        model_fine = NeRF(D=args.netdepth_fine, W=args.netwidth_fine, r=int(args.rank_fine),
                          input_ch=input_ch, output_ch=output_ch, skips=skips,
                          input_ch_views=input_ch_views, use_viewdirs=args.use_viewdirs).to(device)
        grad_vars += list(model_fine.parameters())

    network_query_fn = lambda inputs, viewdirs, network_fn : run_network(inputs, viewdirs, network_fn,
                                                                embed_fn=embed_fn,
                                                                embeddirs_fn=embeddirs_fn,
                                                                netchunk=args.netchunk)

    # Create optimizer
    optimizer = torch.optim.Adam(params=grad_vars, lr=args.lrate, betas=(0.9, 0.999))

    start = 0
    basedir = args.basedir
    expname = args.expname

    ##########################

    # Load checkpoints
    if args.ft_path is not None and args.ft_path!='None':
        ckpts = [args.ft_path]
    else:
        ckpts = [os.path.join(basedir, expname, f) for f in sorted(os.listdir(os.path.join(basedir, expname))) if 'tar' in f]

    print('Found ckpts', ckpts)
    if len(ckpts) > 0 and not args.no_reload:
        ckpt_path = ckpts[-1]
        print('Reloading from', ckpt_path)
        ckpt = torch.load(ckpt_path)

        start = ckpt['global_step']
        optimizer.load_state_dict(ckpt['optimizer_state_dict'])

        # Load model
        print("ckpt['network_fn_state_dict']:", ckpt['network_fn_state_dict'].keys())
        model.load_state_dict(ckpt['network_fn_state_dict'])
        # tp.load_state_dict(model, loaded_state_dict)

        if model_fine is not None:
            model_fine.load_state_dict(ckpt['network_fine_state_dict'])

    ##########################

    render_kwargs_train = {
        'network_query_fn' : network_query_fn,
        'perturb' : args.perturb,
        'N_importance' : args.N_importance,
        'network_fine' : model_fine,
        'N_samples' : args.N_samples,
        'network_fn' : model,
        'use_viewdirs' : args.use_viewdirs,
        'white_bkgd' : args.white_bkgd,
        'raw_noise_std' : args.raw_noise_std,
    }

    # NDC only good for LLFF-style forward facing data
    if args.dataset_type != 'llff' or args.no_ndc:
        print('Not ndc!')
        render_kwargs_train['ndc'] = False
        render_kwargs_train['lindisp'] = args.lindisp

    render_kwargs_test = {k : render_kwargs_train[k] for k in render_kwargs_train}
    render_kwargs_test['perturb'] = False
    render_kwargs_test['raw_noise_std'] = 0.

    return render_kwargs_train, render_kwargs_test, start, grad_vars, optimizer


def raw2outputs(raw, z_vals, rays_d, raw_noise_std=0, white_bkgd=False, pytest=False):
    """Transforms model's predictions to semantically meaningful values.
    Args:
        raw: [num_rays, num_samples along ray, 4]. Prediction from model.
        z_vals: [num_rays, num_samples along ray]. Integration time.
        rays_d: [num_rays, 3]. Direction of each ray.
    Returns:
        rgb_map: [num_rays, 3]. Estimated RGB color of a ray.
        disp_map: [num_rays]. Disparity map. Inverse of depth map.
        acc_map: [num_rays]. Sum of weights along each ray.
        weights: [num_rays, num_samples]. Weights assigned to each sampled color.
        depth_map: [num_rays]. Estimated distance to object.
    """
    raw2alpha = lambda raw, dists, act_fn=F.relu: 1.-torch.exp(-act_fn(raw)*dists)

    dists = z_vals[...,1:] - z_vals[...,:-1]
    dists = torch.cat([dists, torch.Tensor([1e10]).expand(dists[...,:1].shape)], -1)  # [N_rays, N_samples]

    dists = dists * torch.norm(rays_d[...,None,:], dim=-1)

    rgb = torch.sigmoid(raw[...,:3])  # [N_rays, N_samples, 3]
    noise = 0.
    if raw_noise_std > 0.:
        noise = torch.randn(raw[...,3].shape) * raw_noise_std

        # Overwrite randomly sampled data if pytest
        if pytest:
            np.random.seed(0)
            noise = np.random.rand(*list(raw[...,3].shape)) * raw_noise_std
            noise = torch.Tensor(noise)

    alpha = raw2alpha(raw[...,3] + noise, dists)  # [N_rays, N_samples]
    # weights = alpha * tf.math.cumprod(1.-alpha + 1e-10, -1, exclusive=True)
    weights = alpha * torch.cumprod(torch.cat([torch.ones((alpha.shape[0], 1)), 1.-alpha + 1e-10], -1), -1)[:, :-1]
    rgb_map = torch.sum(weights[...,None] * rgb, -2)  # [N_rays, 3]

    depth_map = torch.sum(weights * z_vals, -1)
    disp_map = 1./torch.max(1e-10 * torch.ones_like(depth_map), depth_map / torch.sum(weights, -1))
    acc_map = torch.sum(weights, -1)

    if white_bkgd:
        rgb_map = rgb_map + (1.-acc_map[...,None])

    return rgb_map, disp_map, acc_map, weights, depth_map


def render_rays(ray_batch,
                network_fn,
                network_query_fn,
                N_samples,
                retraw=False,
                lindisp=False,
                perturb=0.,
                N_importance=0,
                network_fine=None,
                white_bkgd=False,
                raw_noise_std=0.,
                verbose=False,
                pytest=False):
    """Volumetric rendering.
    Args:
      ray_batch: array of shape [batch_size, ...]. All information necessary
        for sampling along a ray, including: ray origin, ray direction, min
        dist, max dist, and unit-magnitude viewing direction.
      network_fn: function. Model for predicting RGB and density at each point
        in space.
      network_query_fn: function used for passing queries to network_fn.
      N_samples: int. Number of different times to sample along each ray.
      retraw: bool. If True, include model's raw, unprocessed predictions.
      lindisp: bool. If True, sample linearly in inverse depth rather than in depth.
      perturb: float, 0 or 1. If non-zero, each ray is sampled at stratified
        random points in time.
      N_importance: int. Number of additional times to sample along each ray.
        These samples are only passed to network_fine.
      network_fine: "fine" network with same spec as network_fn.
      white_bkgd: bool. If True, assume a white background.
      raw_noise_std: ...
      verbose: bool. If True, print more debugging info.
    Returns:
      rgb_map: [num_rays, 3]. Estimated RGB color of a ray. Comes from fine model.
      disp_map: [num_rays]. Disparity map. 1 / depth.
      acc_map: [num_rays]. Accumulated opacity along each ray. Comes from fine model.
      raw: [num_rays, num_samples, 4]. Raw predictions from model.
      rgb0: See rgb_map. Output for coarse model.
      disp0: See disp_map. Output for coarse model.
      acc0: See acc_map. Output for coarse model.
      z_std: [num_rays]. Standard deviation of distances along ray for each
        sample.
    """
    N_rays = ray_batch.shape[0]
    rays_o, rays_d = ray_batch[:,0:3], ray_batch[:,3:6] # [N_rays, 3] each
    viewdirs = ray_batch[:,-3:] if ray_batch.shape[-1] > 8 else None
    bounds = torch.reshape(ray_batch[...,6:8], [-1,1,2])
    near, far = bounds[...,0], bounds[...,1] # [-1,1]

    t_vals = torch.linspace(0., 1., steps=N_samples)
    if not lindisp:
        z_vals = near * (1.-t_vals) + far * (t_vals)
    else:
        z_vals = 1./(1./near * (1.-t_vals) + 1./far * (t_vals))

    z_vals = z_vals.expand([N_rays, N_samples])

    if perturb > 0.:
        # get intervals between samples
        mids = .5 * (z_vals[...,1:] + z_vals[...,:-1])
        upper = torch.cat([mids, z_vals[...,-1:]], -1)
        lower = torch.cat([z_vals[...,:1], mids], -1)
        # stratified samples in those intervals
        t_rand = torch.rand(z_vals.shape)

        # Pytest, overwrite u with numpy's fixed random numbers
        if pytest:
            np.random.seed(0)
            t_rand = np.random.rand(*list(z_vals.shape))
            t_rand = torch.Tensor(t_rand)

        z_vals = lower + (upper - lower) * t_rand

    pts = rays_o[...,None,:] + rays_d[...,None,:] * z_vals[...,:,None] # [N_rays, N_samples, 3]


#     raw = run_network(pts)
    raw = network_query_fn(pts, viewdirs, network_fn)
    rgb_map, disp_map, acc_map, weights, depth_map = raw2outputs(raw, z_vals, rays_d, raw_noise_std, white_bkgd, pytest=pytest)

    if N_importance > 0:

        rgb_map_0, disp_map_0, acc_map_0 = rgb_map, disp_map, acc_map

        z_vals_mid = .5 * (z_vals[...,1:] + z_vals[...,:-1])
        z_samples = sample_pdf(z_vals_mid, weights[...,1:-1], N_importance, det=(perturb==0.), pytest=pytest)
        z_samples = z_samples.detach()

        z_vals, _ = torch.sort(torch.cat([z_vals, z_samples], -1), -1)
        pts = rays_o[...,None,:] + rays_d[...,None,:] * z_vals[...,:,None] # [N_rays, N_samples + N_importance, 3]

        run_fn = network_fn if network_fine is None else network_fine
#         raw = run_network(pts, fn=run_fn)
        raw = network_query_fn(pts, viewdirs, run_fn)

        rgb_map, disp_map, acc_map, weights, depth_map = raw2outputs(raw, z_vals, rays_d, raw_noise_std, white_bkgd, pytest=pytest)

    ret = {'rgb_map' : rgb_map, 'disp_map' : disp_map, 'acc_map' : acc_map}
    if retraw:
        ret['raw'] = raw
    if N_importance > 0:
        ret['rgb0'] = rgb_map_0
        ret['disp0'] = disp_map_0
        ret['acc0'] = acc_map_0
        ret['z_std'] = torch.std(z_samples, dim=-1, unbiased=False)  # [N_rays]

    for k in ret:
        if (torch.isnan(ret[k]).any() or torch.isinf(ret[k]).any()) and DEBUG:
            print(f"! [Numerical Error] {k} contains nan or inf.")

    return ret


def config_parser():

    import configargparse
    parser = configargparse.ArgumentParser()
    parser.add_argument('--config', is_config_file=True, 
                        help='config file path')
    parser.add_argument("--expname", type=str, 
                        help='experiment name')
    parser.add_argument("--basedir", type=str, default='./logs/', 
                        help='where to store ckpts and logs')
    parser.add_argument("--datadir", type=str, default='./data/llff/fern', 
                        help='input data directory')

    # training options
    parser.add_argument("--netdepth", type=int, default=8, 
                        help='layers in network')
    parser.add_argument("--netwidth", type=int, default=256, 
                        help='channels per layer')
    parser.add_argument("--netdepth_fine", type=int, default=8, 
                        help='layers in fine network')
    parser.add_argument("--netwidth_fine", type=int, default=256, 
                        help='channels per layer in fine network')
    parser.add_argument("--N_rand", type=int, default=32*32*4, 
                        help='batch size (number of random rays per gradient step)')
    parser.add_argument("--lrate", type=float, default=5e-4, 
                        help='learning rate')
    parser.add_argument("--lrate_decay", type=int, default=250, 
                        help='exponential learning rate decay (in 1000 steps)')
    parser.add_argument("--chunk", type=int, default=1024*32, 
                        help='number of rays processed in parallel, decrease if running out of memory')
    parser.add_argument("--netchunk", type=int, default=1024*64, 
                        help='number of pts sent through network in parallel, decrease if running out of memory')
    parser.add_argument("--no_batching", action='store_true', 
                        help='only take random rays from 1 image at a time')
    parser.add_argument("--no_reload", action='store_true', 
                        help='do not reload weights from saved ckpt')
    parser.add_argument("--ft_path", type=str, default=None, 
                        help='specific weights npy file to reload for coarse network')
    # add rank
    parser.add_argument("--rank", type=int, default=16,
                        help='rank of the low-rank approximation of the network')
    parser.add_argument("--rank_fine", type=int, default=16,
                        help='rank of the low-rank approximation of the fine network')
    parser.add_argument("--pruningratio", type=float, default=0.9,
                        help='pruning ratio')
    parser.add_argument("--bits", type=int, default=8,
                        help='quantization bits')
    parser.add_argument("--ratio", type=float, default=1.5,
                        help='quantization ratio')

    # rendering options
    parser.add_argument("--N_samples", type=int, default=64, 
                        help='number of coarse samples per ray')
    parser.add_argument("--N_importance", type=int, default=0,
                        help='number of additional fine samples per ray')
    parser.add_argument("--perturb", type=float, default=1.,
                        help='set to 0. for no jitter, 1. for jitter')
    parser.add_argument("--use_viewdirs", action='store_true', 
                        help='use full 5D input instead of 3D')
    parser.add_argument("--i_embed", type=int, default=0, 
                        help='set 0 for default positional encoding, -1 for none')
    parser.add_argument("--multires", type=int, default=10, 
                        help='log2 of max freq for positional encoding (3D location)')
    parser.add_argument("--multires_views", type=int, default=4, 
                        help='log2 of max freq for positional encoding (2D direction)')
    parser.add_argument("--raw_noise_std", type=float, default=0., 
                        help='std dev of noise added to regularize sigma_a output, 1e0 recommended')

    parser.add_argument("--render_only", action='store_true', 
                        help='do not optimize, reload weights and render out render_poses path')
    parser.add_argument("--render_test", action='store_true', 
                        help='render the test set instead of render_poses path')
    parser.add_argument("--render_factor", type=int, default=0, 
                        help='downsampling factor to speed up rendering, set 4 or 8 for fast preview')

    # training options
    parser.add_argument("--precrop_iters", type=int, default=0,
                        help='number of steps to train on central crops')
    parser.add_argument("--precrop_frac", type=float,
                        default=.5, help='fraction of img taken for central crops') 

    # dataset options
    parser.add_argument("--dataset_type", type=str, default='llff', 
                        help='options: llff / blender / deepvoxels')
    parser.add_argument("--testskip", type=int, default=8, 
                        help='will load 1/N images from test/val sets, useful for large datasets like deepvoxels')

    ## deepvoxels flags
    parser.add_argument("--shape", type=str, default='greek', 
                        help='options : armchair / cube / greek / vase')

    ## blender flags
    parser.add_argument("--white_bkgd", action='store_true', 
                        help='set to render synthetic data on a white bkgd (always use for dvoxels)')
    parser.add_argument("--half_res", action='store_true', 
                        help='load blender synthetic data at 400x400 instead of 800x800')

    ## llff flags
    parser.add_argument("--factor", type=int, default=8, 
                        help='downsample factor for LLFF images')
    parser.add_argument("--no_ndc", action='store_true', 
                        help='do not use normalized device coordinates (set for non-forward facing scenes)')
    parser.add_argument("--lindisp", action='store_true', 
                        help='sampling linearly in disparity rather than depth')
    parser.add_argument("--spherify", action='store_true', 
                        help='set for spherical 360 scenes')
    parser.add_argument("--llffhold", type=int, default=8, 
                        help='will take every 1/N images as LLFF test set, paper uses 8')

    # logging/saving options
    parser.add_argument("--i_print",   type=int, default=100, 
                        help='frequency of console printout and metric loggin')
    parser.add_argument("--i_img",     type=int, default=500, 
                        help='frequency of tensorboard image logging')
    parser.add_argument("--i_weights", type=int, default=10000, 
                        help='frequency of weight ckpt saving')
    parser.add_argument("--i_testset", type=int, default=50000, 
                        help='frequency of testset saving')
    parser.add_argument("--i_video",   type=int, default=50000, 
                        help='frequency of render_poses video saving')

    return parser


def train():

    parser = config_parser()
    args = parser.parse_args()
    # use nni to fetch parameters
    tuner_params = nni.get_next_parameter()
    args.__dict__.update(tuner_params)
    # print(args)

    # Load data
    K = None
    if args.dataset_type == 'llff':
        images, poses, bds, render_poses, i_test = load_llff_data(args.datadir, args.factor,
                                                                  recenter=True, bd_factor=.75,
                                                                  spherify=args.spherify)
        hwf = poses[0,:3,-1]
        poses = poses[:,:3,:4]
        print('Loaded llff', images.shape, render_poses.shape, hwf, args.datadir)
        if not isinstance(i_test, list):
            i_test = [i_test]

        if args.llffhold > 0:
            print('Auto LLFF holdout,', args.llffhold)
            i_test = np.arange(images.shape[0])[::args.llffhold]

        i_val = i_test
        i_train = np.array([i for i in np.arange(int(images.shape[0])) if
                        (i not in i_test and i not in i_val)])

        print('DEFINING BOUNDS')
        if args.no_ndc:
            near = np.ndarray.min(bds) * .9
            far = np.ndarray.max(bds) * 1.
            
        else:
            near = 0.
            far = 1.
        print('NEAR FAR', near, far)

    elif args.dataset_type == 'blender':
        images, poses, render_poses, hwf, i_split = load_blender_data(args.datadir, args.half_res, args.testskip)
        print('Loaded blender', images.shape, render_poses.shape, hwf, args.datadir)
        i_train, i_val, i_test = i_split

        near = 2.
        far = 6.

        if args.white_bkgd:
            images = images[...,:3]*images[...,-1:] + (1.-images[...,-1:])
        else:
            images = images[...,:3]

    elif args.dataset_type == 'LINEMOD':
        images, poses, render_poses, hwf, K, i_split, near, far = load_LINEMOD_data(args.datadir, args.half_res, args.testskip)
        print(f'Loaded LINEMOD, images shape: {images.shape}, hwf: {hwf}, K: {K}')
        print(f'[CHECK HERE] near: {near}, far: {far}.')
        i_train, i_val, i_test = i_split

        if args.white_bkgd:
            images = images[...,:3]*images[...,-1:] + (1.-images[...,-1:])
        else:
            images = images[...,:3]

    elif args.dataset_type == 'deepvoxels':

        images, poses, render_poses, hwf, i_split = load_dv_data(scene=args.shape,
                                                                 basedir=args.datadir,
                                                                 testskip=args.testskip)

        print('Loaded deepvoxels', images.shape, render_poses.shape, hwf, args.datadir)
        i_train, i_val, i_test = i_split

        hemi_R = np.mean(np.linalg.norm(poses[:,:3,-1], axis=-1))
        near = hemi_R-1.
        far = hemi_R+1.

    else:
        print('Unknown dataset type', args.dataset_type, 'exiting')
        return

    # Cast intrinsics to right types
    H, W, focal = hwf
    H, W = int(H), int(W)
    hwf = [H, W, focal]

    if K is None:
        K = np.array([
            [focal, 0, 0.5*W],
            [0, focal, 0.5*H],
            [0, 0, 1]
        ])

    if args.render_test:
        render_poses = np.array(poses[i_test])

    # Create log dir and copy the config file
    basedir = args.basedir
    expname = args.expname
    os.makedirs(os.path.join(basedir, expname), exist_ok=True)
    f = os.path.join(basedir, expname, 'args.txt')
    with open(f, 'w') as file:
        for arg in sorted(vars(args)):
            attr = getattr(args, arg)
            file.write('{} = {}\n'.format(arg, attr))
    if args.config is not None:
        f = os.path.join(basedir, expname, 'config.txt')
        with open(f, 'w') as file:
            file.write(open(args.config, 'r').read())

    # Create nerf model
    render_kwargs_train, render_kwargs_test, start, grad_vars, optimizer = create_nerf(args)
    global_step = start
    
    # Setup pruning parameters
    imp = tp.importance.MagnitudeImportance(p=2)

    ignored_layers = []
    for m in render_kwargs_train['network_fn'].modules():
        if isinstance(m, nn.Linear) and (m.out_features == 3 or m.out_features == 1):
            ignored_layers.append(m)

    # Display model structure
    print('Model structure:')
    print(render_kwargs_train['network_fn'])
    # Display fine model structure
    if render_kwargs_train['N_importance'] > 0:
        print('Fine model structure:')
        print(render_kwargs_train['network_fine'])
    # Save model structure to file
    with open(os.path.join(basedir, expname, 'model.txt'), 'w') as file:
        file.write(str(render_kwargs_train['network_fn']))

    bds_dict = {
        'near' : near,
        'far' : far,
    }
    render_kwargs_train.update(bds_dict)
    render_kwargs_test.update(bds_dict)

    # Move testing data to GPU
    render_poses = torch.Tensor(render_poses).to(device)

    # Short circuit if only rendering out from trained model
    if args.render_only:
        print('RENDER ONLY')
        with torch.no_grad():
            if args.render_test:
                # render_test switches to test poses
                images = images[i_test]
                # testsavedir = os.path.join(basedir, expname, 'test_images')
                # os.makedirs(testsavedir, exist_ok=True)
                # for i, img in enumerate(images):
                #     imageio.imwrite(os.path.join(testsavedir, f'{i}.png'), img)
            else:
                # Default is smoother render_poses path
                images = None

            testsavedir = os.path.join(basedir, expname, 'renderonly_{}_{:06d}'.format('test' if args.render_test else 'path', start))
            os.makedirs(testsavedir, exist_ok=True)
            print('test poses shape', render_poses.shape)

            rgbs, _ = render_path(render_poses, hwf, K, args.chunk, render_kwargs_test, gt_imgs=images, savedir=testsavedir, render_factor=args.render_factor)
            print('Done rendering', testsavedir)
            imageio.mimwrite(os.path.join(testsavedir, 'video.mp4'), to8b(rgbs), fps=30, quality=8)

            return

    # Prepare raybatch tensor if batching random rays
    N_rand = args.N_rand
    use_batching = not args.no_batching
    if use_batching:
        # For random ray batching
        print('get rays')
        rays = np.stack([get_rays_np(H, W, K, p) for p in poses[:,:3,:4]], 0) # [N, ro+rd, H, W, 3]
        print('done, concats')
        rays_rgb = np.concatenate([rays, images[:,None]], 1) # [N, ro+rd+rgb, H, W, 3]
        rays_rgb = np.transpose(rays_rgb, [0,2,3,1,4]) # [N, H, W, ro+rd+rgb, 3]
        rays_rgb = np.stack([rays_rgb[i] for i in i_train], 0) # train images only
        rays_rgb = np.reshape(rays_rgb, [-1,3,3]) # [(N-1)*H*W, ro+rd+rgb, 3]
        rays_rgb = rays_rgb.astype(np.float32)
        print('shuffle rays')
        np.random.shuffle(rays_rgb)

        print('done')
        i_batch = 0

    # Move training data to GPU
    if use_batching:
        images = torch.Tensor(images).to(device)
    poses = torch.Tensor(poses).to(device)
    if use_batching:
        rays_rgb = torch.Tensor(rays_rgb).to(device)


    N_iters = 200000 + 1
    # for HPO
    # N_iters = 10000 + 1
    # N_iters = 1000 + 1
    print('Begin')
    print('TRAIN views are', i_train)
    print('TEST views are', i_test)
    print('VAL views are', i_val)

    # Summary writers
    # writer = SummaryWriter(os.path.join(basedir, 'summaries', expname))
    
    start = start + 1
    for i in trange(start, N_iters):
        time0 = time.time()

        # Sample random ray batch
        if use_batching:
            # Random over all images
            batch = rays_rgb[i_batch:i_batch+N_rand] # [B, 2+1, 3*?]
            batch = torch.transpose(batch, 0, 1)
            batch_rays, target_s = batch[:2], batch[2]

            i_batch += N_rand
            if i_batch >= rays_rgb.shape[0]:
                print("Shuffle data after an epoch!")
                rand_idx = torch.randperm(rays_rgb.shape[0])
                rays_rgb = rays_rgb[rand_idx]
                i_batch = 0

        else:
            # Random from one image
            img_i = np.random.choice(i_train)
            target = images[img_i]
            target = torch.Tensor(target).to(device)
            pose = poses[img_i, :3,:4]

            if N_rand is not None:
                rays_o, rays_d = get_rays(H, W, K, torch.Tensor(pose))  # (H, W, 3), (H, W, 3)

                if i < args.precrop_iters:
                    dH = int(H//2 * args.precrop_frac)
                    dW = int(W//2 * args.precrop_frac)
                    coords = torch.stack(
                        torch.meshgrid(
                            torch.linspace(H//2 - dH, H//2 + dH - 1, 2*dH), 
                            torch.linspace(W//2 - dW, W//2 + dW - 1, 2*dW)
                        ), -1)
                    if i == start:
                        print(f"[Config] Center cropping of size {2*dH} x {2*dW} is enabled until iter {args.precrop_iters}")                
                else:
                    coords = torch.stack(torch.meshgrid(torch.linspace(0, H-1, H), torch.linspace(0, W-1, W)), -1)  # (H, W, 2)

                coords = torch.reshape(coords, [-1,2])  # (H * W, 2)
                select_inds = np.random.choice(coords.shape[0], size=[N_rand], replace=False)  # (N_rand,)
                select_coords = coords[select_inds].long()  # (N_rand, 2)
                rays_o = rays_o[select_coords[:, 0], select_coords[:, 1]]  # (N_rand, 3)
                rays_d = rays_d[select_coords[:, 0], select_coords[:, 1]]  # (N_rand, 3)
                batch_rays = torch.stack([rays_o, rays_d], 0)
                target_s = target[select_coords[:, 0], select_coords[:, 1]]  # (N_rand, 3)

        #####  Core optimization loop  #####
        rgb, disp, acc, extras = render(H, W, K, chunk=args.chunk, rays=batch_rays,
                                                verbose=i < 10, retraw=True,
                                                **render_kwargs_train)

        optimizer.zero_grad()
        img_loss = img2mse(rgb, target_s)
        trans = extras['raw'][...,-1]
        loss = img_loss
        psnr = mse2psnr(img_loss)

        if 'rgb0' in extras:
            img_loss0 = img2mse(extras['rgb0'], target_s)
            loss = loss + img_loss0
            psnr0 = mse2psnr(img_loss0)

        loss.backward()
        # if i % 100 == 0:
        #     pruner.step()
        optimizer.step()
        # macs, nparams = tp.utils.count_ops_and_params(render_kwargs_train['network_fn'], torch.rand(1, 198))
        # print(f"MACs: {macs}, #Params: {nparams}")
        if i % 4000 == 0:
            # Print model structure
            print('Model structure:')
            print(render_kwargs_train['network_fn'])

        # NOTE: IMPORTANT!
        ###   update learning rate   ###
        decay_rate = 0.1
        decay_steps = args.lrate_decay * 1000
        new_lrate = args.lrate * (decay_rate ** (global_step / decay_steps))
        for param_group in optimizer.param_groups:
            param_group['lr'] = new_lrate
        ################################

        dt = time.time()-time0
        # print(f"Step: {global_step}, Loss: {loss}, Time: {dt}")
        #####           end            #####

        # Rest is logging
        if i%args.i_weights==0:
            path = os.path.join(basedir, expname, '{:06d}.tar'.format(i))
            torch.save({
                'global_step': global_step,
                'network_fn_state_dict': render_kwargs_train['network_fn'].state_dict(),
                # 'network_fn_state_dict': tp.state_dict(render_kwargs_train['network_fn']),
                'network_fine_state_dict': render_kwargs_train['network_fine'].state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
            }, path)
            torch.save(render_kwargs_train['network_fn'], os.path.join(basedir, expname, 'network_fn_{:06d}.pth'.format(i)))
            print('Saved checkpoints at', path)

        # if i%args.i_video==0 and i > 0:
        #     # Turn on testing mode
        #     with torch.no_grad():
        #         rgbs, disps = render_path(render_poses, hwf, K, args.chunk, render_kwargs_test)
        #     print('Done, saving', rgbs.shape, disps.shape)
        #     moviebase = os.path.join(basedir, expname, '{}_spiral_{:06d}_'.format(expname, i))
        #     imageio.mimwrite(moviebase + 'rgb.mp4', to8b(rgbs), fps=30, quality=8)
        #     imageio.mimwrite(moviebase + 'disp.mp4', to8b(disps / np.max(disps)), fps=30, quality=8)

        #     # if args.use_viewdirs:
        #     #     render_kwargs_test['c2w_staticcam'] = render_poses[0][:3,:4]
        #     #     with torch.no_grad():
        #     #         rgbs_still, _ = render_path(render_poses, hwf, args.chunk, render_kwargs_test)
        #     #     render_kwargs_test['c2w_staticcam'] = None
        #     #     imageio.mimwrite(moviebase + 'rgb_still.mp4', to8b(rgbs_still), fps=30, quality=8)

        # if i%args.i_testset==0 and i > 0:
        #     testsavedir = os.path.join(basedir, expname, 'testset_{:06d}'.format(i))
        #     os.makedirs(testsavedir, exist_ok=True)
        #     print('test poses shape', poses[i_test].shape)
        #     with torch.no_grad():
        #         render_path(torch.Tensor(poses[i_test]).to(device), hwf, K, args.chunk, render_kwargs_test, gt_imgs=images[i_test], savedir=testsavedir)
        #     print('Saved test set')

        if i%args.i_print==0:
            tqdm.write(f"[TRAIN] Iter: {i} Loss: {loss.item()}  PSNR: {psnr.item()}")
        
            # rram_num = float(sum(p.numel() for p in render_kwargs_train['network_fn'].parameters() if p.requires_grad))
            rram_num = float(sum(p.numel() for p in render_kwargs_train['network_fn'].parameters() if p.requires_grad)) + \
                          float(sum(p.numel() for p in render_kwargs_train['network_fine'].parameters() if p.requires_grad))

            # hardware energy and latency estimation
            act_delay = 1 / 254e6
            act_power = 0.81e-6
            act_1bit_e = act_delay * act_power

            # RRAM params
            v_RRAM = 0.1  # Mean inference voltage of RRAM, in V
            t_RRAM = 1e-8  # Mean inference voltage pulse width, in s
            g_avg_rram = 34.05538e-6
            e_avg_rram = v_RRAM * v_RRAM * g_avg_rram * t_RRAM

            # ADC params
            erg_ADC_x1 = 3.1e-3 * 1/ 1.2e9

            # DAC params
            erg_DAC_x1 = 227e-15  # One time DAC operation energy, in J

            # other hardware params
            erg_decoder = 0.42e-15
            erg_MUX = 14.06e-15
            erg_driver = 0.18e-15
            erg_amplifier = 2 * 73.75e-6 * 10e-8
            erg_reading = 30e-6 * 0.1 * 0.1 * 10e-8 * 1.5 * 1.1
            erg_vcmac = 30e-6 * 0.1 * 0.1 * (1+0.5) * 1.2 * 10e-8

            # GPU params (A100)
            t_a100_fp32 = 19.5e12
            p_a100 = 300
            erg_gpu = p_a100 / t_a100_fp32

            resolution = H * W * args.N_samples

            total_e_dac = 0
            total_e_vmm = 0
            total_adc = 0
            total_e_mux = 0
            total_e_decoder = 0
            total_e_driver = 0
            total_e_vcmac = 0
            total_e_gpu = 0

            # energy per layer
            # network_fn energy
            for key, param in render_kwargs_train['network_fn'].state_dict().items():
                if 'weight' in key:
                    in_dim = param.shape[1]
                    out_dim = param.shape[0]

                    e_dac = in_dim * resolution * erg_DAC_x1
                    e_vmm = e_avg_rram * in_dim * out_dim * resolution * args.bits
                    adc = out_dim * resolution * erg_ADC_x1 * 2  # one weight uses two ADCs
                    e_mux = erg_MUX * out_dim * resolution * args.bits
                    e_decoder = erg_decoder * in_dim * resolution
                    e_driver = erg_driver * in_dim * out_dim * resolution * args.bits
                    e_vcmac = (in_dim * resolution * erg_vcmac / 2 + erg_amplifier + erg_reading) * out_dim * args.bits
                    e_gpu = in_dim * out_dim * resolution * erg_gpu

                    total_e_dac += e_dac
                    total_e_vmm += e_vmm
                    total_adc += adc
                    total_e_mux += e_mux
                    total_e_decoder += e_decoder
                    total_e_driver += e_driver
                    total_e_vcmac += e_vcmac
                    total_e_gpu += e_gpu

            # network_fine energy
            if render_kwargs_train['network_fine'] is not None:
                for key, param in render_kwargs_train['network_fine'].state_dict().items():
                    if 'weight' in key:
                        in_dim = param.shape[1]
                        out_dim = param.shape[0]

                        e_dac = in_dim * resolution * erg_DAC_x1
                        e_vmm = e_avg_rram * in_dim * out_dim * resolution * args.bits
                        adc = out_dim * resolution * erg_ADC_x1 * 2
                        e_mux = erg_MUX * out_dim * resolution * args.bits
                        e_decoder = erg_decoder * in_dim * resolution
                        e_driver = erg_driver * in_dim * out_dim * resolution * args.bits
                        e_vcmac = (in_dim * resolution * erg_vcmac / 2 + erg_amplifier + erg_reading) * out_dim * args.bits
                        e_gpu = in_dim * out_dim * resolution * erg_gpu

                        total_e_dac += e_dac
                        total_e_vmm += e_vmm
                        total_adc += adc
                        total_e_mux += e_mux
                        total_e_decoder += e_decoder
                        total_e_driver += e_driver
                        total_e_vcmac += e_vcmac
                        total_e_gpu += e_gpu

            total_e_hardware = total_e_dac + total_e_vmm + total_adc + total_e_mux + total_e_decoder + total_e_driver + total_e_vcmac

            # latency estimation based on RRAM delay
            total_latency = resolution * (1 / 1e8) * args.bits  # 1GHz RRAM frequency

            print(f"RRAM: {rram_num}")

            # metric weight coefficients
            w_psnr = 0.7
            w_rram = 0.1
            w_latency = 0.1
            w_energy = 0.1

            # normalize each metric
            norm_psnr = np.log(1 + (psnr.item() - 20) / 5)
            norm_rram = np.log(1 + 32768 / rram_num)
            norm_latency = np.log(1 + 1 / total_latency * 0.5)
            norm_energy = np.log(1 + 0.1 / total_e_hardware)

            # composite score
            score = (w_psnr * norm_psnr +
                     w_rram * norm_rram +
                     w_latency * norm_latency +
                     w_energy * norm_energy)

            print(f"Normalized PSNR: {norm_psnr:.4f}")
            print(f"Normalized RRAM: {norm_rram:.4f}")
            print(f"Normalized Latency: {norm_latency:.4f}")
            print(f"Normalized Energy: {norm_energy:.4f}")
            print(f"Composite Score: {score:.4f}")
            # metric = {'default': score, 'psnr': psnr.item(), 'rram_num': rram_num}
            # metric = {'default': score, 'psnr': psnr.item(), 'rram_num': rram_num, 'latency': total_latency, 'energy': total_e_hardware}
            metric = {'default': psnr.item(), 'psnr': psnr.item(), 'rram_num': rram_num, 'latency': total_latency, 'energy': total_e_hardware, 'score': score}
        
            nni.report_intermediate_result(metric)

            # early stop if PSNR too low
            if psnr.item() < 15 and i > 5000:
                print('Early stop!')
                score = -1
                # metric = {'default': score, 'psnr': psnr.item(), 'rram_num': rram_num}
                # nni.report_final_result(metric)
                break

            # print('Model structure:', render_kwargs_train['network_fn'])

        global_step += 1

    # Add quantization evaluation before final nni.report_final_result
    print('Starting quantization simulation...')

    # Track best results
    best_quantized_psnr = -float('inf')
    best_bits = 0
    best_weights_fn = None
    best_weights_fine = None
    quantization_results = {}

    # Use subset of test images for evaluation
    test_subset = i_test[:2]
    test_poses_subset = poses[test_subset]
    test_images_subset = images[test_subset]

    bits = args.bits
    ratio = args.ratio
    print(f'\nQuantizing with {bits} bits...')

    print('Quantizing network_fn...')
    original_weights_fn = {}
    quantized_weights_fn = {}
    
    for key, param in render_kwargs_train['network_fn'].state_dict().items():
        if 'weight' in key:
            weight = param.clone()

            decimal_weight, weight_diff, relative_error, resistance = haq_simu(weight, n=bits, ratio=ratio, std=0.15, coarse=True)
            quantized_weights_fn[key] = decimal_weight
            print(f'{key} quantized, rel_error > 0.1 ratio: {torch.sum(torch.abs(relative_error) > 0.1).item() / relative_error.numel():.4f}')
            del weight_diff, relative_error, resistance

            # PTQ quantization alternative
            # decimal_weight, binary = ptq(weight, n=bits)
            # quantized_weights_fn[key] = decimal_weight

            if key != list(render_kwargs_train['network_fn'].state_dict().keys())[-1]:
                torch.cuda.empty_cache()


    for key in quantized_weights_fn.keys():
        render_kwargs_train['network_fn'].state_dict()[key].copy_(quantized_weights_fn[key])

    quantized_weights_fine = None
    if render_kwargs_train['network_fine'] is not None:
        print('Quantizing network_fine...')
        quantized_weights_fine = {}

        for key, param in render_kwargs_train['network_fine'].state_dict().items():
            if 'weight' in key:
                weight = param.clone()

                decimal_weight, weight_diff, relative_error, resistance = haq_simu(weight, n=bits, ratio=ratio, std=0.15, coarse=True)
                quantized_weights_fine[key] = decimal_weight
                print(f'{key} quantized, rel_error > 0.1 ratio: {torch.sum(torch.abs(relative_error) > 0.1).item() / relative_error.numel():.4f}')
                del weight_diff, relative_error, resistance

                # PTQ quantization alternative
                # decimal_weight, binary = ptq(weight, n=bits)
                # quantized_weights_fine[key] = decimal_weight

                if key != list(render_kwargs_train['network_fine'].state_dict().keys())[-1]:
                    torch.cuda.empty_cache()

        for key in quantized_weights_fine.keys():
            render_kwargs_train['network_fine'].state_dict()[key].copy_(quantized_weights_fine[key])

    print(f'Running inference with {bits}-bit quantized model...')
    quantized_savedir = os.path.join(basedir, expname, f'quantized_test_{bits}bits_{i:06d}')
    os.makedirs(quantized_savedir, exist_ok=True)

    with torch.no_grad():
        rgbs, disps = render_path(
            torch.Tensor(test_poses_subset).to(device),
            hwf, K,
            args.chunk,
            render_kwargs_test,
            gt_imgs=test_images_subset,
            savedir=quantized_savedir
        )

    quantized_psnrs = []
    for idx, rgb in enumerate(rgbs):
        rgb_tensor = torch.Tensor(rgb).to(device)
        target_tensor = torch.Tensor(test_images_subset[idx]).to(device)
        img_loss = img2mse(rgb_tensor, target_tensor)
        psnr = mse2psnr(img_loss)
        quantized_psnrs.append(psnr.item())

    avg_quantized_psnr = np.mean(quantized_psnrs)
    print(f'{bits}-bit quantized avg PSNR: {avg_quantized_psnr:.2f}')

    quantization_results[bits] = {
        'psnr': avg_quantized_psnr
    }

    # recompute score with quantized PSNR
    norm_psnr = avg_quantized_psnr / 40.0
    score = (w_psnr * norm_psnr +
            w_rram * norm_rram +
            w_latency * norm_latency +
            w_energy * norm_energy)
    print(f"Quantized normalized PSNR: {norm_psnr:.4f}")
    print(f"Updated composite score: {score:.4f}")
    metric['score'] = score

    # use current result as best
    best_quantized_psnr = avg_quantized_psnr
    best_bits = bits
    best_weights_fn = {k: v.clone() for k, v in quantized_weights_fn.items()}
    if quantized_weights_fine is not None:
        best_weights_fine = {k: v.clone() for k, v in quantized_weights_fine.items()}

    del quantized_weights_fn
    if quantized_weights_fine is not None:
        del quantized_weights_fine
    torch.cuda.empty_cache()

    print(f'\nUsing best quantization: {best_bits} bits, PSNR: {best_quantized_psnr:.2f}')

    for key in best_weights_fn.keys():
        render_kwargs_train['network_fn'].state_dict()[key].copy_(best_weights_fn[key])
    if best_weights_fine is not None:
        for key in best_weights_fine.keys():
            render_kwargs_train['network_fine'].state_dict()[key].copy_(best_weights_fine[key])

    metric.update({
        'quantized_psnr': best_quantized_psnr,
        'best_quantization_bits': best_bits,
        # 'energy_hardware': float(total_e_hardware),
        # 'latency': float(total_latency)
    })

    nni.report_final_result(metric)

if __name__=='__main__':
    torch.set_default_tensor_type('torch.cuda.FloatTensor')

    train()
