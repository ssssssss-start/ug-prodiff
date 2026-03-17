from torch.nn import init
import torch.nn.functional as F
from einops import rearrange, repeat
# from tqdm.notebook import tqdm
from functools import partial
import math, os, copy
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
from prettytable import PrettyTable
import scipy.io as sio
import imgvision as iv
from dataset_Houston import *
from network_Houston import *
# from network_nosele import *
import matplotlib.pyplot as plt
import numpy as np
import torch
from utils import *

def fspecial(func_name, kernel_size, sigma):
    if func_name == 'gaussian':
        m = n = (kernel_size - 1.) / 2.
        y, x = ogrid[-m:m + 1, -n:n + 1]
        h = exp(-(x * x + y * y) / (2. * sigma * sigma))
        h[h < finfo(h.dtype).eps * h.max()] = 0
        sumh = h.sum()
        if sumh != 0:
            h /= sumh
        return h
def Gaussian_downsample(x, psf, s):
    if x.ndim == 2:
        x = np.expand_dims(x, axis=0)
    y = np.zeros((x.shape[0], int(x.shape[1] / s), int(x.shape[2] / s)))
    for i in range(x.shape[0]):
        x1 = x[i, :, :]
        x2 = signal.convolve2d(x1, psf, boundary='symm', mode='same')
        y[i, :, :] = x2[0::s, 0::s]
    return y
def downsample_gaussian_hwc(x_hwc, scale=4):
    """
    x_hwc: (H,W,C) -> (h,w,C)
    """
    x_chw = np.transpose(x_hwc, (2,0,1))  # HWC -> CHW
    sigma = scale / 2.0
    PSF = fspecial('gaussian', 13, sigma)
    lr_chw = Gaussian_downsample(x_chw, PSF, scale)  # (C,h,w)
    lr_hwc = np.transpose(lr_chw, (1,2,0))          # CHW -> HWC
    return lr_hwc

def band_affine_calibrate_by_lrhs(pred01_hwc, lrhs01_chw, scale=4, eps=1e-8):
    """
    pred01_hwc: (H,W,C) in [0,1]  (你的 pred01)
    lrhs01_chw: (C,h,w) in [0,1]  (你的 lrHS 转到[0,1])
    scale: 4 (128->32)
    return:
      pred_cal_hwc: (H,W,C) in [0,1]
      alpha: (C,)
      beta: (C,)
    """
    # 1) pred -> LR
    pred_lr_hwc = downsample_gaussian_hwc(pred01_hwc, scale=scale)  # (h,w,C)

    # 2) reshape到 (C, N)
    C = pred01_hwc.shape[2]
    x = pred_lr_hwc.reshape(-1, C).T          # (C, N)
    y = lrhs01_chw.reshape(C, -1)             # (C, N)

    # 3) per-band 线性回归 y ≈ a*x + b
    mx = x.mean(axis=1, keepdims=True)
    my = y.mean(axis=1, keepdims=True)
    vx = ((x - mx) ** 2).mean(axis=1, keepdims=True)

    alpha = (((x - mx) * (y - my)).mean(axis=1, keepdims=True)) / (vx + eps)   # (C,1)
    beta  = my - alpha * mx                                                     # (C,1)

    alpha = alpha[:, 0]
    beta  = beta[:, 0]

    # 4) apply到 HR pred
    pred_cal = pred01_hwc * alpha.reshape(1,1,C) + beta.reshape(1,1,C)
    pred_cal = np.clip(pred_cal, 0.0, 1.0).astype(np.float32)
    return pred_cal, alpha.astype(np.float32), beta.astype(np.float32)
def _to_bchw(x):
    """
    Accept:
      - (H,W,C) numpy
      - (C,H,W) numpy
      - (B,C,H,W) numpy
    Return: (B,C,H,W)
    """
    x = np.asarray(x)
    if x.ndim == 4:         # BCHW
        return x
    if x.ndim == 3:
        # HWC
        if x.shape[-1] <= 512 and x.shape[-1] > 1:
            return np.transpose(x, (2,0,1))[None, ...]
        # CHW
        if x.shape[0] <= 512 and x.shape[0] > 1:
            return x[None, ...]
    raise ValueError(f"Unsupported shape: {x.shape}")

def band_affine_calibrate(pred, gt, eps=1e-8):
    """
    Per-band affine calibration: pred' = a*pred + b
    pred, gt: can be HWC / CHW / BCHW (numpy)
    Return:
      pred_cal: same layout as input pred (for convenience)
      a, b: [B,C,1,1] in BCHW
    """
    pred_in = np.asarray(pred)
    gt_in   = np.asarray(gt)

    P = _to_bchw(pred_in).astype(np.float64)
    G = _to_bchw(gt_in).astype(np.float64)
    assert P.shape == G.shape, (P.shape, G.shape)

    B, C, H, W = P.shape
    P2 = P.reshape(B, C, -1)
    G2 = G.reshape(B, C, -1)

    mp = P2.mean(axis=-1, keepdims=True)
    mg = G2.mean(axis=-1, keepdims=True)
    vp = P2.var(axis=-1, keepdims=True)

    cov = ((P2 - mp) * (G2 - mg)).mean(axis=-1, keepdims=True)
    a = cov / (vp + eps)
    b = mg - a * mp

    Pcal = (a * P2 + b).reshape(B, C, H, W)

    # 还原回原输入布局
    if pred_in.ndim == 4:            # BCHW
        out = Pcal
    elif pred_in.ndim == 3 and pred_in.shape[-1] > 1:  # HWC
        out = np.transpose(Pcal[0], (1,2,0))
    else:                             # CHW
        out = Pcal[0]

    return out.astype(np.float32), a.astype(np.float32), b.astype(np.float32)
def _uiqi(a, b, eps=1e-12):
    """Universal Image Quality Index between two 2D arrays."""
    a = a.astype(np.float64)
    b = b.astype(np.float64)
    ma, mb = a.mean(), b.mean()
    va, vb = a.var(), b.var()
    cov = ((a - ma) * (b - mb)).mean()
    num = 4 * cov * ma * mb
    den = (va + vb + eps) * (ma * ma + mb * mb + eps)
    return num / den

def _downsample_bchw(x_bchw, scale=4):
    """Simple avgpool downsample for tensors, bchw -> bchw."""
    return F.avg_pool2d(x_bchw, kernel_size=scale, stride=scale)

def _apply_srf(hsi_bchw, srf):
    """
    hsi_bchw: [B, C, H, W]  (C=110)
    srf: numpy or torch, shape [3, C] or [C, 3]
    return ms_bchw: [B, 3, H, W]
    """
    if isinstance(srf, np.ndarray):
        srf = torch.from_numpy(srf)
    srf = srf.to(hsi_bchw.device).float()
    if srf.shape[0] != 3 and srf.shape[1] == 3:
        srf = srf.t()  # -> [3, C]
    # normalize each ms band (optional but usually good)
    srf = srf / (srf.sum(dim=1, keepdim=True) + 1e-12)

    B, C, H, W = hsi_bchw.shape
    hsi_flat = hsi_bchw.view(B, C, -1)              # [B, C, HW]
    ms_flat = torch.einsum('kc,bch->bkh', srf, hsi_flat)  # [B, 3, HW]
    return ms_flat.view(B, 4, H, W)

def compute_qnr(pred_hsi_bchw, lr_hsi_bchw, hr_ms_bchw, srf, scale=4, alpha=1.0, beta=1.0):
    """
    All inputs are torch tensors.
    pred_hsi_bchw: [B, C, H, W]  (C=110, H=128)
    lr_hsi_bchw:   [B, C, h, w]  (h=32)
    hr_ms_bchw:    [B, 3, H, W]
    """
    # map to [0,1] if your tensors are in [-1,1]
    def m11_to_01(x):
        # x can be torch.Tensor or np.ndarray
        if isinstance(x, np.ndarray):
            x = np.clip(x, -1.0, 1.0)
            return (x + 1.0) / 2.0
        elif torch.is_tensor(x):
            x = x.clamp(-1.0, 1.0)
            return (x + 1.0) / 2.0
        else:
            raise TypeError(f"Unsupported type: {type(x)}")

    pred = m11_to_01(pred_hsi_bchw)
    lrhs = m11_to_01(lr_hsi_bchw)
    hrms = m11_to_01(hr_ms_bchw)

    # 1) spectral distortion D_lambda: compare band-to-band correlations at LR scale
    pred_lr = _downsample_bchw(pred, scale=scale)  # -> [B,C,h,w]

    pred_lr_np = pred_lr.detach().cpu().numpy()
    lrhs_np = lrhs.detach().cpu().numpy()

    # compute average abs diff of UIQI between all band pairs
    C = pred_lr_np.shape[1]
    diffs = []
    # 为了不 O(C^2) 太慢，可以抽样一些 band 对；先给全算版本
    for i in range(C):
        for j in range(i+1, C):
            q_pred = _uiqi(pred_lr_np[0, i], pred_lr_np[0, j])
            q_ref  = _uiqi(lrhs_np[0, i], lrhs_np[0, j])
            diffs.append(abs(q_pred - q_ref))
    D_lambda = float(np.mean(diffs))

    # 2) spatial distortion D_s: compare structure between predicted MS and HRMS
    pred_ms = _apply_srf(pred, srf)  # [B,3,H,W]
    pred_ms_np = pred_ms.detach().cpu().numpy()
    hrms_np = hrms.detach().cpu().numpy()

    diffs_s = []
    # compare UIQI of each ms band with PAN-like intensity (mean of ms)
    pan_pred = pred_ms_np[0].mean(axis=0)
    pan_ref  = hrms_np[0].mean(axis=0)
    for k in range(4):
        q_pred = _uiqi(pred_ms_np[0, k], pan_pred)
        q_ref  = _uiqi(hrms_np[0, k], pan_ref)
        diffs_s.append(abs(q_pred - q_ref))
    D_s = float(np.mean(diffs_s))

    QNR = (1 - D_lambda) ** alpha * (1 - D_s) ** beta
    return QNR, D_lambda, D_s
def flow_to_hsv_rgb(flow_2hw, clip_mag=None):
    """
    flow_2hw: [2,H,W] (dx,dy) -> rgb [H,W,3] in 0..1 + mag [H,W]
    """
    dx, dy = flow_2hw[0], flow_2hw[1]
    mag = np.sqrt(dx**2 + dy**2)
    ang = np.arctan2(dy, dx)  # [-pi, pi]

    if clip_mag is None:
        clip_mag = np.percentile(mag, 99.0) + 1e-8
    mag_n = np.clip(mag / clip_mag, 0, 1)

    hue = (ang + np.pi) / (2 * np.pi)  # [0,1]
    sat = np.ones_like(hue)
    val = mag_n

    # HSV -> RGB
    h = hue * 6.0
    c = val * sat
    x = c * (1 - np.abs((h % 2) - 1))
    m = val - c

    z = np.zeros_like(h)
    r = np.empty_like(h); g = np.empty_like(h); b = np.empty_like(h)

    conds = [
        (0 <= h) & (h < 1),
        (1 <= h) & (h < 2),
        (2 <= h) & (h < 3),
        (3 <= h) & (h < 4),
        (4 <= h) & (h < 5),
        (5 <= h) & (h <= 6),
    ]
    vals = [
        (c, x, z),
        (x, c, z),
        (z, c, x),
        (z, x, c),
        (x, z, c),
        (c, z, x),
    ]
    for cond, (rc, gc, bc) in zip(conds, vals):
        r[cond], g[cond], b[cond] = rc[cond], gc[cond], bc[cond]

    rgb = np.stack([r + m, g + m, b + m], axis=-1)
    return rgb, mag, clip_mag

def normalize_rgb_3chw(y_3hw):
    """[3,H,W] -> [H,W,3] normalized to 0..1 using percentile."""
    y = np.transpose(y_3hw, (1, 2, 0))
    lo = np.percentile(y, 1.0)
    hi = np.percentile(y, 99.0)
    y = np.clip((y - lo) / (hi - lo + 1e-8), 0, 1)
    return y

def CC_function(ref, tar, eps=1e-12):
    """
    Same as CC_function_chw but vectorized.
    """
    if isinstance(ref, torch.Tensor):
        ref = ref.detach().cpu().numpy()
    if isinstance(tar, torch.Tensor):
        tar = tar.detach().cpu().numpy()

    ref = np.asarray(ref, dtype=np.float64)
    tar = np.asarray(tar, dtype=np.float64)

    if ref.ndim != 3 or tar.ndim != 3:
        raise ValueError(f"Expected (C,H,W), got ref {ref.shape}, tar {tar.shape}")
    if ref.shape != tar.shape:
        raise ValueError(f"Shape mismatch: ref {ref.shape} vs tar {tar.shape}")

    C, H, W = tar.shape
    x = tar.reshape(C, -1)
    y = ref.reshape(C, -1)

    x = x - x.mean(axis=1, keepdims=True)
    y = y - y.mean(axis=1, keepdims=True)

    up = np.sum(x * y, axis=1)
    down = np.sqrt(np.sum(x * x, axis=1) * np.sum(y * y, axis=1)) + eps

    cc = up / down
    return float(cc.mean())
def CPERGAS(gt, pred, scale, eps=1e-12):
    """
    ERGAS (Erreur Relative Globale Adimensionnelle de Synthèse)

    Parameters
    ----------
    gt : ndarray, shape (H, W, B)
        Ground-truth hyperspectral image
    pred : ndarray, shape (H, W, B)
        Fused / reconstructed hyperspectral image
    scale : int or float
        Spatial resolution ratio (e.g., 4, 8, 16)
    eps : float
        Small value to avoid division by zero

    Returns
    -------
    ergas : float
        ERGAS value
    """

    # check shape
    assert gt.shape == pred.shape, "gt and pred must have the same shape"
    assert gt.ndim == 3, "input must be (H, W, B)"

    # per-band MSE
    mse_band = np.mean((pred - gt) ** 2, axis=(0, 1))   # (B,)

    # per-band mean of ground truth
    mu_band = np.mean(gt, axis=(0, 1))                  # (B,)

    # ERGAS formula
    ergas = (100.0 / scale) * np.sqrt(
        np.mean(mse_band / (mu_band ** 2 + eps))
    )

    return ergas
CHIKUSEI_WL = np.array([
    0.40903,0.41419,0.41936,0.42452,0.42968,0.43484,0.44,0.44516,0.45032,0.45548,
    0.46064,0.4658,0.47096,0.47612,0.48129,0.48645,0.49161,0.49677,0.50193,0.50709,
    0.51225,0.51741,0.52257,0.52773,0.53289,0.53806,0.54321,0.54838,0.55354,0.5587,
    0.56386,0.56902,0.57418,0.57934,0.5845,0.58966,0.59483,0.59999,0.60514,0.61031,
    0.61547,0.62063,0.62579,0.63095,0.63611,0.64127,0.64643,0.65159,0.65675,0.66192,
    0.66707,0.67224,0.6774,0.68256,0.68772,0.69288,0.69804,0.7032,0.70836,0.71352,
    0.71868,0.72385,0.72901,0.73417,0.73933,0.74449,0.74965,0.75481,0.75997,0.76513,
    0.77029,0.77545,0.78061,0.78578,0.79094,0.7961,0.80126,0.80642,0.81158,0.81674,
    0.8219,0.82706,0.83223,0.83738,0.84254,0.84771,0.85287,0.85803,0.86319,0.86835,
    0.87351,0.87867,0.88383,0.88899,0.89416,0.89931,0.90448,0.90964,0.9148,0.91996,
    0.92512,0.93028,0.93544,0.9406,0.94576,0.95092,0.95609,0.96125,0.96641,0.97157
], dtype=np.float32)

def denorm_m11_to_01(x):
    return (x.clamp(-1, 1) + 1.0) / 2.0

def stretch_rgb_global(rgb, low=2, high=98, gamma=2.2, ref_rgb=None, eps=1e-6):
    """
    rgb: HWC, [0,1]
    ref_rgb: 用参考图（建议用MSI）计算同一套 lo/hi，让三张图颜色风格一致
    """
    base = ref_rgb if ref_rgb is not None else rgb
    lo = np.percentile(base, low)
    hi = np.percentile(base, high)
    out = (rgb - lo) / (hi - lo + eps)
    out = np.clip(out, 0, 1)
    if gamma is not None:
        out = out ** (1.0 / gamma)
    return out

def hsi_to_rgb_by_idx(hsi_chw, rgb_idx=(47,27,12)):
    # hsi_chw: [C,H,W] in [-1,1]
    hsi01 = denorm_m11_to_01(hsi_chw)
    rgb = hsi01[list(rgb_idx)].permute(1,2,0).detach().cpu().numpy()
    return rgb
"""
    Define U-net Architecture:
    Approximate reverse diffusion process by using U-net
    U-net of SR3 : U-net backbone + Positional Encoding of time + Multihead Self-Attention
"""

import torch
import torch.nn as nn


def _extract_into_tensor(arr, timesteps, broadcast_shape):
    """
    Extract values from a 1-D numpy array for a batch of indices.
    :param arr: the 1-D numpy array.
    :param timesteps: a tensor of indices into the array to extract.
    :param broadcast_shape: a larger shape of K dimensions with the batch
                            dimension equal to the length of timesteps.
    :return: a tensor of shape [batch_size, 1, ...] where the shape has K dims.
    """
    res = torch.from_numpy(arr).to(device=timesteps.device)[timesteps].float()
    while len(res.shape) < len(broadcast_shape):
        res = res[..., None]
    return res + torch.zeros(broadcast_shape, device=timesteps.device)


def calculate_sam(target_data, reference_data):
    # 归一化目标数据和参考数据
    b, c, h, w = target_data.shape
    target_data = target_data.reshape(b, c, h * w).permute(0, 2, 1)
    reference_data = reference_data.reshape(b, c, h * w).permute(0, 2, 1)
    target_data_norm = torch.nn.functional.normalize(target_data, dim=2)
    reference_data_norm = torch.nn.functional.normalize(reference_data, dim=2)

    # 计算点积
    dot_product = torch.einsum('bnc,bnc->bn', target_data_norm, reference_data_norm)

    # 计算长度乘积
    length_product = torch.norm(target_data_norm, dim=2) * torch.norm(reference_data_norm, dim=2)

    # 计算SAM光谱角
    sam = torch.acos(dot_product / length_product)
    sam_mean = torch.mean(torch.mean(sam, dim=1))
    return sam_mean


def extract(a, t, x_shape):
    """
    从给定的张量a中检索特定的元素。t是一个包含要检索的索引的张量，
    这些索引对应于a张量中的元素。这个函数的输出是一个张量，
    包含了t张量中每个索引对应的a张量中的元素
    :param a:
    :param t:
    :param x_shape:
    :return:
    """
    batch_size = t.shape[0]
    out = a.gather(-1, t.cpu())
    return out.reshape(batch_size, *((1,) * (len(x_shape) - 1))).to(t.device)


class PositionalEncoding(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dim = dim

    def forward(self, noise_level):
        # Input : tensor of value of coefficient alpha at specific step of diffusion process e.g. torch.Tensor([0.03])
        # Transform level of noise into representation of given desired dimension
        count = self.dim // 2
        step = torch.arange(count, dtype=noise_level.dtype, device=noise_level.device) / count
        encoding = noise_level.unsqueeze(1) * torch.exp(-math.log(1e4) * step.unsqueeze(0))
        encoding = torch.cat([torch.sin(encoding), torch.cos(encoding)], dim=-1)
        return encoding


class FeatureWiseAffine(nn.Module):
    def __init__(self, in_channels, out_channels, use_affine_level=False):
        super(FeatureWiseAffine, self).__init__()
        self.use_affine_level = use_affine_level
        self.noise_func = nn.Sequential(nn.Linear(in_channels, out_channels * (1 + self.use_affine_level)))

    def forward(self, x, noise_embed):
        noise = self.noise_func(noise_embed).view(x.shape[0], -1, 1, 1)
        if self.use_affine_level:
            gamma, beta = noise.chunk(2, dim=1)
            x = (1 + gamma) * x + beta
        else:
            x = x + noise
        return x


class Swish(nn.Module):
    def forward(self, x):
        return x * torch.sigmoid(x)


class Upsample(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.up = nn.Upsample(scale_factor=2, mode="nearest")
        self.conv = nn.Conv2d(dim, dim, 3, padding=1)

    def forward(self, x):
        return self.conv(self.up(x))


class Downsample(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.conv = nn.Conv2d(dim, dim, 3, 2, 1)

    def forward(self, x):
        return self.conv(x)


class Block(nn.Module):
    def __init__(self, dim, dim_out, groups=32, dropout=0):
        super().__init__()
        self.block = nn.Sequential(
            nn.GroupNorm(groups, dim),
            Swish(),
            nn.Dropout(dropout) if dropout != 0 else nn.Identity(),
            nn.Conv2d(dim, dim_out, 3, padding=1)
        )

    def forward(self, x):
        return self.block(x)


# Linear Multi-head Self-attention
class SelfAtt(nn.Module):
    def __init__(self, channel_dim, num_heads, norm_groups=32, att_num=0):
        super(SelfAtt, self).__init__()
        self.groupnorm = nn.GroupNorm(norm_groups, channel_dim)
        self.num_heads = num_heads
        self.qkv = nn.Conv2d(channel_dim, channel_dim * 3, 1, bias=False)
        self.proj = nn.Conv2d(channel_dim, channel_dim, 1)
        self.att = att_num

    def forward(self, x):
        x_org = x
        b, c, h, w = x.size()
        x = self.groupnorm(x)
        qkv = rearrange(self.qkv(x), "b (qkv heads c) h w -> (qkv) b heads c (h w)", heads=self.num_heads, qkv=3)
        queries, keys, values = qkv[0], qkv[1], qkv[2]

        keys = F.softmax(keys, dim=-1)
        att = torch.einsum('bhdn,bhen->bhde', keys, values)
        out = torch.einsum('bhde,bhdn->bhen', att, queries)
        out = rearrange(out, 'b heads c (h w) -> b (heads c) h w', heads=self.num_heads, h=h, w=w)

        return x_org + self.att * self.proj(out)


class Cross_Att(nn.Module):
    def __init__(self, channel_dim, num_heads, norm_groups=32, att_num=0):
        super(Cross_Att, self).__init__()
        self.att = att_num
        self.groupnorm_1 = nn.GroupNorm(norm_groups, channel_dim)
        self.groupnorm_2 = nn.GroupNorm(norm_groups, channel_dim)
        self.num_heads = num_heads
        self.qkv_1 = nn.Conv2d(channel_dim, channel_dim * 3, 1, bias=False)
        self.qkv_2 = nn.Conv2d(channel_dim, channel_dim * 3, 1, bias=False)

        self.proj = nn.Conv2d(channel_dim, channel_dim, 1)

        self.downsample = nn.Sequential(nn.Conv2d(channel_dim, 2 * channel_dim, 3, 1, 1),
                                        nn.Upsample(scale_factor=0.5, mode='bicubic'),

                                        nn.Conv2d(2 * channel_dim, 2 * channel_dim, 3, 1, 1),
                                        nn.Upsample(scale_factor=0.5, mode='bicubic'),

                                        nn.Conv2d(2 * channel_dim, 4 * channel_dim, 3, 1, 1),
                                        nn.Upsample(scale_factor=0.5, mode='bicubic'),

                                        nn.Conv2d(4 * channel_dim, 4 * channel_dim, 3, 2, 1),
                                        nn.Upsample(scale_factor=0.5, mode='bicubic'),
                                        )

        self.upsample = nn.Sequential(nn.Conv2d(2 * channel_dim, 1 * channel_dim, 3, 1, 1),
                                      nn.Upsample(scale_factor=2, mode='bicubic'),

                                      nn.Conv2d(2 * channel_dim, 2 * channel_dim, 3, 1, 1),
                                      nn.Upsample(scale_factor=2, mode='bicubic'),

                                      nn.Conv2d(4 * channel_dim, 2 * channel_dim, 3, 1, 1),
                                      nn.Upsample(scale_factor=2, mode='bicubic'),

                                      nn.Conv2d(4 * channel_dim, 4 * channel_dim, 3, 1, 1),
                                      nn.Upsample(scale_factor=2, mode='bicubic'),
                                      )

    def forward(self, x, y, mode):
        b, c, h, w = x.size()
        x_org = x
        if mode == 'spe':
            b, c, h, w = x.size()
            x = self.groupnorm_1(x)
            y = self.groupnorm_1(y)
            qkv_1 = rearrange(self.qkv_1(x), "b (qkv heads c) h w -> (qkv) b heads c (h w)", heads=self.num_heads,
                              qkv=3)
            queries_1, keys_1, values_1 = qkv_1[0], qkv_1[1], qkv_1[2]
            qkv_2 = rearrange(self.qkv_2(y), "b (qkv heads c) h w -> (qkv) b heads c (h w)", heads=self.num_heads,
                              qkv=3)
            queries_2, keys_2, values_2 = qkv_2[0], qkv_2[1], qkv_2[2]
            keys_1 = F.softmax(keys_1, dim=-1)
            keys_2 = F.softmax(keys_2, dim=-1)
            att = torch.einsum('bhdn,bhen->bhde', keys_1, values_2)
            out = torch.einsum('bhde,bhdn->bhen', att, queries_1)
            out = rearrange(out, 'b heads c (h w) -> b (heads c) h w', heads=self.num_heads, h=h, w=w)
        else:
            x = self.groupnorm_2(x)
            y = self.groupnorm_2(y)
            if h == 512:
                times = h / 64
            else:
                times = h / 20
            n = np.log(times) / np.log(2)
            for i in range(int(n)):
                x = self.downsample[2 * i](x)
                x = self.downsample[2 * i + 1](x)

            for i in range(int(n)):
                y = self.downsample[2 * i](y)
                y = self.downsample[2 * i + 1](y)

            b, c, h, w = x.size()

            x = x.reshape(b, c, h * w).repeat(1, 1, 3)
            y = y.reshape(b, c, h * w).repeat(1, 1, 3)
            qkv_1 = rearrange(x, "b c (qkv heads h) -> (qkv) b heads h c", heads=self.num_heads, qkv=3)
            queries_1, keys_1, values_1 = qkv_1[0], qkv_1[1], qkv_1[2]
            qkv_2 = rearrange(y, "b c (qkv heads h) -> (qkv) b heads h c", heads=self.num_heads, qkv=3)
            queries_2, keys_2, values_2 = qkv_2[0], qkv_2[1], qkv_2[2]

            keys_1 = F.softmax(keys_1, dim=-1)
            keys_2 = F.softmax(keys_2, dim=-1)
            att = torch.einsum('bhdn,bhen->bhde', keys_1, values_2)
            out = torch.einsum('bhde,bhdn->bhen', att, queries_1)
            out = rearrange(out, 'b heads (h w) c -> b (heads c) h w', heads=self.num_heads, h=h, w=w)

            for i in range(int(n)):
                l = int(n) - 1 - i
                out = self.upsample[2 * l](out)
                out = self.upsample[2 * l + 1](out)

        return x_org + self.att * self.proj(out)


class ResBlock(nn.Module):
    def __init__(self, dim, dim_out, noise_level_emb_dim=None, dropout=0,
                 num_heads=1, use_affine_level=False, norm_groups=32, att=False):
        super().__init__()
        self.noise_func = FeatureWiseAffine(noise_level_emb_dim, dim_out, use_affine_level)
        self.block1 = Block(dim, dim_out, groups=norm_groups)
        self.block2 = Block(dim_out, dim_out, groups=norm_groups, dropout=dropout)
        self.res_conv = nn.Conv2d(dim, dim_out, 1) if dim != dim_out else nn.Identity()

    def forward(self, x, time_emb):
        y = self.block1(x)
        y = self.noise_func(y, time_emb)
        y = self.block2(y)
        x = y + self.res_conv(x)
        return x


class ResBlock_skip(nn.Module):
    def __init__(self, dim, dim_out, noise_level_emb_dim=None, dropout=0,
                 num_heads=1, use_affine_level=False, norm_groups=32, att=True):
        super().__init__()
        self.noise_func = FeatureWiseAffine(noise_level_emb_dim, dim_out, use_affine_level)
        self.block1 = Block(dim, dim_out, groups=norm_groups)
        self.block2 = Block(dim_out, dim_out, groups=norm_groups, dropout=dropout)
        self.res_conv = nn.Conv2d(dim, dim_out, 1) if dim != dim_out else nn.Identity()

    def forward(self, x):
        y = self.block1(x)

        return y + self.res_conv(x)


class SGPD(nn.Module):
    def __init__(self, in_channel=37, out_channel=34, skip_input=31, inner_channel=64, norm_groups=32,
                 channel_mults=[1, 2, 4, 8, 8], res_blocks=3, dropout=0, img_size=160):
        super().__init__()

        self_att = []
        cros_att = []
        dim_out = [inner_channel, inner_channel * 2, inner_channel * 2]
        for i in reversed(range(len(dim_out))):
            self_att.append(SelfAtt(dim_out[i], num_heads=1, norm_groups=norm_groups))

        self.self_att = nn.ModuleList(self_att)

        for j in reversed(range(len(dim_out))):
            cros_att.append(Cross_Att(dim_out[j], num_heads=1, norm_groups=norm_groups))

        self.cros_att = nn.ModuleList(cros_att)

        noise_level_channel = inner_channel
        self.noise_level_mlp = nn.Sequential(
            PositionalEncoding(inner_channel),
            nn.Linear(inner_channel, inner_channel * 4),
            Swish(),
            nn.Linear(inner_channel * 4, inner_channel)
        )

        num_mults = len(channel_mults)
        pre_channel = inner_channel
        pre_channel_skip = inner_channel
        feat_channels = [pre_channel]
        feat_channels_skips = [pre_channel]

        now_res = img_size

        # Downsampling stage of SGPD
        downs = [nn.Conv2d(in_channel, inner_channel, kernel_size=3, padding=1)]
        for ind in range(num_mults):
            is_last = (ind == num_mults - 1)
            channel_mult = inner_channel * channel_mults[ind]
            for _ in range(0, res_blocks):
                downs.append(ResBlock(
                    pre_channel, channel_mult, noise_level_emb_dim=noise_level_channel,
                    norm_groups=norm_groups, dropout=dropout))
                feat_channels.append(channel_mult)
                pre_channel = channel_mult
            if not is_last:
                downs.append(Downsample(pre_channel))
                feat_channels.append(pre_channel)
                now_res = now_res // 2
        self.downs = nn.ModuleList(downs)

        self.mid = nn.ModuleList([
            ResBlock(pre_channel, pre_channel, noise_level_emb_dim=noise_level_channel,
                     norm_groups=norm_groups, dropout=dropout),
            ResBlock(pre_channel, pre_channel, noise_level_emb_dim=noise_level_channel,
                     norm_groups=norm_groups, dropout=dropout, att=False)
        ])

        # Skip stage of SGPD
        skip_downs = [nn.Conv2d(skip_input, inner_channel, kernel_size=3, padding=1)]
        for ind in range(num_mults):
            is_last = (ind == num_mults - 1)
            channel_mult = inner_channel * channel_mults[ind]
            for _ in range(0, res_blocks):
                skip_downs.append(ResBlock_skip(
                    pre_channel_skip, channel_mult, noise_level_emb_dim=noise_level_channel,
                    norm_groups=norm_groups, dropout=dropout, att=False))
                pre_channel_skip = channel_mult
            if not is_last:
                feat_channels_skips.append(channel_mult)
                skip_downs.append(Downsample(pre_channel_skip))
                now_res = now_res // 2
        self.skip_downs = nn.ModuleList(skip_downs)

        # Upsampling stage of SGPD
        ups = []
        for ind in reversed(range(num_mults)):
            is_last = (ind < 1)
            channel_mult = inner_channel * channel_mults[ind]

            for i in range(0, res_blocks + 1):
                ups.append(ResBlock(
                    pre_channel + feat_channels.pop() * 2, channel_mult,
                    noise_level_emb_dim=noise_level_channel,
                    norm_groups=norm_groups, dropout=dropout))
                pre_channel = channel_mult

            if not is_last:
                ups.append(Upsample(pre_channel))
                now_res = now_res * 2

        self.ups = nn.ModuleList(ups)

        self.final_conv = Block(pre_channel, out_channel, groups=norm_groups)

    def forward(self, x, skip_input, noise_level, mode=None):
        # Embedding of time step with noise coefficient alpha
        t = self.noise_level_mlp(noise_level)

        feats_skip = []
        feats = []
        for layer in self.downs:
            if isinstance(layer, ResBlock):
                x = layer(x, t)
            else:
                x = layer(x)
            feats.append(x)

        k = 0
        for i, layer in enumerate(self.skip_downs):
            # skip_input =
            skip_input = layer(skip_input)
            feats_skip.append(skip_input)

        for layer in self.mid:
            x = layer(x, t)
        z = 0
        for i, layer in enumerate(self.ups):

            if isinstance(layer, ResBlock):
                if i == 0:
                    x = layer(torch.cat([x, feats.pop(), feats_skip.pop()], dim=1), t)
                elif isinstance(self.ups[i - 1], Upsample):
                    temp_feats_skip = feats_skip.pop()
                    temp_feats = feats.pop()
                    x = layer(torch.cat([x, self.self_att[z](temp_feats_skip),
                                         self.cros_att[z](temp_feats, temp_feats_skip, mode=mode)], dim=1), t)
                    z = z + 1
                else:
                    x = layer(torch.cat([x, feats.pop(), feats_skip.pop()], dim=1), t)
            else:
                x = layer(x)

        return self.final_conv(x)


"""
    Define Diffusion process framework to train desired model:
    Forward Diffusion process:
        Given original image x_0, apply Gaussian noise ε_t for each time step t
        After proper length of time step, image x_T reachs to pure Gaussian noise
    Objective of model f :
        model f is trained to predict actual added noise ε_t for each time step t
"""


class Diffusion(nn.Module):
    def __init__(self, model, device, img_size, LR_size, channels=3):
        super().__init__()
        self.channels = channels
        self.model = model.to(device)
        self.img_size = img_size
        self.LR_size = LR_size
        self.device = device
        # 粗配准
        self.CRN = AGCN(patch_size=160, dim=256, num_heads=4, in_ch_msi=4,
                                             in_ch_hsi=144).to(device)

        # self.upSample = nn.Upsample(scale_factor=4, mode='bicubic')
        self.downSample = nn.Upsample(scale_factor=0.25, mode='bicubic')
        self.upsample = nn.Upsample(scale_factor=4, mode='bicubic')

        # complementary fusion block
        self.fuse = nn.Sequential(
            nn.Conv2d(31 * 2, 31 * 2, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.Conv2d(31 * 2, 31, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.Conv2d(31, 31, kernel_size=3, stride=1, padding=1),
        ).to(device)

    def set_loss(self, loss_type):
        if loss_type == 'l1':
            self.loss_func = nn.L1Loss(reduction='sum')
        elif loss_type == 'l2':
            self.loss_func = nn.MSELoss(reduction='sum')
        else:
            raise NotImplementedError()

    def make_beta_schedule(self, schedule, n_timestep, linear_start=1e-4, linear_end=2e-2):
        if schedule == 'linear':
            betas = np.linspace(linear_start, linear_end, n_timestep, dtype=np.float64)
        elif schedule == 'warmup':
            warmup_frac = 0.1
            betas = linear_end * np.ones(n_timestep, dtype=np.float64)
            warmup_time = int(n_timestep * warmup_frac)
            betas[:warmup_time] = np.linspace(linear_start, linear_end, warmup_time, dtype=np.float64)
        elif schedule == "cosine":
            cosine_s = 8e-3
            timesteps = torch.arange(n_timestep + 1, dtype=torch.float64) / n_timestep + cosine_s
            alphas = timesteps / (1 + cosine_s) * math.pi / 2
            alphas = torch.cos(alphas).pow(2)
            alphas = alphas / alphas[0]
            betas = 1 - alphas[1:] / alphas[:-1]
            betas = betas.clamp(max=0.999)
        else:
            raise NotImplementedError(schedule)
        return betas

    def set_new_noise_schedule(self, schedule_opt):
        to_torch = partial(torch.tensor, dtype=torch.float32, device=self.device)

        betas = self.make_beta_schedule(
            schedule=schedule_opt['schedule'],
            n_timestep=schedule_opt['n_timestep'],
            linear_start=schedule_opt['linear_start'],
            linear_end=schedule_opt['linear_end']
        )
        betas = betas.detach().cpu().numpy() if isinstance(betas, torch.Tensor) else betas
        alphas = 1. - betas  # 每一个时间步的α，例如：betas为0.0001-0.002那么这个aplhas就是从大到小，但是每一个值都差不了多少，
        alphas_cumprod = np.cumprod(alphas, axis=0)  # 累乘
        alphas_cumprod_prev = np.append(1., alphas_cumprod[:-1])  # 给最开始填入一个1，并把最后一个值去除。

        self.sqrt_alphas_cumprod_prev = np.sqrt(np.append(1., alphas_cumprod))  # 对每一个数开平方
        # self.sqrt_alphas_cumprod_prev的数量比alphas_cumprod_prev多1
        self.num_timesteps = int(len(betas))
        # Coefficient for forward diffusion q(x_t | x_{t-1}) and others
        self.register_buffer('betas', to_torch(betas))
        self.register_buffer('alphas_cumprod', to_torch(alphas_cumprod))
        self.register_buffer('alphas_cumprod_prev', to_torch(alphas_cumprod_prev))
        self.register_buffer('pred_coef1', to_torch(np.sqrt(1. / alphas_cumprod)))  # 计算原始图像
        self.register_buffer('pred_coef2', to_torch(np.sqrt(1. / alphas_cumprod - 1)))  # 计算噪声

        # Coefficient for reverse diffusion posterior q(x_{t-1} | x_t, x_0)
        variance = betas * (1. - alphas_cumprod_prev) / (1. - alphas_cumprod)
        self.register_buffer('variance', to_torch(variance))
        # below: log calculation clipped because the posterior variance is 0 at the beginning of the diffusion chain
        self.register_buffer('posterior_log_variance_clipped', to_torch(np.log(np.maximum(variance, 1e-20))))
        self.register_buffer('posterior_mean_coef1',
                             to_torch(betas * np.sqrt(alphas_cumprod_prev) / (1. - alphas_cumprod)))
        self.register_buffer('posterior_mean_coef2',
                             to_torch((1. - alphas_cumprod_prev) * np.sqrt(alphas) / (1. - alphas_cumprod)))

    # Predict desired image x_0 from x_t with noise z_t -> Output is predicted x_0
    def predict_start(self, x_t, t, noise):
        return self.pred_coef1[t] * x_t - self.pred_coef2[t] * noise

    # Compute mean and log variance of posterior(reverse diffusion process) distribution
    def q_posterior(self, x_start, x_t, t):
        posterior_mean = self.posterior_mean_coef1[t] * x_start + self.posterior_mean_coef2[t] * x_t
        posterior_log_variance_clipped = self.posterior_log_variance_clipped[t]
        return posterior_mean, posterior_log_variance_clipped

    def p_mean_variance(self, x, t, clip_denoised: bool, condition_hrMS=None, condition_lrHS=None, SRF=None):
        batch_size, c = x.shape[0], condition_lrHS.shape[1]
        noise_level = torch.FloatTensor([self.sqrt_alphas_cumprod_prev[t + 1]]).repeat(batch_size, 1).to(x.device)
        # x_recon = self.predict_start(x, t, noise=self.model(torch.cat([condition_x, x], dim=1), noise_level))
        # lrHS_reg = self.CRN(x, condition_lrHS, self.downSample(condition_hrMS))
        if hasattr(self, 'lrHS_reg') and self.lrHS_reg is not None:
            lrHS_reg_input = self.lrHS_reg
        else:
            # 如果万一没有预计算（比如在训练阶段，或者其他调用方式），则回退到现场计算
            # 这里为了保险，还是可以用 x 或者 upsample(condition_lrHS)
            # 但既然是 fallback，保持原样或报错都可以
            lrHS_up = self.upsample(lrHS)
            lrHS_reg_input = self.CRN(lrHS_up, condition_lrHS, self.downSample(condition_hrMS))
        x_start = self.model(torch.cat([lrHS_reg_input, x], dim=1), condition_hrMS, condition_lrHS, SRF)


        posterior_mean = (
                self.posterior_mean_coef1[t] * x_start.clamp(-1, 1) +
                self.posterior_mean_coef2[t] * x
        )  # 例如：xt-1的均值

        posterior_variance = self.posterior_log_variance_clipped[t]

        mean, posterior_log_variance = posterior_mean, posterior_variance
        return mean, posterior_log_variance

    # Progress single step of reverse diffusion process
    # Given mean and log variance of posterior, sample reverse diffusion result from the posterior
    @torch.no_grad()
    def p_sample(self, img_noise, t, clip_denoised=True, condition_hrMS=None, condition_lrHS=None, SRF=None):
        mean, log_variance = self.p_mean_variance(x=img_noise, t=t, clip_denoised=clip_denoised,
                                                  condition_hrMS=condition_hrMS, condition_lrHS=condition_lrHS, SRF=SRF)
        noise = torch.randn_like(img_noise) if t > 0 else torch.zeros_like(img_noise)
        # 生成迭代的例如：Zt-1
        return mean + noise * (0.5 * log_variance).exp()

    # Progress whole reverse diffusion process
    @torch.no_grad()
    def super_resolution(self, gtHS, hrMS, lrHS, SRF):
        img = torch.randn_like(gtHS, device=gtHS.device)
        lrHS_up = self.upsample(lrHS)
        # 粗配准
        self.lrHS_reg = self.CRN(lrHS_up, lrHS, self.downSample(hrMS))
        # self.lrHS_reg = torch.zeros_like(self.upsample(lrHS))
        # save_M_dir="./result_C1"
        # os.makedirs(save_M_dir, exist_ok=True)
        # mask_easy = np.load("./t0050_mask_easy_top30.npy").astype(bool)
        # mask_hard =np.load("t0050_mask_hard_bottom10.npy").astype(bool)
        C1_easy_means = []
        C1_hard_means = []
        for i in reversed(range(0, self.num_timesteps, 1)):
            img = self.p_sample(img, i, condition_hrMS=hrMS, condition_lrHS=lrHS, SRF=SRF)
        #     flow_SF=self.model.FRN.selective_flow_field[0].float().cpu().numpy()#2 H W
        #     sf_u = flow_SF[0][mask_easy]
        #     sf_v = flow_SF[1][mask_easy]
        #     sf_u_hard = flow_SF[0][mask_hard]
        #     sf_v_hard = flow_SF[1][mask_hard]
        #     mag = np.sqrt(sf_u**2 + sf_v**2).mean()
        #     hard=np.sqrt(sf_u_hard**2 + sf_v_hard**2).mean()
        #     C1_easy_means.append(mag)
        #     C1_hard_means.append(hard)
        #     if(i==50):
        #         M = self.model.FRN.M[0, 0].float().cpu().numpy() # H W
        #         y_fake=self.model.FRN.y_fake[0].float().cpu().numpy()#3 H W
        #         ssim_score=self.model.FRN.ssim_score[0,0].float().cpu().numpy()#H W
        #         a = self.model.FRN.y_fake   # 或者你实际参与 ssim 的那个张量
        #         b = hrMS                    # 或对应的 reference

        #         print("y_fake:", a.min().item(), a.max().item(), a.mean().item())
        #         print("hrMS  :", b.min().item(), b.max().item(), b.mean().item())

        #         print(f"ssim_score: min={ssim_score.min():.6f}, max={ssim_score.max():.6f}, mean={ssim_score.mean():.6f}")
        #         flow=self.model.FRN.flow[0].float().cpu().numpy()#2 H W
        #         selective_flow_field=self.model.FRN.selective_flow_field[0].float().cpu().numpy()#2 H W
        #         np.save(os.path.join(save_M_dir, f"t{i:04d}_M.npy"), M)
        #         np.save(os.path.join(save_M_dir, f"t{i:04d}_y_fake.npy"), y_fake)
        #         np.save(os.path.join(save_M_dir, f"t{i:04d}_ssim.npy"), ssim_score)
        #         np.save(os.path.join(save_M_dir, f"t{i:04d}_flow.npy"), flow)
        #         np.save(os.path.join(save_M_dir, f"t{i:04d}_selective_flow.npy"), selective_flow_field)

        #         # 2) 做可视化元素
        #         y_rgb = normalize_rgb_3chw(y_fake)

        #         flow_rgb, flow_mag, flow_clip = flow_to_hsv_rgb(flow)
        #         sflow_rgb, sflow_mag, sflow_clip = flow_to_hsv_rgb(selective_flow_field, clip_mag=flow_clip)  # 用同一色标更可比

        #         # magnitude 的 vmax 用同一套（更可比）
        #         mag_vmax = max(np.percentile(flow_mag, 99.0), np.percentile(sflow_mag, 99.0)) + 1e-8

        #         # 3) 拼一张大图（2行×4列，够用）
        #         fig, axes = plt.subplots(2, 4, figsize=(16, 8), dpi=200)

        #         # Row 1
        #         axes[0, 0].imshow(y_rgb)
        #         axes[0, 0].set_title(f"y_fake (RGB proxy) @ t={i}")
        #         axes[0, 0].axis("off")

        #         im = axes[0, 1].imshow(M, vmin=0, vmax=1)
        #         axes[0, 1].set_title("M mask (0~1)")
        #         axes[0, 1].axis("off")
        #         fig.colorbar(im, ax=axes[0, 1], fraction=0.046, pad=0.04)

        #         # ssim_score 这里按 0~1 画（如果你 SSIM 范围不是 0~1，可改 vmin/vmax）
        #         im = axes[0, 2].imshow(ssim_score, vmin=-1, vmax=1)
        #         axes[0, 2].set_title("SSIM map")
        #         axes[0, 2].axis("off")
        #         fig.colorbar(im, ax=axes[0, 2], fraction=0.046, pad=0.04)

        #         im = axes[0, 3].imshow(flow_mag, vmin=0, vmax=mag_vmax)
        #         axes[0, 3].set_title("Flow magnitude")
        #         axes[0, 3].axis("off")
        #         fig.colorbar(im, ax=axes[0, 3], fraction=0.046, pad=0.04)

        #         # Row 2
        #         axes[1, 0].imshow(flow_rgb)
        #         axes[1, 0].set_title("Flow HSV (angle+hue, mag+value)")
        #         axes[1, 0].axis("off")

        #         im = axes[1, 1].imshow(sflow_mag, vmin=0, vmax=mag_vmax)
        #         axes[1, 1].set_title("Selective flow magnitude")
        #         axes[1, 1].axis("off")
        #         fig.colorbar(im, ax=axes[1, 1], fraction=0.046, pad=0.04)

        #         axes[1, 2].imshow(sflow_rgb)
        #         axes[1, 2].set_title("Selective flow HSV")
        #         axes[1, 2].axis("off")

        #         # 给一个“箭头图”（可选但很直观）
        #         step = max(min(flow.shape[1], flow.shape[2]) // 25, 6)  # 自动稀疏采样
        #         H, W = flow.shape[1], flow.shape[2]
        #         yy, xx = np.mgrid[0:H:step, 0:W:step]
        #         u = flow[0, 0:H:step, 0:W:step]
        #         v = flow[1, 0:H:step, 0:W:step]
        #         axes[1, 3].imshow(flow_mag, vmin=0, vmax=mag_vmax)
        #         axes[1, 3].quiver(xx, yy, u, v, angles="xy", scale_units="xy", scale=1.0, width=0.002)
        #         axes[1, 3].set_title("Flow quiver on magnitude")
        #         axes[1, 3].axis("off")

        #         plt.tight_layout()
        #         out_png = os.path.join(save_M_dir, f"t{i:04d}_FRN_visual.png")
        #         plt.savefig(out_png, bbox_inches="tight")
        #         plt.close(fig)

        #         print(f"[Saved] {out_png}")
        # np.save(os.path.join(save_M_dir, f"t{i:04d}_img.npy"), img.detach().float().cpu().numpy())
        # np.save("./result_C1/C1_easy_means.npy", np.array(C1_easy_means, dtype=np.float32))
        # np.save("./result_C1/C1_hard_means.npy", np.array(C1_hard_means, dtype=np.float32))
        return img


    def net(self, gtHS, hrMS, lrHS_reg, lrHS, SRF):

        gtHS = gtHS
        hrMS = hrMS
        lrHS_reg = lrHS_reg

        b, c, h, w = gtHS.shape
        # 生成一个形状为 (b,) 的随机整数张量，每个元素取值范围是 [1, schedule_opt['n_timestep'] - 1]。
        t = torch.randint(1, schedule_opt['n_timestep'], size=(b,))

        sqrt_alpha_cumprod_t = extract(torch.from_numpy(self.sqrt_alphas_cumprod_prev), t, gtHS.shape)
        sqrt_alpha = sqrt_alpha_cumprod_t.view(-1, 1, 1, 1).type(torch.float32).to(gtHS.device)
        noise = torch.randn_like(gtHS).to(gtHS.device)
        # Perturbed image obtained by forward diffusion process at random time step t
        x_noisy = sqrt_alpha * gtHS + (1 - sqrt_alpha ** 2).sqrt() * noise
        # The bilateral model predict actual x0 added at time step t
        outputs = self.model(torch.cat([lrHS_reg, x_noisy], 1), hrMS, lrHS, SRF)

        # complementary fusion
        Loss = self.loss_func(outputs, gtHS)
        Loss = Loss / (gtHS.shape[0] * gtHS.shape[1] * gtHS.shape[2] * gtHS.shape[3])
        return Loss

    def forward(self, gtHS, hrMS, lrHS, SRF, *args, **kwargs):
        x = lrHS
        y = hrMS
        x = self.upsample(x)
        # 粗配准
        self.lrHS_reg = self.CRN(x, lrHS, self.downSample(y))
        return self.net(gtHS, hrMS, self.lrHS_reg, lrHS, SRF, *args, **kwargs)


# Class to train & test desired model
class SR3():
    def __init__(self, device, img_size, LR_size, loss_type, dataloader, testloader,
                 schedule_opt, save_path, load_path=None, load=True,
                 in_channel=62, out_channel=31, inner_channel=64, norm_groups=8,
                 channel_mults=(1, 2, 4, 8, 8), res_blocks=3, dropout=0, lr=1e-3, distributed=False, SRF=None):
        super(SR3, self).__init__()
        self.dataloader = dataloader
        self.testloader = testloader
        self.device = device
        self.save_path = save_path
        self.img_size = img_size
        self.LR_size = LR_size
        self.SRF = SRF
        model = CCFnet(patch_size=160, in_ch_msi=IN_CH_MSI, in_ch_hsi=IN_CH_HSI).to(device=DEVICE)

        self.sr3 = Diffusion(model, device, img_size, LR_size, out_channel)
        # Apply weight initialization & set loss & set noise schedule
        # self.sr3.apply(self.weights_init_orthogonal)
        self.sr3.set_loss(loss_type)
        self.sr3.set_new_noise_schedule(schedule_opt)

        if distributed:
            assert torch.cuda.is_available()
            self.sr3 = nn.DataParallel(self.sr3)

        self.optimizer = torch.optim.Adam(self.sr3.parameters(), lr=lr)

        params = sum(p.numel() for p in self.sr3.parameters())
        print(f"Number of model parameters : {params}")

        if load:
            self.load(load_path)

    def weights_init_orthogonal(self, m):
        classname = m.__class__.__name__
        if classname.find('Conv') != -1:
            init.orthogonal_(m.weight.data, gain=1)
            if m.bias is not None:
                m.bias.data.zero_()
        elif classname.find('Linear') != -1:
            init.orthogonal_(m.weight.data, gain=1)
            if m.bias is not None:
                m.bias.data.zero_()
        elif classname.find('BatchNorm2d') != -1:
            init.constant_(m.weight.data, 1.0)
            init.constant_(m.bias.data, 0.0)

    def train(self, epoch, verbose):

        train = True
        start_epoch = 0
        for i in range(start_epoch, epoch):
            i = i
            train_loss = 0
            self.sr3.train()
            randn1 = np.random.randint(0, 100)

            if train:
                for step, [gtHS, hrMS, lrHS] in enumerate(tqdm(self.dataloader)):
                    # 高光谱和全色图像
                    gtHS = gtHS.type(torch.float32).to(DEVICE)
                    hrMS = hrMS.type(torch.float32).to(DEVICE)
                    lrHS = lrHS.type(torch.float32).to(DEVICE)

                    self.optimizer.zero_grad()
                    loss = self.sr3(gtHS, hrMS, lrHS)
                    # print(f"Loss 的形狀是: {loss.shape}")
                    # print(f"Loss 的數值 (前5個): {loss.flatten()[:5]}")
                    scalar_loss = loss.mean()
                    scalar_loss.backward()
                    self.optimizer.step()

                    train_loss += scalar_loss.item()

                print('epoch: {}'.format(i))
                print('损失函数:')
                x = PrettyTable()
                x.add_column("loss", ['value'])
                x.add_column("loss_all", [train_loss / float(len(self.dataloader))])
                print(x)

            if (i + 1) % verbose == 0:
                self.sr3.eval()
                test_data = copy.deepcopy(next(iter(self.testloader)))
                [gtHS, hrMS, lrHS] = test_data
                gtHS = gtHS.type(torch.float32).to(DEVICE)
                hrMS = hrMS.type(torch.float32).to(DEVICE)
                lrHS = lrHS.type(torch.float32).to(DEVICE)

                b, c, h, w = gtHS.shape

                randn3 = np.random.randint(0, b)
                gtHS = gtHS[randn3]
                hrMS = hrMS[randn3]
                lrHS = lrHS[randn3]

                self.save(self.save_path, i)
                fuse_result = self.test(gtHS.unsqueeze(0), lrHS.unsqueeze(0), hrMS.unsqueeze(0))

                Metric = iv.spectra_metric(fuse_result[0].permute(1, 2, 0).cpu().detach().numpy(),
                                           gtHS.permute(1, 2, 0).cpu().detach().numpy(),1, 4)
                PSNR = Metric.PSNR()
                SAM = Metric.SAM()
                SSIM = Metric.SSIM()
                MSE = Metric.MSE()
                ERGAS = Metric.ERGAS()
                print('评价指标:')
                y = PrettyTable()
                y.add_column("Index", ['value'])
                y.add_column("PSNR", [PSNR])
                y.add_column("SAM", [SAM])
                y.add_column("SSIM", [SSIM])
                y.add_column("MSE", [MSE])
                y.add_column("ERGAS", [ERGAS])
                print(y)
                # 定义 TXT 文件路径
                txt_file = "../model/Houston/LRHS_SF/evaluation_metrics.txt"

                # 写入数据
                with open(txt_file, "a") as file:
                    file.write(f"Index: {i}, PSNR: {PSNR}, SAM: {SAM}, SSIM: {SSIM}, MSE: {MSE}, ERGAS: {ERGAS}\n")
    import torch

    @torch.no_grad()
    def per_band_corrcoef_torch(self, pred_bchw, gt_bchw, eps=1e-8):
        assert pred_bchw.shape == gt_bchw.shape
        B, C, H, W = pred_bchw.shape

        x = pred_bchw.reshape(B, C, -1)
        y = gt_bchw.reshape(B, C, -1)

        x = x - x.mean(dim=-1, keepdim=True)
        y = y - y.mean(dim=-1, keepdim=True)

        num = (x * y).mean(dim=-1)  # [B,C]
        den = (x.pow(2).mean(dim=-1).sqrt() * y.pow(2).mean(dim=-1).sqrt()) + eps

        r = num / den
        return r.mean(dim=0)        # [C]

    def test_(self):

        self.sr3.eval()
        # test_data = copy.deepcopy(next(iter(self.testloader)))
        # [gtHS, hrMS, lrHS, HSI_Patch, MSI_Patch2] = test_data
        metrics_sum = {
            'PSNR': 0,
            'SAM': 0,
            'SSIM': 0,
            'MSE': 0,
            'ERGAS': 0,
            'SPACC': 0,
            'SPECC': 0,
            'QNR':0
        }
        count = 0
        for step, [gtHS, hrMS, lrHS] in enumerate(tqdm(self.dataloader)):
            # if(step!=27): continue
            gtHS = gtHS.type(torch.float32).to(DEVICE)
            hrMS = hrMS.type(torch.float32).to(DEVICE)
            lrHS = lrHS.type(torch.float32).to(DEVICE)

            b, c, h, w = gtHS.shape

            randn3 = np.random.randint(0, b)
            gtHS = gtHS[randn3]
            hrMS = hrMS[randn3]
            lrHS = lrHS[randn3]


            fuse_result = self.test(gtHS.unsqueeze(0), lrHS.unsqueeze(0), hrMS.unsqueeze(0))
           

            save_result=(fuse_result+1.0)/2.0
            save_result=save_result.detach().squeeze(0).cpu().numpy()
            save_dir='../Proposed/NewYork/band'
            save_path = os.path.join(save_dir, f"{step}.mat")
            sio.savemat(save_path, {"fused": save_result}, do_compression=True)

            gt = gtHS.permute(1, 2, 0).detach().cpu().numpy()
            pred = fuse_result[0].permute(1, 2, 0).detach().cpu().numpy()

            # [-1,1] -> [0,1]
            gt01 = np.clip((gt + 1.0) / 2.0, 0.0, 1.0)
            pred01 = np.clip((pred + 1.0) / 2.0, 0.0, 1.0)
            # 然后分别算一次 SAM/ERGAS：
            Metric_raw = iv.spectra_metric(gt01,
                                        pred01, 1, 4)
            PSNR =Metric_raw.PSNR()
            SAM = Metric_raw.SAM()
            SSIM = Metric_raw.SSIM()
            MSE = Metric_raw.MSE()
            ERGAS = Metric_raw.ERGAS()
            SPECC = Metric_raw.CC()
            SPACC = CC_function(gt01.transpose(2, 0, 1),  pred01.transpose(2, 0, 1))


            print('评价指标:')
            up=nn.Upsample(scale_factor=4, mode='bicubic')
            print("lrHS shape:", lrHS.shape)
            Zup = up(lrHS.unsqueeze(0))
            QNR, Dlam, Ds = compute_qnr(
            pred_hsi_bchw=Zup,   # [1,110,128,128] in [-1,1]
            lr_hsi_bchw=lrHS.unsqueeze(0) if lrHS.ndim==3 else lrHS,  # 保证是 [1,110,32,32]
            hr_ms_bchw=hrMS.unsqueeze(0) if hrMS.ndim==3 else hrMS,   # [1,3,128,128]
            srf=self.SRF,
            scale=4
            )

            print(f"QNR={QNR:.6f}, D_lambda={Dlam:.6f}, D_s={Ds:.6f}")

            y = PrettyTable()
            y.add_column("Index", ['value'])
            y.add_column("PSNR", [PSNR])
            y.add_column("SAM", [SAM])
            y.add_column("SSIM", [SSIM])
            y.add_column("MSE", [MSE])
            y.add_column("ERGAS", [ERGAS])
            y.add_column("SPECC", [SPECC])
            y.add_column("SPACC", [SPACC])
            y.add_column("QNR", [QNR])
            print(y)
            # 累加各项指标
            metrics_sum['PSNR'] += PSNR
            metrics_sum['SAM'] += SAM
            metrics_sum['SSIM'] += SSIM
            metrics_sum['MSE'] += MSE
            metrics_sum['ERGAS'] += ERGAS
            metrics_sum['SPECC'] += SPECC
            metrics_sum['SPACC'] += SPACC
            metrics_sum['QNR'] += QNR
            count += 1
            # 计算均值并打印
        if count > 0:
            avg_metrics = {k: v / count for k, v in metrics_sum.items()}
            y = PrettyTable()
            y.add_column("Index", ['Average'])
            y.add_column("PSNR", [avg_metrics['PSNR']])
            y.add_column("SAM", [avg_metrics['SAM']])
            y.add_column("SSIM", [avg_metrics['SSIM']])
            y.add_column("MSE", [avg_metrics['MSE']])
            y.add_column("ERGAS", [avg_metrics['ERGAS']])
            y.add_column("SPACC", [avg_metrics['SPACC']])
            y.add_column("SPECC", [avg_metrics['SPECC']])
            y.add_column("QNR", [avg_metrics['QNR']])
            print("\nAverage Metrics:")
            print(y)

    def test(self, gtHS, lrHS, hrMS):
        lrHS = lrHS
        hrMS = hrMS
        gtHS = gtHS
        self.sr3.eval()
        with torch.no_grad():
            if isinstance(self.sr3, nn.DataParallel):
                result_SR = self.sr3.module.super_resolution(gtHS, hrMS, lrHS, self.SRF)
            else:
                result_SR = self.sr3.super_resolution(gtHS, hrMS, lrHS, self.SRF)
        self.sr3.train()
        return result_SR

    def save(self, save_path, i):
        network = self.sr3
        if isinstance(self.sr3, nn.DataParallel):
            network = network.module
        state_dict = network.state_dict()
        for key, param in state_dict.items():
            state_dict[key] = param.cpu()
        torch.save(state_dict, save_path + 'SR3_model_epoch-{}.pt'.format(i))

    def load(self, load_path):
        network = self.sr3
        if isinstance(self.sr3, nn.DataParallel):
            network = network.module
        network.load_state_dict(torch.load(load_path))
        print("Model loaded successfully")


if __name__ == "__main__":
    batch_size = 12
    LR_size = 40
    img_size = 160

    # 超参数
    EPOCH = 1000
    BATCHSIZE = 8
    DEVICE = torch.device('cuda:3' if torch.cuda.is_available() else 'cpu')
    PATCH_SIZE = 16
    IN_CH_HSI = 144
    IN_CH_MSI = 4
    SRF = sio.loadmat("../data/Houston/srf_matrix.mat")
    SRF = SRF['SRF']


    test_datasat = Datasat('test', 160, IN_CH_HSI=IN_CH_HSI, IN_CH_MSI=IN_CH_MSI)
    test_loader = DataLoader(test_datasat, batch_size=1, shuffle=False, num_workers=0)

    cuda = torch.cuda.is_available()
    device = torch.device("cuda:3" if cuda else "cpu")
    schedule_opt = {'schedule': 'linear', 'n_timestep': 2000, 'linear_start': 1e-4, 'linear_end': 0.002}

    sr3 = SR3(device, img_size=img_size, LR_size=LR_size, loss_type='l1',
              dataloader=test_loader, testloader=test_loader, schedule_opt=schedule_opt,
              save_path='../model/Houston/LRHS_SF/',
              load_path='../model/Houston/LRHS_Elastic1000_C1/SR3_model_epoch-3249.pt', load=True,
              inner_channel=64,
              norm_groups=16, channel_mults=(1, 2, 2, 2), dropout=0, res_blocks=2, lr=1e-4, distributed=False, SRF=SRF)

    sr3.test_()



