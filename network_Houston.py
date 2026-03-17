import numpy as np
import skimage
import torchvision
from matplotlib import pyplot as plt
from torch import nn
import torch
import torch.nn.functional as F
import einops
from einops import rearrange
from scipy.ndimage import gaussian_filter
import math

DEVICE = 'cuda:1'

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_dct import dct_2d, idct_2d
from ssim import SSIM


def match_mean_std(src, ref, eps=1e-6):
    # src/ref: [B,C,H,W]
    src_m = src.mean(dim=(2,3), keepdim=True)
    src_s = src.std(dim=(2,3), keepdim=True)
    ref_m = ref.mean(dim=(2,3), keepdim=True)
    ref_s = ref.std(dim=(2,3), keepdim=True)
    return (src - src_m) / (src_s + eps) * ref_s + ref_m

class FrequencyEdgeEnhancer(nn.Module):
    def __init__(self, in_ch, threshold=0.1, scale=0.01):
        super().__init__()
        self.convh = nn.Conv2d(in_ch, 1, 3, 1, 1, bias=False)
        self.threshold = threshold
        self.scale = scale
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):  # x: (B, C, H, W)
        # 1) DCT
        x_freq = dct_2d(x, norm='ortho')

        # 2) 根据幅值做掩膜：|freq| < threshold 保留（更多高频），其余抑制
        mask = (torch.abs(x_freq) < self.threshold).float()
        x_masked = x_freq * mask  # 仿照 Registration 的写法

        # 3) IDCT 回空间
        x_high = idct_2d(x_masked, norm='ortho')

        # 4) 卷积 + Sigmoid 生成边缘权重
        edge_weight = self.scale * self.sigmoid(self.convh(x_high))  # (B,1,H,W)

        # 5) 用边缘权重增强原图
        x_enh = x + edge_weight * x

        return x_enh
class UGFR(nn.Module):
    def __init__(self, in_ch_hsi, in_ch_msi, embed_dim=128,
                 search_radius=2, max_flow_magnitude=2.0):
        """
        OOM-Safe 架构，使用“相关性体积”替代“全局交叉注意力”。

        search_radius: 我们假设粗配准后，精细位移在 +/- 4 像素范围内。
                       总搜索窗口将是 (2*4+1) * (2*4+1) = 9x9。
        """
        super().__init__()

        # --- 2. “相关性” 特征投影层 ---
        self.q_proj = nn.Conv2d(in_ch_hsi, embed_dim, kernel_size=1)
        self.k_proj = nn.Conv2d(in_ch_msi, embed_dim, kernel_size=1)
        self.search_radius = search_radius

        # 9x9 窗口 = 81 个相关性得分
        correlation_channels = (2 * self.search_radius + 1) ** 2

        # --- 3. “位移形变场” 回归器 (Flow Regressor) ---
        # 它的输入是所有信息的集合：
        # 1. HSI 特征 (embed_dim 通道)
        # 2. 相关性体积 (correlation_channels 通道)
        # 3. 掩码 M (1 通道)
        regressor_in_channels = embed_dim + embed_dim + correlation_channels + 1

        self.flow_regressor = nn.Sequential(
            nn.Conv2d(regressor_in_channels, 128, kernel_size=3, padding=1),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(128, 64, kernel_size=3, padding=1),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(64, 2, kernel_size=1),  # (B, 2, H, W) -> (dx, dy)
            nn.Tanh()  # 平滑限制到 [-1, 1]
        )

        self.max_flow_magnitude = max_flow_magnitude

        # 初始化最后一个卷积层
        self.flow_regressor[-2].weight.data.zero_()
        self.flow_regressor[-2].bias.data.zero_()
        self.Conv_half = nn.Conv2d(in_ch_hsi * 2, in_ch_hsi, 1, 1, 0)
        self.freq_enhancer = FrequencyEdgeEnhancer(in_ch=in_ch_hsi * 2, threshold=0.1)
        self.ssim = SSIM(9)

    @staticmethod
    def compute_correlation_volume(query, key, search_radius=2):
        """
        (静态方法)
        高效地计算相关性体积。
        query: (B, C, H, W) - 来自 HSI
        key:   (B, C, H, W) - 来自 MSI
        """
        B, C, H, W = query.shape
        R = search_radius

        # 0. 归一化特征 (可选，但通常能稳定训练)
        query = F.normalize(query, p=2, dim=1)
        key = F.normalize(key, p=2, dim=1)

        # 1. 准备 key_padded，以便安全地 "移动"
        # 我们需要在左右/上下各填充 R 像素
        key_padded = F.pad(key, [R, R, R, R], mode='replicate')

        correlation_list = []

        # 2. 迭代搜索窗口
        for dy in range(-R, R + 1):
            for dx in range(-R, R + 1):
                # 3. "移动" key, 裁剪回 (H, W) 尺寸
                # H 维度: [R + dy : R + dy + H]
                # W 维度: [R + dx : R + dx + W]
                key_shifted = key_padded[:, :, R + dy: R + dy + H, R + dx: R + dx + W]

                # 4. 计算点积 (相似度)
                # (B, C, H, W) * (B, C, H, W) -> (B, C, H, W)
                dot_product = query * key_shifted

                # 5. 沿着 C 维度求和, 得到 (B, 1, H, W) 的相似度图
                correlation_score = torch.sum(dot_product, dim=1, keepdim=True)
                correlation_list.append(correlation_score)

        # 6. 堆叠成 (B, (2R+1)^2, H, W) 的体积
        # (2R+1)**2 = (e.g., 81)
        correlation_volume = torch.cat(correlation_list, dim=1)
        return correlation_volume

    def warp(self, x, flow):
        # (这个辅助函数保持不变)
        B, C, H, W = x.shape
        affine_matrix = torch.tensor([[1., 0., 0.], [0., 1., 0.]],
                                     dtype=x.dtype, device=x.device)
        affine_matrix = affine_matrix.unsqueeze(0).repeat(B, 1, 1)
        identity_grid = F.affine_grid(affine_matrix, x.size(),
                                      align_corners=False)
        flow_permuted = flow.permute(0, 2, 3, 1)
        scale = torch.tensor([2.0 / W, 2.0 / H],
                             dtype=x.dtype, device=x.device).view(1, 1, 1, 2)
        flow_normalized = flow_permuted * scale
        Grid = identity_grid + flow_normalized
        x_warped = F.grid_sample(x, Grid, mode='bilinear',
                                 padding_mode='reflection',
                                 align_corners=False)
        return x_warped

    def forward(self, x, y, SRF):
        # x: (B, C_hsi, H, W)
        # y: (B, C_msi, H, W)
        # x = self.freq_enhancer(x)  # 频率增强
        x = self.Conv_half(x)
        x_original = x
        if not isinstance(SRF, torch.Tensor):
            srf_weight = torch.from_numpy(SRF).float()
        else:
            srf_weight = SRF.float()

        srf_weight = srf_weight.to(x.device)

        # 2. 调整维度以适配 conv2d: (Out, In) -> (Out, In, 1, 1)
        if srf_weight.dim() == 2:
            srf_weight = srf_weight.view(srf_weight.size(0), srf_weight.size(1), 1, 1)

        # 3. 执行投影 HSI -> Simulated MSI
        # x: (B, C_hsi, H, W) * weight: (C_msi, C_hsi, 1, 1) -> (B, C_msi, H, W)
        x_projected = F.conv2d(x, srf_weight)
        x_projected = x_projected.clamp(-1, 1)  # 进行大小限制
        x_aligned = match_mean_std(x_projected, y).clamp(-1, 1)
        ssim_score = self.ssim(x_aligned, y)

        difficulty_map = ((1.0 - ssim_score).unsqueeze(1) / 2.0).clamp(0, 1)
        q_feat = self.q_proj(x)
        k_feat = self.k_proj(y)

        # --- 3. 计算相关性体积 (OOM-Safe) ---
        # (B, 25, H, W)
        corr_volume = self.compute_correlation_volume(
            q_feat, k_feat, self.search_radius
        )
        regressor_input = torch.cat([q_feat, k_feat, corr_volume, difficulty_map], dim=1)

        flow_normalized = self.flow_regressor(regressor_input)
        flow_field = flow_normalized * self.max_flow_magnitude
        selective_flow_field = flow_field * difficulty_map
        x_warped = self.warp(x_original, selective_flow_field)
        return x_warped


def robust_pca_lowrank(X, k, center):
    """
    一個更穩健的 pca_lowrank 封裝，
    1. 使用與數據尺度相關的隨機噪聲來穩定 SVD。
    2. 在 GPU SVD 失敗時能回退到 CPU。
    """
    X = X.to(dtype=torch.float32)

    if center:
        Xc = X - X.mean(dim=-2, keepdim=True)
    else:
        Xc = X

    # --- 1. 計算穩健的擾動矩陣 (只做一次) ---
    # 這是解決 NaN 梯度的關鍵
    with torch.no_grad():
        # 根據數據標準差決定噪聲尺度
        std = Xc.std(dim=(-2, -1), keepdim=True)
        std[std == 0] = 1.0  # 避免除以零或乘以零
    # 生成與數據尺度相關的微小隨機噪聲
    noise = torch.randn_like(Xc) * std * 1e-6
    Xc_stable = Xc + noise

    # --- 2. 執行 SVD，並加入 GPU->CPU 的容錯機制 ---
    try:
        # 優先嘗試在 GPU 上對穩定後的矩陣進行 SVD
        U, S, Vh = torch.linalg.svd(Xc_stable, full_matrices=False)
    except RuntimeError as e:
        # 如果 GPU 失敗，打印錯誤並回退到 CPU
        print(f"GPU SVD failed with error: {e}. Falling back to CPU.")
        U, S, Vh = torch.linalg.svd(Xc_stable.cpu(), full_matrices=False)
        U, S, Vh = U.to(Xc.device), S.to(Xc.device), Vh.to(Xc.device)

    if k is not None:
        U, S, Vh = U[..., :, :k], S[..., :k], Vh[..., :k, :]

    V = Vh.transpose(-2, -1)
    return U, S, V


def PCA_Batch_Feat(X, k=1, center=True):
    """
    param X: BxCxHxW
    param k: scalar
    return:
    """
    B, C, H, W = X.shape
    X = X.permute(0, 2, 3, 1)  # BxHxWxC
    X = X.reshape(B, H * W, C)
    # 利用主成分分析把原来的C降维成了K
    U, S, V = robust_pca_lowrank(X, k=k, center=center)
    Y = torch.bmm(X, V[:, :, :k])

    Y = Y.reshape(B, H, W, k)
    Y = Y.permute(0, 3, 1, 2)  # BxkxHxW
    Y = Y.repeat(1, 256, 1, 1)

    # 这里可以用一个卷积升维来代替
    # up = nn.Conv2d(k, 256, kernel_size=1, bias=False)
    # Y = up(Y)  # [B, 256, H, W]

    return Y


class ConvGuidedFilter(nn.Module):
    def __init__(self, radius=1, norm=nn.BatchNorm2d):
        super(ConvGuidedFilter, self).__init__()
        # 其实这个就是 Mean Filter
        self.box_filter = nn.Conv2d(256, 256, kernel_size=3, padding=radius, dilation=radius, bias=False)
        self.conv_a = nn.Sequential(nn.Conv2d(512, 256, kernel_size=1, bias=False),
                                    norm(256),
                                    nn.ReLU(inplace=True),
                                    nn.Conv2d(256, 256, kernel_size=1, bias=False),
                                    norm(256),
                                    nn.ReLU(inplace=True),
                                    nn.Conv2d(256, 256, kernel_size=1, bias=False))
        self.box_filter.weight.data[...] = 1.0

    def forward(self, x_lr, y_Guide):
        y_lr = PCA_Batch_Feat(y_Guide)
        b, c, h_lrx, w_lrx = x_lr.size()

        N = self.box_filter(x_lr.data.new().resize_((b, c, h_lrx, w_lrx)).fill_(1.0))  # 构建一个和x_lr大小一样的但是内部的数据都是1的tensor
        # 下面几个计算公式与引导滤波一致
        # mean_x
        mean_x = self.box_filter(x_lr) / N
        # mean_y
        mean_y = self.box_filter(y_lr) / N
        # cov_xy
        cov_xy = self.box_filter(x_lr * y_lr) / N - mean_x * mean_y
        # var_x
        var_x = self.box_filter(x_lr * x_lr) / N - mean_x * mean_x

        # A 这里引入了卷积求解 ak
        A = self.conv_a(torch.cat([cov_xy, var_x], dim=1))
        # b
        b = mean_y - A * mean_x

        # 最终用双线性插值，放大特征图，获得最终的大尺寸的输出 O_H

        return A * x_lr + b


class ResBlock(nn.Module):
    def __init__(self, in_channel, out_channel, strides=1, same_shape=True):
        super(ResBlock, self).__init__()
        self.same_shape = same_shape
        # if not same_shape:
        #     strides = 2
        self.strides = strides
        self.block = nn.Sequential(
            nn.Conv2d(in_channel, out_channel, kernel_size=3, stride=strides, padding=1, bias=False),
            nn.BatchNorm2d(out_channel),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channel, out_channel, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channel)

        )
        if not same_shape:
            self.conv3 = nn.Conv2d(in_channel, out_channel, kernel_size=1, stride=strides, bias=False)
            self.bn3 = nn.BatchNorm2d(out_channel)

    def forward(self, x):
        out = self.block(x)
        if not self.same_shape:
            x = self.bn3(self.conv3(x))
        return F.relu(out + x)


class MSA(nn.Module):
    def __init__(self, num_vector, num_heads_column, heads_number):
        super(MSA, self).__init__()
        self.num_vector = num_vector
        self.num_heads_column = num_heads_column
        self.heads_number = heads_number
        self.to_q = nn.Linear(num_vector, num_heads_column * heads_number, bias=False)
        self.to_k = nn.Linear(num_vector, num_heads_column * heads_number, bias=False)
        self.to_v = nn.Linear(num_vector, num_heads_column * heads_number, bias=False)
        self.rescale = nn.Parameter(torch.ones(heads_number, 1, 1))  # 权重参数*CORE
        self.proj = nn.Linear(num_heads_column * heads_number, num_vector)
        self.pos_emb = nn.Sequential(
            nn.Linear(num_heads_column * heads_number, num_vector),
            nn.modules.activation.GELU(),
            nn.Linear(num_vector, num_vector),
        )

    def forward(self, x_in):
        b, n, c = x_in.shape
        x = x_in

        q_inp = self.to_q(x)
        k_inp = self.to_k(x)
        v_inp = self.to_v(x)
        q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> b h n d', h=self.heads_number),
                      (q_inp, k_inp, v_inp))
        v = v
        q = q.transpose(-2, -1)  # q,k,v: b,heads,c,hw
        k = k.transpose(-2, -1)
        v = v.transpose(-2, -1)
        q = F.normalize(q, dim=-1, p=2)
        k = F.normalize(k, dim=-1, p=2)
        attn = (k @ q.transpose(-2, -1))  # A = K^T*Q
        attn = attn * self.rescale
        attn = attn.softmax(dim=-1)
        x = attn @ v  # b,heads,d,hw
        x = x.permute(0, 3, 1, 2)  # Transpose
        x = x.reshape(b, n, self.heads_number * self.num_heads_column)
        out_c = self.proj(x)

        out_p = self.pos_emb(v_inp)

        return out_c + out_p


class Degenerate(torch.nn.Module):
    def __init__(self):
        super(Degenerate, self).__init__()
        self.c = 110

    def forward(self, output, coeff, sf, w1):
        PLR = nn.functional.conv2d(output, coeff, bias=None, stride=sf, padding=int((w1 - 1) / 2), groups=self.c)
        return PLR


def batch_index_select(x, idx):
    if len(x.size()) == 3:
        B, N, C = x.size()
        N_new = idx.size(1)
        offset = torch.arange(B, dtype=torch.long, device=x.device).view(B, 1) * N
        idx = idx + offset
        out = x.reshape(B * N, C)[idx.reshape(-1)].reshape(B, N_new, C)
        return out
    elif len(x.size()) == 2:
        B, N = x.size()
        N_new = idx.size(1)
        offset = torch.arange(B, dtype=torch.long, device=x.device).view(B, 1) * N
        idx = idx + offset
        out = x.reshape(B * N)[idx.reshape(-1)].reshape(B, N_new)
        return out
    else:
        raise NotImplementedError


def batch_index_fill2(x, x1, idx1):
    B, N, C = x.size()
    B, N1, C = x1.size()
    dev = x.device
    x1 = x1.to(dev, dtype=x.dtype)
    offset = torch.arange(B, dtype=torch.long, device=x.device).view(B, 1)
    idx1 = idx1.to(dev, dtype=torch.long)
    idx1 = idx1 + offset * N

    x = x.reshape(B * N, C)

    x[idx1.reshape(-1)] = x1.reshape(B * N1, C)
    x = x.reshape(B, N, C)
    return x


class Transformer(nn.Module):
    def __init__(self, x_channel):
        super(Transformer, self).__init__()
        self.saln1 = nn.LayerNorm(x_channel)
        self.saln2 = nn.LayerNorm(x_channel)
        self.sa = MSA(x_channel, x_channel // 4, 4)
        self.re_conv1 = nn.Sequential(
            nn.Linear(x_channel, x_channel // 2, bias=False),
            nn.LeakyReLU(0.1),
            nn.Linear(x_channel // 2, x_channel, bias=False),
        )

    def forward(self, v1):
        nor_v1 = self.saln1(v1)
        re_fea1 = self.sa(nor_v1) + v1
        norre_fea1 = self.saln2(re_fea1)
        refine1 = self.re_conv1(norre_fea1) + re_fea1

        return refine1


class Loss_SAM3(nn.Module):
    def __init__(self):
        super(Loss_SAM3, self).__init__()
        self.eps = 2.2204e-16  # torch.finfo(torch.float32).eps  # Minimum positive value of torch float32
        self.cosine_similarity = nn.CosineSimilarity(dim=1, eps=self.eps)

    def forward(self, im1, im2):
        assert im1.shape == im2.shape
        B, C, H, W = im1.shape

        # Reshape images
        im1 = im1.contiguous().view(B, C, -1).permute(0, 2, 1)  # Shape: [B, H*W, C]
        im2 = im2.contiguous().view(B, C, -1).permute(0, 2, 1)  # Shape: [B, H*W, C]

        core = torch.mul(im1, im2)  # Element-wise multiplication
        mole = torch.sum(core, dim=2)  # Sum along the channel dimension

        im1_norm = torch.sqrt(torch.sum(torch.square(im1), dim=2))  # compute 欧式distance
        im2_norm = torch.sqrt(torch.sum(torch.square(im2), dim=2))
        deno = torch.mul(im1_norm, im2_norm)

        sam = torch.acos(((mole + self.eps) / (deno + self.eps)).clamp(-1, 1))
        sam_deg = torch.rad2deg(sam)

        return sam_deg


def get_gaussian_kernel2d(kernel_size, sigma, channels, device, dtype):
    # 创建高斯核
    ax = torch.arange(-kernel_size // 2 + 1., kernel_size // 2 + 1., device=device, dtype=dtype)
    xx, yy = torch.meshgrid(ax, ax, indexing='ij')
    kernel = torch.exp(-(xx ** 2 + yy ** 2) / (2. * sigma ** 2))
    kernel = kernel / kernel.sum()
    # 扩展为 Depthwise 卷积核
    kernel = kernel.expand(channels, 1, kernel_size, kernel_size)
    return kernel


def gaussian_blur_2d(input, kernel_size=7, sigma=2.0):
    b, c, h, w = input.shape
    device = input.device
    dtype = input.dtype
    kernel = get_gaussian_kernel2d(kernel_size, sigma, c, device, dtype)
    input_padded = F.pad(input, (kernel_size // 2,) * 4, mode='reflect')
    return F.conv2d(input_padded, kernel, groups=c)


class SelectionModule(nn.Module):
    def __init__(self, in_channels):
        super(SelectionModule, self).__init__()
        self.selector = nn.Sequential(
            nn.Conv2d(in_channels, 64, kernel_size=3, padding=1),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(64, 1, kernel_size=1),  # 输出一个单通道的图
            nn.Sigmoid()  # 使用 Sigmoid 函数将值压缩到 0 和 1 之间
        )

    def forward(self, x):
        return self.selector(x)


class ProjectedCrossAttention(nn.Module):
    def __init__(self, dim_in, dim_model, num_heads=4, qkv_bias=False, proj_drop=0.):
        """
        dim_in:     输入的通道维度 (C_in), 例如 102
        dim_model:  内部注意力计算的维度 (C_model), 例如 256
        num_heads:  注意力头的数量
        """
        super().__init__()

        # 确保内部维度可以被头数整除
        assert dim_model % num_heads == 0, "dim_model 必须能被 num_heads 整除"

        self.num_heads = num_heads
        self.head_dim = dim_model // num_heads
        self.scale = self.head_dim ** -0.5

        # 1. 输入投影层 (dim_in -> dim_model)
        self.to_q = nn.Linear(dim_in, dim_model, bias=qkv_bias)
        self.to_k = nn.Linear(dim_in, dim_model, bias=qkv_bias)
        self.to_v = nn.Linear(dim_in, dim_model, bias=qkv_bias)

        # 2. 输出投影层 (dim_model -> dim_in)
        self.proj = nn.Linear(dim_model, dim_in)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x_q, x_kv):
        """
        x_q:  Query 特征, shape: [B, K, C_in] (e.g., 102)
        x_kv: Key/Value 源特征, shape: [B, L, C_in] (e.g., 102)
        """

        # 1. 投影到 dim_model
        # x_q: [B, K, 102] -> q: [B, K, 256]
        # x_kv: [B, L, 102] -> k/v: [B, L, 256]
        q = self.to_q(x_q)
        k = self.to_k(x_kv)
        v = self.to_v(x_kv)

        # 2. 拆分为多头
        # [B, K, 256] -> [B, H, K, D] (H=4, D=64)
        q = rearrange(q, 'b k (h d) -> b h k d', h=self.num_heads)
        # [B, L, 256] -> [B, H, L, D]
        k = rearrange(k, 'b l (h d) -> b h l d', h=self.num_heads)
        v = rearrange(v, 'b l (h d) -> b h l d', h=self.num_heads)

        # 3. 计算注意力
        # (B, H, K, D) @ (B, H, D, L) -> (B, H, K, L)
        attn_scores = torch.einsum('b h k d, b h l d -> b h k l', q, k) * self.scale
        attn_probs = attn_scores.softmax(dim=-1)

        # 4. 加权 V
        # (B, H, K, L) @ (B, H, L, D) -> (B, H, K, D)
        context = torch.einsum('b h k l, b h l d -> b h k d', attn_probs, v)

        # 5. 合并多头
        # (B, H, K, D) -> [B, K, C_model] (e.g., [B, K, 256])
        context = rearrange(context, 'b h k d -> b k (h d)')

        # 6. 投影回 dim_in
        # [B, K, 256] -> [B, K, 102]
        output = self.proj(context)
        output = self.proj_drop(output)

        # 输出维度为 [B, K, C_in], 可以安全地进行残差连接
        return output


class ProjectedCrossAttentionTransformer(nn.Module):
    def __init__(self, dim_in, dim_model, num_heads=4, ffn_expansion=4):
        """
        dim_in:        输入的通道维度 (C_in), 例如 102
        dim_model:     内部注意力计算的维度 (C_model), 例如 256
        num_heads:     注意力头的数量
        ffn_expansion: FFN 隐藏层的扩展倍数 (标准Transformer为4)
        """
        super(ProjectedCrossAttentionTransformer, self).__init__()
        self.x_channel = dim_in
        self.num_heads = num_heads

        # LayerNorms (在 C_in 维度上操作, 位于残差路径上)
        self.norm_q = nn.LayerNorm(dim_in)
        self.norm_kv = nn.LayerNorm(dim_in)
        self.norm_ffn = nn.LayerNorm(dim_in)

        # 1. 核心交叉注意力模块 (执行 102 -> 256 -> 102)
        self.attn = ProjectedCrossAttention(dim_in, dim_model, num_heads=num_heads)

        # 2. 前馈网络 (FFN)
        #    同样执行 102 -> 256*4 -> 102 的 "扩展-压缩"
        ffn_hidden_dim = dim_model * ffn_expansion  # e.g., 256 * 4 = 1024
        self.ffn = nn.Sequential(
            nn.Linear(dim_in, ffn_hidden_dim, bias=False),
            nn.GELU(),  # GELU 是 Transformer 中更标准的激活函数
            nn.Linear(ffn_hidden_dim, dim_in, bias=False),
        )

    def forward(self, q_feat, kv_feat):
        """
        q_feat:  Query 特征, shape: [B, K, C_in] (e.g., 102)
        kv_feat: Key/Value 源特征, shape: [B, L, C_in] (e.g., 102)
        """

        # 1. 交叉注意力块 (Pre-Norm + 残差连接)
        # self.attn 返回 [B, K, 102]
        attn_out = self.attn(self.norm_q(q_feat), self.norm_kv(kv_feat))
        q_feat = q_feat + attn_out  # 第一个残差连接 (102 + 102)

        # 2. 前馈网络块 (Pre-Norm + 残差连接)
        # self.ffn 返回 [B, K, 102]
        ffn_out = self.ffn(self.norm_ffn(q_feat))
        q_feat = q_feat + ffn_out  # 第二个残差连接 (102 + 102)

        return q_feat


class LocalAlignedFeatureFusion(nn.Module):
    def __init__(self, dim, window_size=5):
        super().__init__()
        self.dim = dim
        self.window_size = window_size
        self.pad = window_size // 2

        # 降维以节省显存 (256 -> 64)
        self.inter_dim = max(32, dim // 4)

        self.proj_q = nn.Conv2d(dim, self.inter_dim, 1)
        self.proj_k = nn.Conv2d(dim, self.inter_dim, 1)

        # Value 不降维，保证特征质量
        self.proj_v = nn.Identity()

        self.scale = self.inter_dim ** -0.5
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, anchor_feat, source_feat):
        B, C, H, W = anchor_feat.shape

        # 1. 降维计算 Q
        q_small = self.proj_q(anchor_feat)  # (B, 64, H, W)
        # 调整 Q: (B, L, 1, 64)
        q = q_small.view(B, self.inter_dim, -1).permute(0, 2, 1).unsqueeze(2)

        # 2. 降维并 Unfold K
        k_small = self.proj_k(source_feat)  # (B, 64, H, W)
        k_wins = F.unfold(k_small, kernel_size=self.window_size, padding=self.pad)

        # View 拆分: dim 1 是通道(64), dim 2 是窗口(25)
        # (B, 64*25, L) -> (B, 64, 25, L)
        k_wins = k_wins.view(B, self.inter_dim, -1, H * W)

        # [关键修复] Permute: (B, L, 64, 25)
        # 这里的顺序必须是 (0, 3, 1, 2)，让 64 在倒数第二位
        k_wins = k_wins.permute(0, 3, 1, 2)

        # 3. 计算 Attention Score
        # (B, L, 1, 64) @ (B, L, 64, 25) -> (B, L, 1, 25)
        attn_score = torch.matmul(q, k_wins) * self.scale
        attn_probs = self.softmax(attn_score)

        # 4. 处理 Value (保持原通道 C=256)
        v_wins = F.unfold(source_feat, kernel_size=self.window_size, padding=self.pad)
        v_wins = v_wins.view(B, C, -1, H * W)

        # V 的 Permute 保持 (0, 3, 2, 1) -> (B, L, 25, 256)
        # 因为我们要用 (1, 25) 去乘 (25, 256)
        v_wins = v_wins.permute(0, 3, 2, 1)

        # 5. 特征聚合
        # (B, L, 1, 25) @ (B, L, 25, 256) -> (B, L, 1, 256)
        aligned_feat = torch.matmul(attn_probs, v_wins)

        # 还原形状
        aligned_feat = aligned_feat.squeeze(2).permute(0, 2, 1).view(B, C, H, W)
        return aligned_feat


class SSR(nn.Module):
    def __init__(self, patch_size=128, in_ch_msi=3, in_ch_hsi=31):
        super(SSR, self).__init__()

        # --- 1. 保留所有原始的特征提取分支 ---
        self.patch_size = patch_size
        self.Conv = nn.Conv2d(in_ch_hsi, 256, 3, 1, 1)

        self.ResBlock_MSI_1 = ResBlock(in_ch_msi, 64, same_shape=False)
        self.ResBlock_MSI_2 = ResBlock(64, 256, same_shape=False)  # 输入64，输出256，加深深度
        self.ResBlock_HSI_1 = ResBlock(256, 256, same_shape=False)
        self.ResBlock_HSI_reshape = ResBlock(256 * 2, 256, same_shape=False)
        self.Channel_Att = nn.Conv1d(in_channels=patch_size * patch_size, out_channels=256, kernel_size=3, padding=1)
        self.Conv2D_down = nn.Conv2d(256, in_ch_hsi, 1, 1, 0)
        self.common_dim = 256  # 定义一个统一的融合维度

        # 融合门控CNN：学习3个流的权重
        self.ResBlock_HSI = nn.ModuleList([
            ResBlock(in_ch_hsi * 2 + 256, 256, same_shape=False),
            nn.ReLU(),
            ResBlock(256, 256),
            nn.ReLU(),
            ResBlock(256, 128, same_shape=False),
            nn.ReLU(),
            nn.Conv2d(128, in_ch_hsi, 3, 1, 1)
        ])
        self.Conv2D = nn.Conv2d(in_ch_hsi, in_ch_hsi, 3, 1, 1)
        # 引入lrhsi的特征
        self.sam_refine = Transformer(in_ch_hsi)
        self.hsicat = nn.Conv2d(in_ch_hsi * 2, in_ch_hsi, 1)
        self.up = nn.Sequential(
            nn.Conv2d(in_ch_hsi, in_ch_hsi * 16, 1)
        )
        self.ps = nn.PixelShuffle(4)
        # 和空间精细化相关的部分
        self.ssim = SSIM(9)
        self.pancat = nn.Conv2d(in_ch_hsi + in_ch_msi, in_ch_hsi, 1)
        self.ssim_refine = Transformer(in_ch_hsi)
        self.to_r_ssim = nn.Sequential(
            nn.AdaptiveAvgPool2d((1, 1)),
        )
        self.sam = Loss_SAM3()
        self.local_align_module_2 = LocalAlignedFeatureFusion(dim=in_ch_hsi, window_size=5)

    def conv_downsample(self, HSI, scale=4):
        # (此函数保持不变)
        sigma = scale / 2.0
        kernel_size = int(2 * round(3 * sigma) + 1)
        HSI_smooth = gaussian_blur_2d(HSI, kernel_size=kernel_size, sigma=sigma)
        LR_HSI_full = HSI_smooth[:, :, ::scale, ::scale]
        return LR_HSI_full

    def forward(self, x, y, lrhsi, SRF):
        # --- 1. 原始特征提取 (完全保留) ---
        y_feat_1 = self.ResBlock_MSI_1(y)
        y_hat = self.ResBlock_MSI_2(y_feat_1)  # (B, 64, H, W) -> 这里的 y_hat 就是空间纹理库

        x_hsi_feat = self.Conv(x)
        x_hsi_feat = self.ResBlock_HSI_1(x_hsi_feat)
        # 经过两个特征变化
        x_reshape = x_hsi_feat.reshape(x_hsi_feat.shape[0], x_hsi_feat.shape[1], -1).permute(0, 2, 1)
        spe_attation = self.Channel_Att(x_reshape)
        # 矩阵乘法一个1DCNN生成的特征权重
        x_hat = torch.matmul(x_reshape, spe_attation).permute(0, 2, 1).reshape(x_hsi_feat.shape[0], x_hsi_feat.shape[1],
                                                                               self.patch_size,
                                                                               self.patch_size)
        x_hat = self.ResBlock_HSI_reshape(torch.cat([x_hsi_feat, x_hat], dim=1))

        # 融合得到 x_gudie (作为 Conv2D_down 的输入)
        x_gudie = x_hat + y_hat

        dev = x_gudie.device
        dty = x_gudie.dtype
        # 交叉选择性机制
        x_gudie_down = self.Conv2D_down(x_gudie)
        b, c, h, w = x_gudie_down.shape
        if not isinstance(SRF, torch.Tensor):
            srf_weight = torch.from_numpy(SRF).float()
        else:
            srf_weight = SRF.float()
        if srf_weight.dim() == 2:
            srf_weight = srf_weight.view(srf_weight.size(0), srf_weight.size(1), 1, 1)
        srf_weight = srf_weight.to(x.device)
        x_projected = F.conv2d(x_gudie_down, srf_weight)
        x_projected = x_projected.clamp(-1, 1)
        ssim_score = self.ssim(x_projected, y)

        self.M = ssim_score.unsqueeze(1).detach()
        num_keep_node_bili = 0.3 * (1 - self.to_r_ssim(ssim_score).reshape(-1)) / 2.0

        ssim_score = ssim_score.reshape(b, -1)
        B1, N1 = ssim_score.shape
        ssim_idx = torch.argsort(ssim_score, dim=1, descending=False)  # descending=False，代表是升序
        num_keep_node = int(torch.mean(N1 * num_keep_node_bili))  # N_new
        ssim_idx1 = ssim_idx[:, :num_keep_node]  # bad points
        fea_add_pan = self.pancat(torch.cat([x_gudie_down, y], dim=1))
        ssim_v1 = batch_index_select(fea_add_pan.reshape(b, c, -1).permute(0, 2, 1), ssim_idx1)  # B, N_new, C
        ssim_refine = self.ssim_refine(ssim_v1)
        ssim_out = torch.zeros_like(x_gudie_down)
        ssim_out = batch_index_fill2(ssim_out.reshape(b, c, -1).permute(0, 2, 1), ssim_refine, ssim_idx1)
        ssim_out = ssim_out.permute(0, 2, 1).reshape(b, c, h, w) + fea_add_pan  # 经过空间坏点处理之后的特征

        b, c, h, w = ssim_out.shape
        PLR = self.conv_downsample(ssim_out)  # (B, C_hsi, H//4, W//4)
        aligned_lrhsi = self.local_align_module_2(anchor_feat=PLR, source_feat=lrhsi)
        selection_input = torch.cat([PLR, aligned_lrhsi], dim=1)
        fuse_cat = self.hsicat(selection_input)
        attention_map = self.sam(PLR, aligned_lrhsi).reshape(b, h // 4, w // 4, -1)

        # A. 確定 Top-K 索引

        B, C, H, W = fuse_cat.shape
        k = int(torch.mean(H * W * num_keep_node_bili))
        scores = attention_map.view(b, -1)
        _, top_k_indices = torch.topk(scores, k=k, dim=1)

        # B. Gather: 收集 Top-K 特徵
        samfea_sequence = fuse_cat.flatten(2).permute(0, 2, 1)  # b L C
        top_k_indices_expanded = top_k_indices.unsqueeze(-1).expand(-1, -1, C)  # 提取C维数据 B K C
        features_to_refine = torch.gather(samfea_sequence, 1, top_k_indices_expanded)

        refined_features_sparse = self.sam_refine(features_to_refine)
        # D. Scatter: 將精細化特徵放回原位
        output_sequence = torch.zeros_like(samfea_sequence)
        output_sequence.scatter_(1, top_k_indices_expanded, refined_features_sparse)

        refined_features = output_sequence.permute(0, 2, 1).view(B, C, H, W)

        fuse = fuse_cat + refined_features  # 残差连接

        fuse = self.up(fuse).to(device=dev, dtype=dty)
        fuse = self.ps(fuse).to(device=dev, dtype=dty)

        x = torch.cat([fuse, ssim_out, y_hat], dim=1)
        for block in self.ResBlock_HSI:
            x = block(x)

        x = self.Conv2D(x)

        return x


class Attention(nn.Module):

    def __init__(self, dim, num_heads=8, qkv_bias=False, qk_scale=None, attn_drop=0., proj_drop=0., patch_size=0):
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = qk_scale or head_dim ** -0.5
        self.qkv_1 = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.qkv_2 = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.localization_linear = nn.Sequential(
            nn.Linear(in_features=dim * patch_size * patch_size, out_features=32),
            nn.ReLU(),
            nn.Linear(in_features=32, out_features=2 * 3)
        )
        nn.init.zeros_(self.localization_linear[-1].weight)
        nn.init.zeros_(self.localization_linear[-1].bias)

    def forward(self, x, y):
        B, L, C = x.shape
        qkv_1 = self.qkv_1(x)
        qkv_2 = self.qkv_2(y)

        qkv_1 = einops.rearrange(qkv_1, 'B L (K H D) -> K B H L D', K=3, H=self.num_heads)
        q_1, k_1, v_1 = qkv_1[0], qkv_1[1], qkv_1[2]  # B H L D
        qkv_2 = einops.rearrange(qkv_2, 'B L (K H D) -> K B H L D', K=3, H=self.num_heads)
        q_2, k_2, v_2 = qkv_2[0], qkv_2[1], qkv_2[2]  # B H L D

        # k_2 的最后两个维度互换，得到 [B, H, D, L]
        # H指头数
        attn_1 = (q_1 @ k_2.transpose(-2, -1)) * self.scale
        attn_1 = (attn_1).softmax(dim=-1)  # 最后一个维度上进行softmax
        attn_1 = self.attn_drop(attn_1)

        x = (attn_1 @ v_2).transpose(1, 2).reshape(B, L * C)
        # 这里为什么要压缩空间和通道特征？
        theta = self.localization_linear(x)

        return theta


class AGCN(nn.Module):

    def __init__(self, patch_size=32, dim=256, num_heads=4, qkv_bias=False, qk_scale=None, in_ch_msi=3, in_ch_hsi=31):
        super(AGCN, self).__init__()
        patch_size = patch_size // 4
        self.pos_embed = nn.Parameter(torch.zeros(1, patch_size ** 2, dim))  # 定义一个可学习的三维全零张量,每一个patch内部的空间位置关系
        self.Embedding_HSI = nn.Conv2d(in_ch_hsi, 256, 3, 1, 1)
        self.Embedding_MSI = nn.Conv2d(in_ch_msi, 256, 3, 1, 1)
        self.norm = nn.LayerNorm(dim)
        self.attn = Attention(dim, num_heads=num_heads, qkv_bias=qkv_bias, qk_scale=qk_scale, patch_size=patch_size)

    def forward(self, x, x_org, y):
        # y：下采样的HR-MSI，x：上采样的LR-HSI，x_org：未上采样的LR-HSI
        x_org = self.Embedding_HSI(x_org).flatten(2).transpose(1, 2)  # [B, H*W, 256]
        y = self.Embedding_MSI(y).flatten(2).transpose(1, 2)
        x_org = x_org + self.pos_embed
        y = y + self.pos_embed
        delta_raw = self.attn(self.norm(x_org), self.norm(y))  # (B,6)

        # 1) tanh 限幅到 [-1,1]
        delta = torch.tanh(delta_raw)

        # 2) 设定最大扰动幅度
        max_scale_shear = 0.15  # a,b,c,d 的最大变化
        max_trans_norm = 0.15  # tx,ty 的最大变化（归一化坐标）

        scale = torch.tensor(
            [max_scale_shear, max_scale_shear, max_trans_norm,
             max_scale_shear, max_scale_shear, max_trans_norm],
            device=delta.device, dtype=delta.dtype
        )
        delta = delta * scale
        # 3) 加单位仿射
        base = torch.tensor([1, 0, 0, 0, 1, 0], device=delta.device, dtype=delta.dtype).view(1, 6)
        theta = (base + delta).view(-1, 2, 3)
        grid = F.affine_grid(theta, x.size(), align_corners=False)
        x = F.grid_sample(x, grid, mode='bilinear',
                          padding_mode='reflection',
                          align_corners=False)  # 对空间结构进行细微变换

        return x


class CCFnet(nn.Module):

    def __init__(self, patch_size, in_ch_msi=3, in_ch_hsi=31):
        super(CCFnet, self).__init__()
        self.conv = nn.Conv2d(in_ch_hsi, 64, 3, 1, 1)
        self.conv_final = nn.Conv2d(256, in_ch_hsi, 3, 1, 1)
        self.Conv_256 = nn.Conv2d(in_ch_hsi, 256, 3, 1, 1)
        self.upSample = nn.Upsample(scale_factor=4, mode='bicubic')
        self.downSample = nn.Upsample(scale_factor=0.25, mode='bicubic')

        self.CRN = AGCN(patch_size=patch_size, dim=256, num_heads=4, in_ch_msi=in_ch_msi,
                                             in_ch_hsi=in_ch_hsi)

        self.SSR = SSR(patch_size=patch_size, in_ch_msi=in_ch_msi, in_ch_hsi=in_ch_hsi)

        self.UGFR =UGFR(in_ch_msi=in_ch_msi, in_ch_hsi=in_ch_hsi)

    def forward(self, HSI, MSI, lrHS, SRF):
        x = self.UGFR(HSI, MSI, SRF)
        x = self.SSR(x, MSI, lrHS, SRF)

        return x


