import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from torch.nn.utils import spectral_norm

# ==========================================
# 1. 物理算子库
# ==========================================
class MRPhysicsOperators:
    @staticmethod
    def fft2c(x):
        x = torch.fft.ifftshift(x, dim=(-2, -1))
        x = torch.fft.fft2(x, norm="ortho")
        x = torch.fft.fftshift(x, dim=(-2, -1))
        return x

    @staticmethod
    def ifft2c(x):
        x = torch.fft.ifftshift(x, dim=(-2, -1))
        x = torch.fft.ifft2(x, norm="ortho")
        x = torch.fft.fftshift(x, dim=(-2, -1))
        return x


class PaperConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        # 论文要求使用谱归一化来保证 Lipschitz 常数 L < 1 [cite: 298, 304]
        self.conv = spectral_norm(nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1))
        self.in_norm = nn.InstanceNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        return self.relu(self.in_norm(self.conv(x)))


class VNImageBlock(nn.Module):
    def __init__(self, num_filters=64): # 论文指定 64 通道 
        super().__init__()
        self.alpha_step = nn.Parameter(torch.tensor(1.0, dtype=torch.float32))
        
        # 论文结构: 9 个 Conv Block + 1 个最终 Conv 层 
        blocks = []
        blocks.append(PaperConvBlock(2, num_filters))
        for _ in range(8):
            blocks.append(PaperConvBlock(num_filters, num_filters))
        
        self.feature_extractor = nn.Sequential(*blocks)
        self.final_conv = spectral_norm(nn.Conv2d(num_filters, 2, kernel_size=3, padding=1)) [cite: 290]
        
        self.radius = 1.0 # 论文指定的投影半径 r=1 [cite: 479]

    def forward(self, u_k, c_k, f_kspace, mask):
        ops = MRPhysicsOperators()
        
        # --- 1. 物理梯度下降 (与你原代码一致，对应公式 2.10) ---
        c_times_u = c_k * u_k 
        k_pred = ops.fft2c(c_times_u)
        residual_img = ops.ifft2c(mask * (k_pred - f_kspace)) 
        grad_u_fidelity = torch.sum(torch.conj(c_k) * residual_img, dim=1, keepdim=True) 
        u_mid = u_k - self.alpha_step * grad_u_fidelity
        
        # --- 2. U-Subnet 近端算子 ---
        u_real_imag = torch.cat([u_mid.real, u_mid.imag], dim=1) 
        
        # 论文强调的 F_u = Id + \tilde{F}_u 残差连接 
        features = self.feature_extractor(u_real_imag)
        u_residual = self.final_conv(features)
        u_prior = u_real_imag + u_residual 
        
        # --- 3. 边界投影约束 \Pi_{B_r} (Eq 2.13) [cite: 296, 297] ---
        norm = torch.sqrt(u_prior[:, 0:1, ...]**2 + u_prior[:, 1:2, ...]**2 + 1e-8)
        scale = torch.clamp(self.radius / norm, max=1.0)
        u_prior_projected = u_prior * scale
        
        u_next = torch.complex(u_prior_projected[:, 0:1, ...], u_prior_projected[:, 1:2, ...]) 
        
        return u_next

# ==========================================
# ✨ [你的核心创新] 敏感度图 c 的更新块 (显式物理先验) ✨
# ==========================================
class VNSensitivityBlock(nn.Module):
    def __init__(self):
        super().__init__()
        # 初始化为负数，经过 softplus 后起步较小，防止梯度爆炸
        self.beta_step = nn.Parameter(torch.tensor(-2.0, dtype=torch.float32))  
        self.lambda_reg = nn.Parameter(torch.tensor(-1.0, dtype=torch.float32)) 

    def forward(self, c_k, u_next, f_kspace, mask, G_tensor):
        ops = MRPhysicsOperators()
        beta = F.softplus(self.beta_step)
        lamb = F.softplus(self.lambda_reg)

        # 1. 计算保真项梯度
        c_times_u_next = c_k * u_next
        k_pred_next = ops.fft2c(c_times_u_next)
        residual_img_next = ops.ifft2c(mask * (k_pred_next - f_kspace)) 
        grad_c_fidelity = torch.conj(u_next) * residual_img_next 

        # 2. 计算你的物理约束梯度 (\lambda H^H H c) 
        # 神级 einsum：直接完成 (B, H, W, Nc, Nc) 与 (B, Nc, H, W) 的通道矩阵乘法
        grad_c_prior = lamb * torch.einsum('bhwij, bjhw -> bihw', G_tensor, c_k)

        # 3. 联合更新 c
        c_next = c_k - beta * (grad_c_fidelity + grad_c_prior)
        return c_next

# ==========================================
# 完整级联网络组合
# ==========================================
class VariationalNetwork(nn.Module):
    def __init__(self, num_steps=10, num_filters=24):
        super().__init__()
        self.u_blocks = nn.ModuleList([VNImageBlock(num_filters) for _ in range(num_steps)])
        self.c_blocks = nn.ModuleList([VNSensitivityBlock() for _ in range(num_steps)])

    def forward(self, f_kspace, mask, G_tensor, u_init, c_init):
        u_k = u_init
        c_k = c_init
        
        # 逐层交替更新
        for u_block, c_block in zip(self.u_blocks, self.c_blocks):
            u_k = u_block(u_k, c_k, f_kspace, mask)
            c_k = c_block(c_k, u_k, f_kspace, mask, G_tensor)
            
        return u_k, c_k
