import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from torch.nn.utils import spectral_norm

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
        self.conv = spectral_norm(nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1))
        self.in_norm = nn.InstanceNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        return self.relu(self.in_norm(self.conv(x)))

class VNImageBlock(nn.Module):
    def __init__(self, num_filters=64): 
        super().__init__()
        self.alpha_step = nn.Parameter(torch.tensor(1.0, dtype=torch.float32))
        blocks = []
        blocks.append(PaperConvBlock(2, num_filters))
        for _ in range(8):
            blocks.append(PaperConvBlock(num_filters, num_filters))
        self.feature_extractor = nn.Sequential(*blocks)
        self.final_conv = spectral_norm(nn.Conv2d(num_filters, 2, kernel_size=3, padding=1)) 
        self.radius = 1.0 

    def forward(self, u_k, c_k, f_kspace, mask):
        ops = MRPhysicsOperators()
        c_times_u = c_k * u_k 
        k_pred = ops.fft2c(c_times_u)
        residual_img = ops.ifft2c(mask * (k_pred - f_kspace)) 
        grad_u_fidelity = torch.sum(torch.conj(c_k) * residual_img, dim=1, keepdim=True) 
        u_mid = u_k - self.alpha_step * grad_u_fidelity
        
        u_real_imag = torch.cat([u_mid.real, u_mid.imag], dim=1) 
        features = self.feature_extractor(u_real_imag)
        u_residual = self.final_conv(features)
        u_prior = u_real_imag + u_residual 
        
        norm = torch.sqrt(u_prior[:, 0:1, ...]**2 + u_prior[:, 1:2, ...]**2 + 1e-8)
        scale = torch.clamp(self.radius / norm, max=1.0)
        u_prior_projected = u_prior * scale
        u_next = torch.complex(u_prior_projected[:, 0:1, ...], u_prior_projected[:, 1:2, ...]) 
        return u_next

class VNSensitivityBlock(nn.Module):
    def __init__(self):
        super().__init__()
        self.beta_step = nn.Parameter(torch.tensor(-1.0, dtype=torch.float32))  
        self.lambda_reg = nn.Parameter(torch.tensor(-3.0, dtype=torch.float32)) 

    # ⚠️ 彻底修复：参数签名去掉了 lamb 和 beta
    def forward(self, c_k, u_next, f_kspace, mask, G_tensor):
        ops = MRPhysicsOperators()
        
        beta = F.softplus(self.beta_step)
        lamb = F.softplus(self.lambda_reg)

        # 1. 计算数据保真项梯度
        c_times_u_next = c_k * u_next
        k_pred_next = ops.fft2c(c_times_u_next)
        residual_img_next = ops.ifft2c(mask * (k_pred_next - f_kspace)) 
        grad_c_fidelity = torch.conj(u_next) * residual_img_next 

        # 2. 计算物理先验项梯度
        grad_c_prior = lamb * torch.einsum('bhwij, bjhw -> bihw', G_tensor, c_k)

        # 3. 联合更新 c
        c_next = c_k - beta * (grad_c_fidelity + grad_c_prior)
        
        # ==========================================================
        # ✨ 紧急修复：只保留绝对纯净的 RSS=1 物理投影
        # 删掉所有跟 periodic_window 和 pad_h/pad_w 相关的代码！
        # ==========================================================
        c_rss = torch.sqrt(torch.sum(torch.abs(c_next)**2, dim=1, keepdim=True) + 1e-8)
        c_next = c_next / c_rss
        
        return c_next

class VariationalNetwork(nn.Module):
    def __init__(self, num_steps=10, num_filters=24):
        super().__init__()
        self.u_blocks = nn.ModuleList([VNImageBlock(num_filters) for _ in range(num_steps)])
        self.c_blocks = nn.ModuleList([VNSensitivityBlock() for _ in range(num_steps)])

    def forward(self, f_kspace, mask, G_tensor, u_init, c_init):
        u_k = u_init
        c_k = c_init
        
        # ✨ 1. 新增：创建一个空列表记录迭代历史
        c_history = []
        
        for u_block, c_block in zip(self.u_blocks, self.c_blocks):
            u_k = u_block(u_k, c_k, f_kspace, mask)
            c_k = c_block(c_k, u_k, f_kspace, mask, G_tensor)
            
            # ✨ 2. 新增：将当前这一层的 C 存入列表
            c_history.append(c_k)
            
        # ✨ 3. 修改：返回时带上这个历史记录
        return u_k, c_k, c_history
