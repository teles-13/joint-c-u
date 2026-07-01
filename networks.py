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
        residual_img = ops.ifft2c(mask * ((mask * k_pred) - (mask * f_kspace))) 
        grad_u_fidelity = torch.sum(torch.conj(c_k) * residual_img, dim=1, keepdim=True) 
        u_mid = u_k - self.alpha_step * grad_u_fidelity
        
        u_real_imag = torch.cat([u_mid.real, u_mid.imag], dim=1) 
        features = self.feature_extractor(u_real_imag)
        u_residual = self.final_conv(features)
        u_prior = u_real_imag + u_residual #残差
        
        norm = torch.sqrt(u_prior[:, 0:1, ...]**2 + u_prior[:, 1:2, ...]**2 + 1e-8)#取通道 0（实部）和通道 1（虚部）的平方和开根号，得到每个像素点的幅度（Magnitude）
        scale = torch.clamp(self.radius / norm, max=1.0)
        u_prior_projected = u_prior * scale
        u_next = torch.complex(u_prior_projected[:, 0:1, ...], u_prior_projected[:, 1:2, ...]) 
        return u_next

class VNSensitivityBlock(nn.Module):
    def __init__(self):
        super().__init__()
        self.beta_step = nn.Parameter(torch.tensor(-1.0, dtype=torch.float32))  
        self.lambda_reg = nn.Parameter(torch.tensor(-3.0, dtype=torch.float32)) 

    def forward(self, c_k, u_next, f_kspace, mask, G_tensor):
        ops = MRPhysicsOperators()
        
        beta = F.softplus(self.beta_step)
        lamb = F.softplus(self.lambda_reg)

        # 1. 计算数据保真项梯度
        c_times_u_next = c_k * u_next
        k_pred_next = ops.fft2c(c_times_u_next)
        residual_img_next = (mask * (ops.ifft2c(mask * (k_pred_next - f_kspace)))) 
        grad_c_fidelity = torch.conj(u_next) * residual_img_next 

        # 2. 计算物理先验项梯度
        grad_c_prior = lamb * torch.einsum('bhwij, bjhw -> bihw', G_tensor, c_k)

        # 3. 联合更新 c
        c_next = c_k - beta * (grad_c_fidelity + grad_c_prior)
        
        
        return c_next
    
class VariationalNetwork(nn.Module):
    # ✨ 新增参数 inner_c_steps=10
    def __init__(self, num_steps=20, num_filters=24, inner_c_steps=10):
        super().__init__()
        self.num_steps = num_steps
        self.inner_c_steps = inner_c_steps
        

        self.u_blocks = nn.ModuleList([VNImageBlock(num_filters) for _ in range(num_steps)])
        
    
        self.c_blocks = nn.ModuleList([VNSensitivityBlock() for _ in range(num_steps * inner_c_steps)])

    def forward(self, f_kspace, mask, G_tensor, u_init, c_init):
        u_k = u_init
        c_k = c_init
        
        c_history = []
        c_block_idx = 0  # 追踪当前调用到了第几个 C_block
        
        # 外层交替优化
        for i in range(self.num_steps):
            u_block = self.u_blocks[i]
            
            # 1. 内部循环：C 先利用当前的 u_k 连续梯度下降 10 次
            for j in range(self.inner_c_steps):
                c_block = self.c_blocks[c_block_idx]
                c_k = c_block(c_k, u_k, f_kspace, mask, G_tensor)
                c_block_idx += 1
            
            # 记录这 10 次下降结束后的 C，用作绘图历史
            c_history.append(c_k)
            
            # 2. u 更新：利用刚才深度优化了 10 次的最新 C，对 u 优化 1 次
            u_k = u_block(u_k, c_k, f_kspace, mask)
            
        return u_k, c_k, c_history
