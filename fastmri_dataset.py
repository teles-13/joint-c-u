# /home/liujunda/joint-c-u/fastmri_dataset.py

import os
import glob
import h5py
import torch
import numpy as np
from torch.utils.data import Dataset
from pisco_g_matrix import compute_G_for_fastmri_slice

class JointFastMRIDataset(Dataset):
    """
    盲联合重建 (Blind PI) 专用版 Dataset。
    彻底抛弃预计算的 smaps，仅从原始 kspace 数据提取所有信息。
    """
    def __init__(self, data_dir, target_shape=(16, 16, 640, 320), target_slice=7):
        super().__init__()
        self.data_dir = data_dir
        self.target_shape = target_shape
        self.target_slice = target_slice
        
        self.files = glob.glob(os.path.join(data_dir, "**/*.h5"), recursive=True)
        self.slice_indices = []
        self.volume_norms = {} 
        
        print(f"🔄 正在扫描盲重建数据集...")
        
        for f_path in self.files:
            try:
                with h5py.File(f_path, 'r') as f:
                    kspace_shape = f['kspace'].shape
                    if kspace_shape == self.target_shape:
                        num_slices = kspace_shape[0] 
                        k_volume_preview = f['kspace'][:]
                        f_volume_norm = np.linalg.norm(k_volume_preview)
                        paper_norm = (np.sqrt(num_slices * 10000.0)) / (f_volume_norm + 1e-8)
                        self.volume_norms[f_path] = paper_norm
                        
                        for s_idx in range(num_slices):
                            if self.target_slice is None or s_idx == self.target_slice:
                                self.slice_indices.append((f_path, s_idx))
            except Exception as e:
                pass
        
        print(f"✅ 扫描完成！共找到 {len(self.slice_indices)} 个目标切片。")

        # 生成固定的欠采样掩膜 (Mask)
        H, W = target_shape[2], target_shape[3]
        mask_tmp = torch.zeros((1, H, W), dtype=torch.complex64)
        num_low = int(W * 0.08)
        pad = (W - num_low + 1) // 2
        mask_tmp[:, :, pad : pad + num_low] = 1

        torch.manual_seed(42) 
        high_mask = torch.rand(1, 1, W) < (0.25 - (num_low / W))
        self.fixed_mask = torch.logical_or(mask_tmp.bool(), high_mask).to(torch.complex64)

    def __len__(self):
        return len(self.slice_indices)

    def __getitem__(self, idx):
        f_path, s_idx = self.slice_indices[idx]
        paper_norm = self.volume_norms[f_path]

        # 💡 核心修改：只读取 f_orig，不再读取 f_smap！
        with h5py.File(f_path, 'r') as f_orig:
            k_slice = torch.tensor(f_orig['kspace'][s_idx], dtype=torch.complex64)
            target = torch.tensor(f_orig['reconstruction_rss'][s_idx], dtype=torch.float32) 

        mask = self.fixed_mask.clone()
        f_kspace = k_slice * mask

        # =========================================================================
        # ✨ 路线 A 核心修改：最大值归一化 ✨
        # 目的：强行把真实图像的最大像素值压到 1.0，以完美匹配文献中的 r1 = 1.0 理论约束
        # =========================================================================
        target_max = torch.max(torch.abs(target)) + 1e-8
        scale_factor = 1.0 / target_max

        # 使用最大值缩放因子替代 paper_norm
        f_norm = f_kspace * scale_factor
        target_norm = target * scale_factor
        k_slice_norm = k_slice * scale_factor

        # =========================================================================
        # 提前计算初始敏感度图 c_init_lowres (保留我们之前第二步的修复)
        # =========================================================================
        # --- 替换为以下物理感知初始化逻辑 ---
        Nc, H, W = k_slice_norm.shape
        cal_length = 32  # 保持你原来的 ACS 大小
        
        # 1. 提取中心 ACS 块
        h_start, h_end = H // 2 - cal_length // 2, H // 2 + cal_length // 2
        w_start, w_end = W // 2 - cal_length // 2, W // 2 + cal_length // 2
        acs_block = k_slice_norm[:, h_start:h_end, w_start:w_end]
        
        # 2. 生成二维汉宁窗 (消除硬截断导致的波浪伪影)
        window_1d = torch.hann_window(cal_length, periodic=False, device=k_slice_norm.device)
        window_2d = window_1d.unsqueeze(1) * window_1d.unsqueeze(0)
        
        # 3. 将窗函数应用到 ACS 块上并补回全零矩阵
        kspace_acs = torch.zeros_like(k_slice_norm)
        kspace_acs[:, h_start:h_end, w_start:w_end] = acs_block * window_2d
        
        # 4. 执行 IFFT 得到平滑的低频感应场
        img_low = torch.fft.ifftshift(kspace_acs, dim=(-2, -1))
        img_low = torch.fft.ifft2(img_low, norm="ortho")
        img_low = torch.fft.fftshift(img_low, dim=(-2, -1))
        
        # 5. 计算 RSS 并应用 Sigmoid 软掩膜 (消除空气区域噪声放大)
        rss_low = torch.sqrt(torch.sum(torch.abs(img_low) ** 2, dim=0, keepdim=True))
        
        # 构建软掩膜: 仅在有信号区域 RSS 趋近 1，背景处趋近 0
        rss_norm = rss_low / (rss_low.max() + 1e-8)
        soft_mask = torch.sigmoid((rss_norm - 0.15) * 50)
        
        # 最终初始化结果：归一化并施加保护掩膜
        c_init_lowres = (img_low / (rss_low + 1e-8)) * soft_mask
        # ----------------------------------

        # =========================================================================
        # 使用最优线圈合并计算 u 的初始估计 (消除相位抵消)
        # =========================================================================
        Finv = torch.fft.ifftshift(f_kspace, dim=(-2, -1))
        Finv = torch.fft.ifft2(Finv, norm="ortho")
        Finv = torch.fft.fftshift(Finv, dim=(-2, -1))
        
        input0 = torch.sum(Finv * torch.conj(c_init_lowres), dim=-3)
        # 同样使用 scale_factor 进行缩放
        input0_norm = input0 * scale_factor

        # 计算 G_tensor
        k_slice_np = k_slice_norm.numpy()
        G_tensor = compute_G_for_fastmri_slice(k_slice_np, cal_length=32)
        return {
            'u_t': input0_norm,                                     
            'f': f_norm,                                            
            'sampling_mask': mask.real.to(torch.float32).squeeze(0),
            'reference': target_norm,                               
            'kspace_raw': k_slice_norm,                             
            'G_tensor': G_tensor, 
            'c_init': c_init_lowres,  # <--- 将算好的初始敏感度图传给 DataLoader
            'slice_idx': s_idx
        }
       
