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
                            if s_idx == self.target_slice:  
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
        
        # 💡 BPMRI 初始化：在不知敏感度的情况下的初始图像
        Finv = torch.fft.ifftshift(f_kspace, dim=(-2, -1))
        Finv = torch.fft.ifft2(Finv, norm="ortho")
        Finv = torch.fft.fftshift(Finv, dim=(-2, -1))
        input0 = torch.sum(Finv, dim=-3)

        # 应用 Paper Norm 归一化
        f_norm = f_kspace * paper_norm
        input0_norm = input0 * paper_norm
        target_norm = target * paper_norm
        k_slice_norm = k_slice * paper_norm

        # 在 Dataset 中直接计算 G_tensor
        k_slice_np = k_slice_norm.numpy()
        G_tensor = compute_G_for_fastmri_slice(k_slice_np, cal_length=32)

        Nc, H, W = k_slice_norm.shape
        cal_length = 32  # 保持与自校准区域一致的尺度
        
        # 1. 创建一个全零的 k 空间矩阵，仅保留中心 ACS 区域
        kspace_acs = torch.zeros_like(k_slice_norm)
        
        # 提取中心低频区域 (32x32)
        h_start, h_end = H // 2 - cal_length // 2, H // 2 + cal_length // 2
        w_start, w_end = W // 2 - cal_length // 2, W // 2 + cal_length // 2
        kspace_acs[:, h_start:h_end, w_start:w_end] = k_slice_norm[:, h_start:h_end, w_start:w_end]
        
        # 2. 逆傅里叶变换到图像域，得到低分辨率平滑线圈图像 (使用标准的中心化 ifft2c 逻辑)
        img_low = torch.fft.ifftshift(kspace_acs, dim=(-2, -1))
        img_low = torch.fft.ifft2(img_low, norm="ortho")
        img_low = torch.fft.fftshift(img_low, dim=(-2, -1))
        
        # 3. 计算多线圈低分辨图像的平方和根 (RSS)
        rss_low = torch.sqrt(torch.sum(torch.abs(img_low) ** 2, dim=0, keepdim=True) + 1e-8)
        
        # 4. 线圈图像除以 RSS 得到初始的敏感度图分布
        c_init_lowres = img_low / rss_low
        # =========================================================================

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
       
