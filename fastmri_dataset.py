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
        f_path = self.valid_files[idx]

        with h5py.File(f_path, 'r') as f_orig:
            # 1. 仅读取第 8 个切片（索引为 7）
            # kspace 原始形状: (num_slices, Nc, H, W) -> 抽取后: (Nc, H, W)
            k_slice = torch.tensor(f_orig['kspace'][7], dtype=torch.complex64)
            # reconstruction_rss 原始形状: (num_slices, H, W) -> 抽取后: (H, W)
            target_slice = torch.tensor(f_orig['reconstruction_rss'][7], dtype=torch.float32) 

        Nc, H, W = k_slice.shape
        
        # 2. 施加欠采样掩膜 (fixed_mask 形状为 1, 1, H, W，用 squeeze 去掉多余维度)
        mask_2d = self.fixed_mask.squeeze(0).squeeze(0)  # (H, W)
        f_kspace = k_slice * mask_2d  # (Nc, H, W)

        # 3. 最大值归一化
        target_max = torch.max(torch.abs(target_slice)) + 1e-8
        scale_factor = 1.0 / target_max

        f_norm = f_kspace * scale_factor
        target_norm = target_slice * scale_factor
        k_slice_norm = k_slice * scale_factor

        # 4. 仅针对第 8 个切片计算 c_init 和 G_tensor（无需循环）
        cal_length = 32
        h_start, h_end = H // 2 - cal_length // 2, H // 2 + cal_length // 2
        w_start, w_end = W // 2 - cal_length // 2, W // 2 + cal_length // 2
        
        # 生成二维汉宁窗
        window_1d = torch.hann_window(cal_length, periodic=False, device=k_slice_norm.device)
        window_2d = window_1d.unsqueeze(1) * window_1d.unsqueeze(0)

        # --- 物理感知初始化低清灵敏度图 ---
        acs_block = k_slice_norm[:, h_start:h_end, w_start:w_end]
        kspace_acs = torch.zeros_like(k_slice_norm)
        kspace_acs[:, h_start:h_end, w_start:w_end] = acs_block * window_2d
        
        img_low = torch.fft.ifftshift(kspace_acs, dim=(-2, -1))
        img_low = torch.fft.ifft2(img_low, norm="ortho")
        img_low = torch.fft.fftshift(img_low, dim=(-2, -1))
        
        rss_low = torch.sqrt(torch.sum(torch.abs(img_low) ** 2, dim=0, keepdim=True))
        rss_norm = rss_low / (rss_low.max() + 1e-8)
        soft_mask = torch.sigmoid((rss_norm - 0.15) * 50)
        
        c_init_volume = (img_low / (rss_low + 1e-8)) * soft_mask  # (Nc, H, W)
        
        # --- 计算该切片的 G_tensor ---
        G_tensor_volume = compute_G_for_fastmri_slice(k_slice_norm.numpy(), cal_length=32) # (H, W, Nc, Nc)

        # 5. 使用最优线圈合并计算 u 的初始估计
        Finv = torch.fft.ifftshift(f_norm, dim=(-2, -1))
        Finv = torch.fft.ifft2(Finv, norm="ortho")
        Finv = torch.fft.fftshift(Finv, dim=(-2, -1))
        
        input0_norm = torch.sum(Finv * torch.conj(c_init_volume), dim=0) # (H, W)

        return {
            'u_t': input0_norm,                                     # (H, W)
            'f': f_norm,                                            # (Nc, H, W)
            'sampling_mask': mask_2d.real.to(torch.float32),        # (H, W)
            'reference': target_norm,                               # (H, W)
            'G_tensor': G_tensor_volume,                            # (H, W, Nc, Nc)
            'c_init': c_init_volume,                                # (Nc, H, W)
            'vol_path': f_path
        }
