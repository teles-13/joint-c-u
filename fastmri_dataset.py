# /home/liujunda/joint-c-u/fastmri_dataset.py

import os
import glob
import h5py
import torch
import numpy as np
import torch.nn.functional as F
from torch.utils.data import Dataset
from pisco_g_matrix import compute_G_for_fastmri_slice

class JointFastMRIDataset(Dataset):
    """
    盲联合重建 (Blind PI) 专用版 Dataset。
    每一个样本代表一个完整的 Volume（包含该数据的所有切片）。
    """
    def __init__(self, data_dir, target_shape=(16, 16, 640, 320), target_slice=None):
        super().__init__()
        self.data_dir = data_dir
        self.target_shape = target_shape
        # 注意：此处 target_slice 参数已被忽略，因为我们要一次性训练所有切片
        
        self.files = glob.glob(os.path.join(data_dir, "**/*.h5"), recursive=True)
        self.valid_files = []
        
        print(f"🔄 正在扫描盲重建数据集（全切片联合版）...")
        
        for f_path in self.files:
            try:
                with h5py.File(f_path, 'r') as f:
                    kspace_shape = f['kspace'].shape
                    # 匹配符合目标形状的 Volume
                    if kspace_shape == self.target_shape:
                        self.valid_files.append(f_path)
            except Exception as e:
                pass
        
        print(f"✅ 扫描完成！共找到 {len(self.valid_files)} 个目标Volume数据。")

        # 生成固定的二维欠采样掩膜 (Mask) -> 形状: (1, 1, H, W) 方便后续广播
        H, W = target_shape[2], target_shape[3]
        mask_tmp = torch.zeros((1, 1, H, W), dtype=torch.complex64)
        num_low = int(W * 0.08)
        pad = (W - num_low + 1) // 2
        mask_tmp[:, :, :, pad : pad + num_low] = 1

        torch.manual_seed(42) 
        high_mask = torch.rand(1, 1, 1, W) < (0.25 - (num_low / W))
        self.fixed_mask = torch.logical_or(mask_tmp.bool(), high_mask).to(torch.complex64)

    def __len__(self):
        return len(self.valid_files)

    def __getitem__(self, idx):
        f_path = self.valid_files[idx]

        with h5py.File(f_path, 'r') as f_orig:
            # 1. 一次性读取该数据的所有切片! 形状: (num_slices, Nc, H, W)
            k_volume = torch.tensor(f_orig['kspace'][:], dtype=torch.complex64)
            # 形状: (num_slices, H, W)
            target_volume = torch.tensor(f_orig['reconstruction_rss'][:], dtype=torch.float32) 

        num_slices, Nc, H, W = k_volume.shape
        
        # 2. 施加欠采样掩膜
        mask = self.fixed_mask.clone()  # (1, 1, H, W)
        f_kspace = k_volume * mask      # 广播特性自动应用到所有切片 (num_slices, Nc, H, W)

        # 3. 全局最大值归一化 (以整个 Volume 的最大像素值为基准，保持切片间的相对物理比例)
        target_max = torch.max(torch.abs(target_volume)) + 1e-8
        scale_factor = 1.0 / target_max

        f_norm = f_kspace * scale_factor
        target_norm = target_volume * scale_factor
        k_volume_norm = k_volume * scale_factor

        # 4. 遍历当前 Volume 的所有切片，计算各自的 c_init 和 G_tensor
        c_init_list = []
        G_tensor_list = []
        
        cal_length = 32
        h_start, h_end = H // 2 - cal_length // 2, H // 2 + cal_length // 2
        w_start, w_end = W // 2 - cal_length // 2, W // 2 + cal_length // 2
        
        # 生成二维汉宁窗
        window_1d = torch.hann_window(cal_length, periodic=False, device=k_volume_norm.device)
        window_2d = window_1d.unsqueeze(1) * window_1d.unsqueeze(0)

        for s_idx in range(num_slices):
            k_slice_norm = k_volume_norm[s_idx]
            
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
            c_init_slice = (img_low / (rss_low + 1e-8)) * soft_mask
            c_init_list.append(c_init_slice)
            
            # --- 计算该切片的 G_tensor ---
            G_tensor_slice = compute_G_for_fastmri_slice(k_slice_norm.numpy(), cal_length=32)
            G_tensor_list.append(G_tensor_slice)

        # 堆叠回四维/五维张量
        c_init_volume = torch.stack(c_init_list, dim=0)       # (num_slices, Nc, H, W)
        G_tensor_volume = torch.stack(G_tensor_list, dim=0)   # (num_slices, H, W, Nc, Nc)

        # 5. 使用最优线圈合并计算整个 Volume 中所有 u 的初始估计
        Finv = torch.fft.ifftshift(f_norm, dim=(-2, -1))
        Finv = torch.fft.ifft2(Finv, norm="ortho")
        Finv = torch.fft.fftshift(Finv, dim=(-2, -1))
        
        # input0_norm 形状: (num_slices, H, W)
        input0_norm = torch.sum(Finv * torch.conj(c_init_volume), dim=1)

        return {
            'u_t': input0_norm,                                     
            'f': f_norm,                                            
            'sampling_mask': mask.real.to(torch.float32).squeeze(0).squeeze(0), # (H, W)
            'reference': target_norm,                               
            'G_tensor': G_tensor_volume, 
            'c_init': c_init_volume,  
            'vol_path': f_path
        }
