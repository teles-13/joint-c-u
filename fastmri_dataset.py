import os
import glob
import h5py
import torch
import numpy as np
from torch.utils.data import Dataset

class JointFastMRIDataset(Dataset):
    def __init__(self, data_dir, target_shape=(16, 16, 640, 320), target_slice=7):
        super().__init__()
        self.data_dir = data_dir
        self.target_shape = target_shape
        self.target_slice = target_slice
        
        self.files = glob.glob(os.path.join(data_dir, "**/*.h5"), recursive=True)
        self.slice_indices = []
        
        print(f"🔄 正在扫描盲重建数据集...")
        for f_path in self.files:
            try:
                with h5py.File(f_path, 'r') as f:
                    if f['kspace'].shape == self.target_shape:
                        num_slices = self.target_shape[0]
                        for s_idx in range(num_slices):
                            if self.target_slice is None or s_idx == self.target_slice:
                                self.slice_indices.append((f_path, s_idx))
            except Exception:
                pass
        
        print(f"✅ 扫描完成！共找到 {len(self.slice_indices)} 个目标切片。")

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

        with h5py.File(f_path, 'r') as f_orig:
            k_slice = torch.tensor(f_orig['kspace'][s_idx], dtype=torch.complex64)
            target_slice = torch.tensor(f_orig['reconstruction_rss'][s_idx], dtype=torch.float32) 

        mask_2d = self.fixed_mask.squeeze(0).squeeze(0)
        
        # ⚠️ 纯粹的数据读写，没有卡脖子的计算
        return {
            'k_slice': k_slice,
            'target_slice': target_slice,
            'sampling_mask': mask_2d.real.to(torch.float32),
            'slice_idx': s_idx,
            'vol_path': f_path
        }
