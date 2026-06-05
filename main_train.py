import matplotlib
matplotlib.use('Agg')

import os
import torch
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
from skimage.metrics import structural_similarity as ssim_func

# 导入你写好的各个模块
from fastmri_dataset import JointFastMRIDataset 
from pisco_g_matrix import compute_G_for_fastmri_slice
from networks import VariationalNetwork  # 导入我们刚刚理清的网络

# 锁定 GPU
os.environ["CUDA_VISIBLE_DEVICES"] = "3"
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

def center_crop(data, shape=(320, 320)):
    h, w = data.shape[-2], data.shape[-1]
    th, tw = shape
    h_start = (h - th) // 2 
    w_start = (w - tw) // 2
    return data[..., h_start:h_start+th, w_start:w_start+tw]

def main():
    print("====== 正在初始化盲联合重建 (Blind Joint Recon) ======")
    data_dir = "/home/liujunda/data_fastmri_train"  
    image_save_dir = 'recon_images'
    os.makedirs(image_save_dir, exist_ok=True)
    
    # 初始化数据集
    dataset = JointFastMRIDataset(data_dir=data_dir, target_shape=(16, 16, 640, 320), target_slice=7)
    dataloader = DataLoader(dataset, batch_size=4, shuffle=True, num_workers=0, pin_memory=True)
    
    model = VariationalNetwork(num_steps=10, num_filters=24).to(device)
    optimizer = optim.Adam(model.parameters(), lr=5e-6)
    
    num_epochs = 1000
    loss_hist, mse_hist, ssim_hist, psnr_hist = [], [], [], []
    
    print("====== 开始训练 ======")
    for epoch in range(num_epochs):
        model.train()
        epoch_loss, epoch_mse, epoch_ssim, epoch_psnr = 0.0, 0.0, 0.0, 0.0
        num_batches = len(dataloader)
        
        for batch_idx, data_dict in enumerate(dataloader):
            # 获取数据
            u_t = data_dict['u_t'].to(device)                 
            f_kspace = data_dict['f'].to(device)              
            mask = data_dict['sampling_mask'].to(device)      
            target = data_dict['reference'].to(device)        
            kspace_raw = data_dict['kspace_raw']              
            current_slice = data_dict['slice_idx'][0].item() 
            
            B, Nc, H, W = f_kspace.shape
            
            # =========================================================
            # 你的物理底座：动态计算本批次的 G 矩阵
            # =========================================================
            G_tensor_list = []
            for i in range(B):
                k_slice_np = kspace_raw[i].numpy() 
                g_tensor_single = compute_G_for_fastmri_slice(k_slice_np, cal_length=32)
                G_tensor_list.append(g_tensor_single)
            G_tensor_batch = torch.stack(G_tensor_list).to(device)
            
            # 调整维度
            if u_t.dim() == 3: u_t = u_t.unsqueeze(1) 
            if target.dim() == 3: target = target.unsqueeze(1)
            if mask.dim() == 3: mask = mask.unsqueeze(1)

            # =========================================================
            # 你的核心实验点：盲重建初始化 (c 设为全 1)
            # =========================================================
            c_init = torch.ones(B, Nc, H, W, dtype=torch.complex64).to(device) 
            
            optimizer.zero_grad()
            
            # 传入网络进行交替联合优化
            u_pred_unprocessed, c_final = model(f_kspace, mask, G_tensor_batch, u_t, c_init)
            
            # 裁剪并计算 Loss
            u_pred = center_crop(u_pred_unprocessed, (320, 320))
            target_cropped = center_crop(target, (320, 320))
            
            target_complex = target_cropped.to(torch.complex64)
            loss = F.mse_loss(torch.view_as_real(u_pred), torch.view_as_real(target_complex))
            
            loss.backward()
            optimizer.step()
            
            # ==========================================
            # 指标计算与记录
            # ==========================================
            u_pred_mag = torch.abs(u_pred)
            target_mag = torch.abs(target_cropped)
            
            with torch.no_grad():
                recon_mag_np = u_pred_mag.cpu().detach().numpy()
                ref_mag_np = target_mag.cpu().numpy()  
                
                batch_ssim_sum = 0.0
                batch_psnr_sum = 0.0
                actual_B = recon_mag_np.shape[0]
                
                for b in range(actual_B):
                    recon_b = recon_mag_np[b, 0, :, :]
                    ref_b = ref_mag_np[b, 0, :, :]
                    data_range = ref_b.max() - ref_b.min()
                    
                    val_ssim = ssim_func(recon_b, ref_b, data_range=data_range, gaussian_weights=True, use_sample_covariance=False)
                    batch_ssim_sum += val_ssim
                    
                    mse_val = np.mean((ref_b - recon_b) ** 2)
                    val_psnr = 100.0 if mse_val == 0 else 20 * np.log10(data_range / np.sqrt(mse_val))
                    batch_psnr_sum += val_psnr
                
                avg_batch_ssim = batch_ssim_sum / actual_B
                avg_batch_psnr = batch_psnr_sum / actual_B
                scaled_mse = loss.item() * 100
                
                epoch_loss += loss.item()
                epoch_mse += scaled_mse
                epoch_ssim += avg_batch_ssim
                epoch_psnr += avg_batch_psnr

                # 每 50 个 batch 画一次图
                if batch_idx == 0 or (batch_idx + 1) % 50 == 0:
                    under_mag_img = center_crop(torch.abs(u_t), (320, 320)).cpu().detach().numpy()[0, 0]
                    recon_img = recon_mag_np[0, 0]
                    ref_img = ref_mag_np[0, 0]
                    
                    error_scale = 1.0
                    error_map = np.abs(ref_img - recon_img) * error_scale
                    
                    combined_img = np.concatenate((under_mag_img, recon_img, ref_img, error_map), axis=1)
                    combined_img_uint8 = np.clip(combined_img * 255.0, 0, 255).astype(np.uint8)
                    
                    plt.figure(figsize=(15, 5))
                    plt.imshow(combined_img_uint8, cmap='gray', vmin=0, vmax=255)
                    plt.title(f"Epoch {epoch+1} | Batch {batch_idx+1} - Recon Result (Slice {current_slice})")
                    plt.axis('off')
                    plt.tight_layout()
                    plt.savefig(os.path.join(image_save_dir, f'recon_epoch_{epoch+1:03d}_batch_{batch_idx+1:04d}.png'), bbox_inches='tight')
                    plt.close()

        # Epoch 总结与保存
        avg_epoch_loss = epoch_loss / num_batches
        avg_epoch_ssim = epoch_ssim / num_batches
        avg_epoch_psnr = epoch_psnr / num_batches
        
        loss_hist.append(avg_epoch_loss)
        mse_hist.append(epoch_mse / num_batches)
        ssim_hist.append(avg_epoch_ssim)
        psnr_hist.append(avg_epoch_psnr)

        print(f"Epoch [{epoch+1}/{num_epochs}] | Loss: {avg_epoch_loss:.6f} | SSIM: {avg_epoch_ssim:.4f} | PSNR: {avg_epoch_psnr:.2f} dB")

        # 绘制训练曲线
        fig, (ax1, ax2, ax3, ax4) = plt.subplots(1, 4, figsize=(24, 5))
        ax1.plot(loss_hist, color='green')
        ax1.set_title("Training Loss")
        ax1.set_yscale('log')
        ax2.plot(mse_hist, color='red')
        ax2.set_title("Scaled MSE")
        ax2.set_yscale('log')
        ax3.plot(ssim_hist, color='blue')
        ax3.set_title("SSIM (0-1 Range)")
        ax4.plot(psnr_hist, color='purple')
        ax4.set_title("PSNR (dB)")
        plt.tight_layout()
        plt.savefig('current_training_curves.png')
        plt.close(fig) 

    # 保存模型
    os.makedirs('saved_models', exist_ok=True)
    model_save_path = 'saved_models/joint_blind_vn.pth' 
    torch.save(model.state_dict(), model_save_path)
    print(f"🎉 模型已保存至: {model_save_path}")

if __name__ == "__main__":
    main()
