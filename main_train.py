import matplotlib
matplotlib.use('Agg')

import os
# 彻底限制隐式多线程，把 CPU 使用率封印在低水平
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["VECLIB_MAXIMUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"

import torch
# 主进程最多只分配 2 个核
torch.set_num_threads(2)

import torch.nn.functional as F
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
from skimage.metrics import structural_similarity as ssim_func

# 导入你写好的各个模块
from fastmri_dataset import JointFastMRIDataset 
from pisco_g_matrix import compute_G_for_fastmri_slice
from networks import VariationalNetwork  

# 锁定 GPU
os.environ["CUDA_VISIBLE_DEVICES"] = "2"
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

def center_crop(data, shape=(320, 320)):
    h, w = data.shape[-2], data.shape[-1]
    th, tw = shape
    h_start = (h - th) // 2 
    w_start = (w - tw) // 2
    return data[..., h_start:h_start+th, w_start:w_start+tw]

def worker_init_fn(worker_id):
    import numpy as np
    import torch
    # 保证不同进程的数据增强/采样随机种子不同
    np.random.seed(np.random.get_state()[1][0] + worker_id)
    # 强制每个子进程只能使用 1 个 CPU 核
    torch.set_num_threads(1)

def main():
    print("====== 正在初始化盲联合重建 (Blind Joint Recon) ======")
    data_dir = "/home/liujunda/data_fastmri_brain_train/multicoil_train"  
    image_save_dir = 'recon_images'
    os.makedirs(image_save_dir, exist_ok=True)
    
    # 初始化数据集
    dataset = JointFastMRIDataset(data_dir=data_dir, target_shape=(16, 16, 640, 320), target_slice=7)
    dataloader = DataLoader(dataset, batch_size=1, shuffle=True, 
                            num_workers=4, pin_memory=True, 
                            worker_init_fn=worker_init_fn)
    
    model = VariationalNetwork(num_steps=10, num_filters=24).to(device)
    optimizer = optim.Adam(model.parameters(), lr=5e-5)
    
    num_epochs = 1000
    loss_hist, mse_hist, ssim_hist, psnr_hist = [], [], [], []
    
    print("====== 开始训练 ======")
    for epoch in range(num_epochs):
        model.train()
        epoch_loss, epoch_mse, epoch_ssim, epoch_psnr = 0.0, 0.0, 0.0, 0.0
        num_batches = len(dataloader)
        
        for batch_idx, data_dict in enumerate(dataloader):
            u_t = data_dict['u_t'].to(device)                 
            f_kspace = data_dict['f'].to(device)              
            mask = data_dict['sampling_mask'].to(device)      
            target = data_dict['reference'].to(device)        
            G_tensor_batch = data_dict['G_tensor'].to(device) 
            
            # === 1. 修改：直接从 data_dict 中提取在 Dataset 里算好的 c_init ===
            c_init = data_dict['c_init'].to(device)
            
            current_slice = data_dict['slice_idx'][0].item() 
            
            B, Nc, H, W = f_kspace.shape
            
            # 调整维度
            if u_t.dim() == 3: u_t = u_t.unsqueeze(1) 
            if target.dim() == 3: target = target.unsqueeze(1)
            if mask.dim() == 3: mask = mask.unsqueeze(1)

            # === 2. 修改：注销或删掉下面这行原有的全 1 初始化代码 ===
            # c_init = torch.ones(B, Nc, H, W, dtype=torch.complex64).to(device) 
            
            optimizer.zero_grad()
            # 此时传入模型的 c_init 已经包含了物理合理的空间平滑度与线圈间相位差
            u_pred_unprocessed, c_final = model(f_kspace, mask, G_tensor_batch, u_t, c_init)
            
            u_pred = center_crop(u_pred_unprocessed, (320, 320))
            target_cropped = center_crop(target, (320, 320))
            
            # <--- 修正 Loss：在幅值域计算 MSE
            u_pred_mag = torch.abs(u_pred)
            target_mag = torch.abs(target_cropped)
            loss = F.mse_loss(u_pred_mag, target_mag)
            
            loss.backward()
            optimizer.step()
            
            # ==========================================
            # 指标计算与记录
            # ==========================================
            u_pred_mag = torch.abs(u_pred)
            target_mag = torch.abs(target_cropped)
            
            with torch.no_grad():
                # ✨ 直接在 GPU 上计算 MSE 和 PSNR（速度提升上百倍）
                mse_gpu = torch.mean((target_mag - u_pred_mag) ** 2)
                data_range_gpu = target_mag.max() - target_mag.min()
                
                if mse_gpu == 0:
                    val_psnr = 100.0
                else:
                    val_psnr = 20 * torch.log10(data_range_gpu / torch.sqrt(mse_gpu))
                
                scaled_mse = loss.item() * 100
                epoch_loss += loss.item()
                epoch_mse += scaled_mse
                # 取标量值
                epoch_psnr += val_psnr.item()

                # 每 50 个 batch 画一次图，只有画图时才拉回 CPU 算一次 SSIM 供参考
                if batch_idx == 0 or (batch_idx + 1) % 50 == 0:
                    recon_mag_np = u_pred_mag.cpu().detach().numpy()
                    ref_mag_np = target_mag.cpu().numpy()  
                    
                    under_mag_img = center_crop(torch.abs(u_t), (320, 320)).cpu().detach().numpy()[0, 0]
                    recon_img = recon_mag_np[0, 0]
                    ref_img = ref_mag_np[0, 0]
                    
                    # 在这里算一下 SSIM 即可，不要给每个 batch 都算
                    val_ssim = ssim_func(recon_img, ref_img, data_range=ref_img.max() - ref_img.min(), gaussian_weights=True, use_sample_covariance=False)
                    epoch_ssim += val_ssim # 用采样的 SSIM 代表整体即可
                    
                    error_scale = 1.0
                    error_map = np.abs(ref_img - recon_img) * error_scale
                    
                    # 提取估计出的敏感度图 c
                    c_final_cropped = center_crop(c_final, (320, 320))
                    c_img = torch.abs(c_final_cropped).cpu().detach().numpy()[0, 0]
                    
                    # 改用 subplots 布局，一行放 5 张图
                    fig, axes = plt.subplots(1, 5, figsize=(20, 4))
                    
                    # 1. 欠采样输入图
                    axes[0].imshow(under_mag_img, cmap='gray')
                    axes[0].set_title("Input (Zero-filled)")
                    axes[0].axis('off')
                    
                    # 2. 模型重建图 u
                    axes[1].imshow(recon_img, cmap='gray')
                    axes[1].set_title("Recon Image (u)")
                    axes[1].axis('off')
                    
                    # 3. 真实参考图
                    axes[2].imshow(ref_img, cmap='gray')
                    axes[2].set_title("Reference (GT)")
                    axes[2].axis('off')
                    
                    # 4. 误差图
                    axes[3].imshow(error_map, cmap='gray')
                    axes[3].set_title("Error Map")
                    axes[3].axis('off')
                    
                    # 5. ✨ 新增：估计出的线圈敏感度图 c（自动适应其特有的数值范围）
                    axes[4].imshow(c_img, cmap='gray')
                    axes[4].set_title("Estimated Smap (Coil 0)")
                    axes[4].axis('off')
                    
                    plt.suptitle(f"Epoch {epoch+1} | Batch {batch_idx+1} - Result (Slice {current_slice})")
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
