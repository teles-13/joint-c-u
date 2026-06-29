import matplotlib
matplotlib.use('Agg')
import os
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["VECLIB_MAXIMUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"

import torch
torch.set_num_threads(2)
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
from skimage.metrics import structural_similarity as ssim_func
from torchmetrics.functional import structural_similarity_index_measure as ssim_func_pt

from fastmri_dataset import JointFastMRIDataset 
from pisco_g_matrix import compute_G_for_fastmri_slice
from networks import VariationalNetwork  

os.environ["CUDA_VISIBLE_DEVICES"] = "3"
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
    np.random.seed(np.random.get_state()[1][0] + worker_id)
    torch.set_num_threads(1)

def main():
    print("====== 正在初始化盲联合重建 (Blind Joint Recon) ======")
    data_dir = "/home/liujunda/data_fastmri_brain_train/multicoil_train"  
    image_save_dir = 'recon_images'
    os.makedirs(image_save_dir, exist_ok=True)
    
    # ⚠️ 确保聚焦第七个切片
    dataset = JointFastMRIDataset(data_dir=data_dir, target_shape=(16, 16, 640, 320), target_slice=7)
    dataloader = DataLoader(dataset, batch_size=1, shuffle=True, 
                            num_workers=4, pin_memory=True, 
                            worker_init_fn=worker_init_fn)
    
    num_steps = 20
    model = VariationalNetwork(num_steps=num_steps, num_filters=24).to(device)
    optimizer = optim.Adam(model.parameters(), lr=5e-5)
    
    num_epochs = 1000
    loss_hist, mse_hist, ssim_hist, psnr_hist = [], [], [], []
    
    weight_hist = {
        'alpha': {i: [] for i in range(num_steps)},
        'beta': {i: [] for i in range(num_steps)},
        'lambda': {i: [] for i in range(num_steps)}
    }
    
    print("====== 开始训练 ======")
    for epoch in range(num_epochs):
        model.train()
        epoch_loss, epoch_mse, epoch_ssim, epoch_psnr = 0.0, 0.0, 0.0, 0.0
        num_batches = len(dataloader)
        
        for batch_idx, data_dict in enumerate(dataloader):
            # 1. 纯数据直接丢入 GPU
            k_slice = data_dict['k_slice'].squeeze(0).to(device)       
            target_slice = data_dict['target_slice'].squeeze(0).to(device) 
            mask_2d = data_dict['sampling_mask'].squeeze(0).to(device) 
            current_slice = data_dict['slice_idx'][0].item()
            
            Nc, H, W = k_slice.shape
            
            # 2. 欠采样与幅度值归一化 (GPU)
            f_kspace = k_slice * mask_2d
            target_max = torch.max(torch.abs(target_slice)) + 1e-8
            scale_factor = 1.0 / target_max
            
            f_norm = f_kspace * scale_factor
            k_slice_norm = k_slice * scale_factor
            target_norm = target_slice * scale_factor

            # 3. G Tensor (利用主进程单线程 NumPy 飞速解算 32x32，绝不死锁)
            k_slice_norm_np = k_slice_norm.cpu().numpy()
            G_tensor_slice = compute_G_for_fastmri_slice(k_slice_norm_np, cal_length=32)
            G_tensor_batch = G_tensor_slice.unsqueeze(0).to(device)

            # 4. 物理感知 Smap 初始化 (全 GPU 并行傅里叶)
            cal_length = 32
            h_start, h_end = H // 2 - cal_length // 2, H // 2 + cal_length // 2
            w_start, w_end = W // 2 - cal_length // 2, W // 2 + cal_length // 2
            
            window_1d = torch.hann_window(cal_length, periodic=False, device=device)
            window_2d = window_1d.unsqueeze(1) * window_1d.unsqueeze(0)

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
            c_init = c_init_slice.unsqueeze(0) 

            # 5. u 初始化 (GPU)
            Finv = torch.fft.ifftshift(f_norm, dim=(-2, -1))
            Finv = torch.fft.ifft2(Finv, norm="ortho")
            Finv = torch.fft.fftshift(Finv, dim=(-2, -1))
            u_t = torch.sum(Finv * torch.conj(c_init_slice), dim=0).unsqueeze(0).unsqueeze(0)

            # 6. 配置网络张量尺寸 (B, C, H, W)
            f_kspace_input = f_norm.unsqueeze(0)             
            mask_input = mask_2d.unsqueeze(0).unsqueeze(0)   
            target_input = target_norm.unsqueeze(0).unsqueeze(0) 

            optimizer.zero_grad()
            u_pred_unprocessed, c_final, c_history = model(f_kspace_input, mask_input, G_tensor_batch, u_t, c_init)
            
            u_pred_cropped = center_crop(u_pred_unprocessed, (320, 320))
            target_cropped_mag = center_crop(target_input, (320, 320))
            
            # ✨ 彻底修正：使用纯幅度 MSE Loss 解耦，完全切断波纹来源
            u_pred_mag_loss = torch.abs(u_pred_cropped)
            target_mag_loss = torch.abs(target_cropped_mag)
            
            loss = F.mse_loss(u_pred_mag_loss, target_mag_loss)
            
            loss.backward()
            optimizer.step()
            
            # 评价指标
            with torch.no_grad():
                u_pred_mag = torch.abs(center_crop(u_pred_unprocessed, (320, 320)))
                target_mag = torch.abs(center_crop(target_input, (320, 320)))

                mse_gpu = torch.mean((target_mag - u_pred_mag) ** 2)
                data_range_gpu = target_mag.max() - target_mag.min()
                
                if mse_gpu == 0:
                    val_psnr = 100.0
                else:
                    val_psnr = 20 * torch.log10(data_range_gpu / torch.sqrt(mse_gpu))
                
                val_ssim = ssim_func_pt(u_pred_mag, target_mag, data_range=data_range_gpu)
                
                epoch_loss += loss.item()
                epoch_mse += mse_gpu.item()  
                epoch_psnr += val_psnr.item()
                epoch_ssim += val_ssim.item() 

                if batch_idx == 0 or (batch_idx + 1) % 50 == 0:
                    recon_mag_np = u_pred_mag.cpu().detach().numpy()
                    ref_mag_np = target_mag.cpu().numpy()  
                    
                    under_mag_img = center_crop(torch.abs(u_t), (320, 320)).cpu().detach().numpy()[0, 0]
                    recon_img = recon_mag_np[0, 0]
                    ref_img = ref_mag_np[0, 0]
                    
                    error_map = np.abs(ref_img - recon_img)
                    
                    c_final_cropped = center_crop(c_final, (320, 320))
                    c_img = torch.abs(c_final_cropped).cpu().detach().numpy()[0, 0]
                    
                    fig, axes = plt.subplots(1, 5, figsize=(20, 4))
                    axes[0].imshow(under_mag_img, cmap='gray')
                    axes[0].set_title("Input (Zero-filled)")
                    axes[0].axis('off')
                    axes[1].imshow(recon_img, cmap='gray')
                    axes[1].set_title("Recon Image (u)")
                    axes[1].axis('off')
                    axes[2].imshow(ref_img, cmap='gray')
                    axes[2].set_title("Reference (GT)")
                    axes[2].axis('off')
                    axes[3].imshow(error_map, cmap='gray')
                    axes[3].set_title("Error Map")
                    axes[3].axis('off')
                    axes[4].imshow(c_img, cmap='gray')
                    axes[4].set_title("Estimated Smap (Coil 0)")
                    axes[4].axis('off')
                    
                    plt.suptitle(f"Epoch {epoch+1} | Batch {batch_idx+1} - Result (Slice {current_slice})")
                    plt.tight_layout()
                    plt.savefig(os.path.join(image_save_dir, f'recon_epoch_{epoch+1:03d}_batch_{batch_idx+1:04d}.png'), bbox_inches='tight')
                    plt.close()

                    fig_c, axes_c = plt.subplots(4, 5, figsize=(20, 16))
                    axes_c = axes_c.flatten() # 展平一维，方便用 for 循环遍历
                    
                    for step_idx in range(num_steps):
                        # 提取第 step_idx 层的敏感度图
                        c_step_tensor = c_history[step_idx]
                        c_step_cropped = center_crop(c_step_tensor, (320, 320))
                        
                        # 取出当前层 Coil 0 的幅值图，转到 CPU 上变 numpy
                        c_step_img = torch.abs(c_step_cropped).cpu().detach().numpy()[0, 0]
                        
                        # 画在对应的格子里
                        axes_c[step_idx].imshow(c_step_img, cmap='gray')
                        axes_c[step_idx].set_title(f"Step {step_idx + 1}")
                        axes_c[step_idx].axis('off')
                        
                    plt.suptitle(f"Epoch {epoch+1} | Batch {batch_idx+1} - Smap Evolution (Coil 0)", fontsize=20)
                    plt.tight_layout()
                    
                    # 额外保存为一张新图片
                    plt.savefig(os.path.join(image_save_dir, f'smap_evolution_epoch_{epoch+1:03d}_batch_{batch_idx+1:04d}.png'), bbox_inches='tight')
                    plt.close(fig_c)

        with torch.no_grad():
            for step_idx in range(num_steps):
                alpha_val = model.u_blocks[step_idx].alpha_step.item()
                beta_val = F.softplus(model.c_blocks[step_idx].beta_step).item()
                lambda_val = F.softplus(model.c_blocks[step_idx].lambda_reg).item()
                
                weight_hist['alpha'][step_idx].append(alpha_val)
                weight_hist['beta'][step_idx].append(beta_val)
                weight_hist['lambda'][step_idx].append(lambda_val)

        fig_w, (ax_alpha, ax_beta, ax_lambda) = plt.subplots(1, 3, figsize=(20, 5))
        colors = plt.cm.plasma(np.linspace(0, 1, num_steps))
        
        for step_idx in range(num_steps):
            ax_alpha.plot(weight_hist['alpha'][step_idx], color=colors[step_idx], label=f'Step {step_idx+1}')
            ax_beta.plot(weight_hist['beta'][step_idx], color=colors[step_idx], label=f'Step {step_idx+1}')
            ax_lambda.plot(weight_hist['lambda'][step_idx], color=colors[step_idx], label=f'Step {step_idx+1}')
            
        ax_alpha.set_title("Alpha (Image Block)")
        ax_alpha.legend(fontsize='small', ncol=2)
        ax_beta.set_title("Beta (Smap Block)")
        ax_beta.legend(fontsize='small', ncol=2)
        ax_lambda.set_title("Lambda (G-Matrix Prior)")
        ax_lambda.legend(fontsize='small', ncol=2)
        plt.tight_layout()
        plt.savefig('learned_parameters_curves.png')
        plt.close(fig_w)

        avg_epoch_loss = epoch_loss / num_batches
        avg_epoch_ssim = epoch_ssim / num_batches
        avg_epoch_psnr = epoch_psnr / num_batches
        
        loss_hist.append(avg_epoch_loss)
        mse_hist.append(epoch_mse / num_batches)
        ssim_hist.append(avg_epoch_ssim)
        psnr_hist.append(avg_epoch_psnr)

        print(f"Epoch [{epoch+1}/{num_epochs}] | Loss: {avg_epoch_loss:.6f} | SSIM: {avg_epoch_ssim:.4f} | PSNR: {avg_epoch_psnr:.2f} dB")

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

    os.makedirs('saved_models', exist_ok=True)
    model_save_path = 'saved_models/joint_blind_vn.pth' 
    torch.save(model.state_dict(), model_save_path)
    print(f"🎉 模型已保存至: {model_save_path}")

if __name__ == "__main__":
    main()
