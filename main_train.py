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
from torchmetrics.functional import structural_similarity_index_measure as ssim_func_pt

# 导入你写好的各个模块
from fastmri_dataset import JointFastMRIDataset 
from pisco_g_matrix import compute_G_for_fastmri_slice
from networks import VariationalNetwork  

# 锁定 GPU
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
    # 保证不同进程的数据增强/采样随机种子不同
    np.random.seed(np.random.get_state()[1][0] + worker_id)
    # 强制每个子进程只能使用 1 个 CPU 核
    torch.set_num_threads(1)

def main():
    print("====== 正在初始化盲联合重建 (Blind Joint Recon) ======")
    data_dir = "/home/liujunda/data_fastmri_brain_train/multicoil_train"  
    image_save_dir = 'recon_images'
    os.makedirs(image_save_dir, exist_ok=True)
    
    
    dataset = JointFastMRIDataset(data_dir=data_dir, target_shape=(16, 16, 640, 320), target_slice=None)
    dataloader = DataLoader(dataset, batch_size=1, shuffle=True, 
                            num_workers=4, pin_memory=True, 
                            worker_init_fn=worker_init_fn)
    
    num_steps = 20
    model = VariationalNetwork(num_steps=num_steps, num_filters=24).to(device)
    optimizer = optim.Adam(model.parameters(), lr=5e-5)
    
    num_epochs = 1000
    loss_hist, mse_hist, ssim_hist, psnr_hist = [], [], [], []
    
    # ==========================================
    # ✨ 新增：初始化用于记录网络权重的字典
    # ==========================================
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
            u_t = data_dict['u_t'].to(device)                 
            f_kspace = data_dict['f'].to(device)              
            mask = data_dict['sampling_mask'].to(device)      
            target = data_dict['reference'].to(device)        
            G_tensor_batch = data_dict['G_tensor'].to(device) 
            c_init = data_dict['c_init'].to(device)
            current_slice = data_dict['slice_idx'][0].item() 
            
            B, Nc, H, W = f_kspace.shape
            
            if u_t.dim() == 3: u_t = u_t.unsqueeze(1) 
            if target.dim() == 3: target = target.unsqueeze(1)
            if mask.dim() == 3: mask = mask.unsqueeze(1)
            
            optimizer.zero_grad()
            u_pred_unprocessed, c_final = model(f_kspace, mask, G_tensor_batch, u_t, c_init)
            
            u_pred = center_crop(u_pred_unprocessed, (320, 320))
            target_cropped = center_crop(target, (320, 320))
            
            u_pred_mag = torch.abs(u_pred)
            target_mag = torch.abs(target_cropped)
            
            # ==========================================
            # 修正 1：在复数域计算 Loss，防止相位崩塌
            # ==========================================
            u_pred_cropped = center_crop(u_pred_unprocessed, (320, 320))
            target_cropped_mag = center_crop(target, (320, 320))
            u_t_cropped = center_crop(u_t, (320, 320))

            # 提取物理上最合理的初始相位
            target_phase = torch.angle(u_t_cropped)     
            # 构造带有物理相位的复数 Target
            target_complex = target_cropped_mag * torch.exp(1j * target_phase)

            # 计算复数 MSE (等价于分别计算实部和虚部的 MSE)
            loss = F.mse_loss(torch.view_as_real(u_pred_cropped), torch.view_as_real(target_complex))
            
            loss.backward()
            optimizer.step()
            
            # 指标计算与记录
            with torch.no_grad():
                # 为了 torchmetrics，确保维度是 (B, C, H, W)
                if target_mag.dim() == 3: target_mag = target_mag.unsqueeze(1)
                if u_pred_mag.dim() == 3: u_pred_mag = u_pred_mag.unsqueeze(1)

                # ✨ 直接在 GPU 上计算 MSE 和 PSNR
                mse_gpu = torch.mean((target_mag - u_pred_mag) ** 2)
                data_range_gpu = target_mag.max() - target_mag.min()
                
                if mse_gpu == 0:
                    val_psnr = 100.0
                else:
                    val_psnr = 20 * torch.log10(data_range_gpu / torch.sqrt(mse_gpu))
                
                # ✨ 新增：直接在 GPU 上计算 SSIM（每个 batch 都算）
                val_ssim = ssim_func_pt(u_pred_mag, target_mag, data_range=data_range_gpu)
                
                # 记录指标 (⚠️ 注意这里去掉了 loss.item() * 100 的缩放，还原真实的 MSE)
                epoch_loss += loss.item()
                epoch_mse += mse_gpu.item()  # 修复：记录真实的物理 MSE
                epoch_psnr += val_psnr.item()
                epoch_ssim += val_ssim.item() # 修复：每个 batch 都累加真实的 SSIM

                # 每 50 个 batch 画一次图
                if batch_idx == 0 or (batch_idx + 1) % 50 == 0:
                    recon_mag_np = u_pred_mag.cpu().detach().numpy()
                    ref_mag_np = target_mag.cpu().numpy()  
                    
                    under_mag_img = center_crop(torch.abs(u_t), (320, 320)).cpu().detach().numpy()[0, 0]
                    recon_img = recon_mag_np[0, 0]
                    ref_img = ref_mag_np[0, 0]
                    
                    # ⚠️ 注意：这里删除了原来计算 val_ssim 和累加 epoch_ssim 的代码，因为上面已经算过了
                    
                    error_scale = 1.0
                    error_map = np.abs(ref_img - recon_img) * error_scale
                    
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

        # ==========================================
        # ✨ 新增：在每个 Epoch 结束时记录模型权重并画图
        # ==========================================
        with torch.no_grad():
            for step_idx in range(num_steps):
                # 提取 alpha (VNImageBlock)
                alpha_val = model.u_blocks[step_idx].alpha_step.item()
                # 提取 beta 和 lambda (VNSensitivityBlock)，并套用真实前向传播使用的 softplus
                beta_val = F.softplus(model.c_blocks[step_idx].beta_step).item()
                lambda_val = F.softplus(model.c_blocks[step_idx].lambda_reg).item()
                
                weight_hist['alpha'][step_idx].append(alpha_val)
                weight_hist['beta'][step_idx].append(beta_val)
                weight_hist['lambda'][step_idx].append(lambda_val)

        # 绘制权重演变曲线
        fig_w, (ax_alpha, ax_beta, ax_lambda) = plt.subplots(1, 3, figsize=(20, 5))
        
        # 使用 cmap 渐变色区分不同的级联层
        colors = plt.cm.plasma(np.linspace(0, 1, num_steps))
        
        for step_idx in range(num_steps):
            ax_alpha.plot(weight_hist['alpha'][step_idx], color=colors[step_idx], label=f'Step {step_idx+1}')
            ax_beta.plot(weight_hist['beta'][step_idx], color=colors[step_idx], label=f'Step {step_idx+1}')
            ax_lambda.plot(weight_hist['lambda'][step_idx], color=colors[step_idx], label=f'Step {step_idx+1}')
            
        ax_alpha.set_title("Alpha (Image Block Step Size)")
        ax_alpha.set_xlabel("Epoch")
        ax_alpha.legend(fontsize='small', ncol=2)
        
        ax_beta.set_title("Beta (Smap Block Step Size) [Effective]")
        ax_beta.set_xlabel("Epoch")
        ax_beta.legend(fontsize='small', ncol=2)
        
        ax_lambda.set_title("Lambda (G-Matrix Prior Weight) [Effective]")
        ax_lambda.set_xlabel("Epoch")
        ax_lambda.legend(fontsize='small', ncol=2)
        
        plt.tight_layout()
        plt.savefig('learned_parameters_curves.png')
        plt.close(fig_w)
        # ==========================================

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
