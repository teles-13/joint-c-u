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
    print("====== 正在初始化盲联合重建 (Blind Joint Recon - 全 Volume 版) ======")
    data_dir = "/home/liujunda/data_fastmri_brain_train/multicoil_train"  
    image_save_dir = 'recon_images'
    os.makedirs(image_save_dir, exist_ok=True)
    
    # 实例化全 Volume 数据集
    dataset = JointFastMRIDataset(data_dir=data_dir, target_shape=(16, 16, 640, 320))
    # 注意：batch_size=1 意味着每次读取 1 个完整的 Volume（内部自带 16 个切片作为网络的 Batch 维度）
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
            # 因为 DataLoader batch_size=1，所以多了一层多余的外面 Batch 维度，直接用 squeeze(0) 去掉它
            # 从而把真正的切片维度释放出来，变为网络的 Batch 维度 B = num_slices
            u_t = data_dict['u_t'].squeeze(0).to(device)                 # (B, H, W)
            f_kspace = data_dict['f'].squeeze(0).to(device)              # (B, Nc, H, W)
            mask = data_dict['sampling_mask'].squeeze(0).to(device)      # (H, W)
            target = data_dict['reference'].squeeze(0).to(device)        # (B, H, W)
            G_tensor_batch = data_dict['G_tensor'].squeeze(0).to(device) # (B, H, W, Nc, Nc)
            c_init = data_dict['c_init'].squeeze(0).to(device)           # (B, Nc, H, W)
            
            B, Nc, H, W = f_kspace.shape # 此时 B 自动等于 16 (切片总数)
            
            # 为图像扩充通道维度，变成符合网络要求的 (B, 1, H, W)
            u_t = u_t.unsqueeze(1) 
            target = target.unsqueeze(1)
            # 掩膜广播扩充为 (B, 1, H, W)
            mask = mask.unsqueeze(0).unsqueeze(0).expand(B, 1, H, W)
            
            optimizer.zero_grad()
            
            # 核心网络前向传播：16个切片的数据作为一个矩阵同时输入，一次迭代更新所有切片的灵敏度 c_final 
            u_pred_unprocessed, c_final = model(f_kspace, mask, G_tensor_batch, u_t, c_init)
            
            # 裁剪到目标视野
            u_pred_cropped = center_crop(u_pred_unprocessed, (320, 320))
            target_cropped = center_crop(target, (320, 320))
            u_t_cropped = center_crop(u_t, (320, 320))

            # 物理相位损失计算
            target_phase = torch.angle(u_t_cropped)     
            target_complex = torch.abs(target_cropped) * torch.exp(1j * target_phase)

            # 在复数域计算整个 Volume (16个切片) 的总平均 Loss
            loss = F.mse_loss(torch.view_as_real(u_pred_cropped), torch.view_as_real(target_complex))
            
            loss.backward()
            optimizer.step()
            
            # 指标计算与记录
            with torch.no_grad():
                u_pred_mag = torch.abs(u_pred_cropped)
                target_mag = torch.abs(target_cropped)

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

                # 每 50 个 batch 画图展示（我们默认抽取该 Volume 的中间切片进行可视化）
                if batch_idx == 0 or (batch_idx + 1) % 50 == 0:
                    mid_slice = B // 2  # 取中间那个切片
                    
                    under_mag_img = center_crop(torch.abs(u_t), (320, 320)).cpu().detach().numpy()[mid_slice, 0]
                    recon_img = u_pred_mag.cpu().detach().numpy()[mid_slice, 0]
                    ref_img = target_mag.cpu().numpy()[mid_slice, 0]
                    
                    error_map = np.abs(ref_img - recon_img)
                    
                    c_final_cropped = center_crop(c_final, (320, 320))
                    c_img = torch.abs(c_final_cropped).cpu().detach().numpy()[mid_slice, 0] # 提取中间切片的第0个线圈
                    
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
                    
                    plt.suptitle(f"Epoch {epoch+1} | Volume Batch {batch_idx+1} - Mid-Slice {mid_slice} Result")
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
