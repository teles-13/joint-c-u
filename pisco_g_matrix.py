import torch
import numpy as np

def even_pisco(int_val):
    return int_val % 2 == 0

def ChC_FFT_convolutions(X, N1, N2, Nc, tau, pad, kernel_shape):
    device = X.device
    grid_1d = torch.arange(-tau, tau+1, device=device)
    in1, in2 = torch.meshgrid(grid_1d, grid_1d, indexing='xy')
    
    if kernel_shape == 1:
        mask = in1**2 + in2**2 <= tau**2
        mask_flat = mask.t().flatten() # 对应 order='F'
        i = torch.where(mask_flat)[0]
    else:
        i = torch.arange(in1.numel(), device=device)
        
    in1 = in1.t().flatten()[i]
    in2 = in2.t().flatten()[i]
    patchSize = len(in1)

    if pad:
        import math
        N1n = 2 ** math.ceil(math.log2(N1 + 2*tau))
        N2n = 2 ** math.ceil(math.log2(N2 + 2*tau))
    else:
        N1n = N1
        N2n = N2

    row_inds = (N1n // 2) - in1.unsqueeze(1) + in1.unsqueeze(0)
    col_inds = (N2n // 2) - in2.unsqueeze(1) + in2.unsqueeze(0)
    row_inds = torch.clamp(row_inds, 0, N1n-1).long()
    col_inds = torch.clamp(col_inds, 0, N2n-1).long()
    
    # 对应 order='F' 的线性索引
    inds = col_inds * N1n + row_inds
    
    n1_freq = torch.fft.fftshift(torch.fft.fftfreq(N1n, device=device))
    n2_freq = torch.fft.fftshift(torch.fft.fftfreq(N2n, device=device))
    n2, n1 = torch.meshgrid(n2_freq, n1_freq, indexing='xy')
    
    phaseKernel = torch.exp(-1j * 2 * torch.pi * (n1 * ((N1n+1)//2 + tau) + n2 * ((N2n+1)//2 + tau)))
    cphaseKernel = torch.exp(-1j * 2 * torch.pi * (n1 * ((N1n+1)//2) + n2 * ((N2n+1)//2)))

    x = torch.fft.fft2(X, s=(N1n, N2n), dim=(0,1)) * phaseKernel.unsqueeze(2)

    PhP = torch.zeros((patchSize, patchSize, Nc, Nc), dtype=torch.complex64, device=device)
    for q in range(Nc):
        x_rest = x[:, :, q:]
        x_q = x[:, :, q] 
        prod = torch.conj(x_rest) * x_q.unsqueeze(2) * cphaseKernel.unsqueeze(2)
        b = torch.fft.ifft2(prod, dim=(0,1)) 

        # 完美复现 reshape(..., order='F') 
        b_flat = b.permute(1, 0, 2).reshape(-1, Nc - q)
        inds_flat = inds.t().flatten()
        b_selected = b_flat[inds_flat, :]
        b_selected = b_selected.view(patchSize, patchSize, Nc - q).permute(1, 0, 2)
        
        PhP[:, :, q:, q] = b_selected
        if q < Nc - 1:
            PhP[:, :, q, q+1:] = torch.conj(PhP[:, :, q+1:, q].permute(1, 0, 2))

    PhP = PhP.permute(0, 2, 1, 3) 
    PhP = PhP.permute(3, 2, 1, 0).reshape(patchSize * Nc, patchSize * Nc).t()
    return PhP

def nullspace_vectors_C_matrix(kCal, tau, threshold, kernel_shape):
    ChC = ChC_FFT_convolutions(kCal, kCal.shape[0], kCal.shape[1], kCal.shape[2], tau, 1, kernel_shape)
    # 极速 GPU SVD 分解
    U_svd, S, Vh = torch.linalg.svd(ChC, full_matrices=False)
    sing = torch.sqrt(torch.abs(S))
    sing = sing / sing[0]
    
    valid_idx = torch.where(sing >= threshold * sing[0])[0]
    Nvect = valid_idx[-1].item()
    
    U = Vh.conj().t()[:, Nvect+1:] 
    return U

def G_matrices(kCal, N1, N2, tau, U, kernel_shape):
    device = kCal.device
    N1_cal, N2_cal, Nc = kCal.shape
    grid_1d = torch.arange(-tau, tau + 1, device=device)
    in1, in2 = torch.meshgrid(grid_1d, grid_1d, indexing='xy')
    
    flat_in1 = in1.t().flatten()
    flat_in2 = in2.t().flatten()
    
    if kernel_shape == 0:
        ind = torch.arange(len(flat_in1), device=device)
    else:
        mask = in1**2 + in2**2 <= tau**2
        ind = torch.where(mask.t().flatten())[0]
        
    in1 = flat_in1[ind].long()
    in2 = flat_in2[ind].long()
    
    patchSize = len(in1)
    eind = torch.arange(patchSize, 0, -1, device=device) - 1
    total_size = 2 * (2 * tau + 1)
    
    G_flat = torch.zeros((total_size * total_size, Nc, Nc), dtype=torch.complex64, device=device)
    
    W = U @ U.conj().t()
    W = W.t().reshape(Nc, patchSize, Nc, patchSize).permute(3, 2, 1, 0)
    W = W.permute(0, 1, 3, 2)
    
    for s in range(patchSize):
        r0 = 2 * tau + 1 + in1[eind] + in1[s] 
        c0 = 2 * tau + 1 + in2[eind] + in2[s] 
        r0 = torch.clamp(r0, 0, total_size-1)
        c0 = torch.clamp(c0, 0, total_size-1)
        linear_idx = c0 * total_size + r0 
        G_flat[linear_idx, :, :] += W[:, :, :, s]

    G = G_flat.permute(2, 1, 0).reshape(Nc, Nc, total_size, total_size).permute(3, 2, 1, 0)
    
    N1_g, N2_g = N1, N2 
    n1 = torch.fft.fftfreq(N1_g, device=device)
    n2 = torch.fft.fftfreq(N2_g, device=device)
    n2, n1 = torch.meshgrid(n2, n1, indexing='xy')
    
    phaseKernel = torch.exp(-1j * 2 * torch.pi * (n1 * (N1_g - 2*tau - 1) + n2 * (N2_g - 2*tau - 1)))
    G = torch.fft.ifft2(G, s=(N1_g, N2_g), dim=(0,1)) * phaseKernel.unsqueeze(2).unsqueeze(3)
    G = torch.fft.fftshift(G, dim=(0,1))
    
    return G

def compute_G_for_fastmri_slice(kspace_slice, cal_length=32, tau=3, threshold=0.08, kernel_shape=1):
    """
    针对 FastMRI 的预处理包装函数（全 GPU 运行版）。
    输入:
        kspace_slice: PyTorch Tensor, 形状为 (Nc, N1, N2)，必须已经在 GPU 上
    输出:
        G_tensor: PyTorch Tensor, 形状为 (N1, N2, Nc, Nc)，在 GPU 上
    """
    kData = kspace_slice.permute(1, 2, 0)
    N1, N2, Nc = kData.shape
    device = kData.device
    
    center_x = int(np.ceil(N1 / 2)) + even_pisco(N1)
    center_y = int(np.ceil(N2 / 2)) + even_pisco(N2)
    
    cal_index_x = torch.arange(center_x - int(np.floor(cal_length / 2)), 
                               center_x + int(np.floor(cal_length / 2)) - even_pisco(cal_length), device=device)
    cal_index_y = torch.arange(center_y - int(np.floor(cal_length / 2)), 
                               center_y + int(np.floor(cal_length / 2)) - even_pisco(cal_length), device=device)
    
    kCal = kData[cal_index_x.unsqueeze(1), cal_index_y, :]
    
    U = nullspace_vectors_C_matrix(kCal, tau, threshold, kernel_shape)
    G_tensor = G_matrices(kCal, N1, N2, tau, U, kernel_shape)
    
    return G_tensor
