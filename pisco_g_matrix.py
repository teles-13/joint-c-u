import numpy as np
import torch

def even_pisco(int_val):
    return int_val % 2 == 0

def ChC_FFT_convolutions(X, N1, N2, Nc, tau, pad, kernel_shape):
    in1, in2 = np.meshgrid(np.arange(-tau, tau+1), np.arange(-tau, tau+1), indexing='xy')
    if kernel_shape == 1:
        mask = in1**2 + in2**2 <= tau**2
        i = np.where(mask.flatten(order='F'))[0]
    else:
        i = np.arange(in1.size)
    in1 = in1.flatten(order='F')[i]
    in2 = in2.flatten(order='F')[i]
    patchSize = len(in1)

    if pad:
        N1n = 2 ** int(np.ceil(np.log2(N1 + 2*tau)))
        N2n = 2 ** int(np.ceil(np.log2(N2 + 2*tau)))
    else:
        N1n = N1
        N2n = N2

    row_inds = (N1n // 2) - in1[:, np.newaxis] + in1[np.newaxis, :]
    col_inds = (N2n // 2) - in2[:, np.newaxis] + in2[np.newaxis, :]
    row_inds = np.clip(row_inds, 0, N1n-1).astype(int)
    col_inds = np.clip(col_inds, 0, N2n-1).astype(int)
    inds = np.ravel_multi_index((row_inds, col_inds), (N1n, N2n), order='F') 

    n1_freq = np.fft.fftshift(np.fft.fftfreq(N1n))
    n2_freq = np.fft.fftshift(np.fft.fftfreq(N2n))
    n2, n1 = np.meshgrid(n2_freq, n1_freq, indexing='xy')

    phaseKernel = np.exp(-1j * 2 * np.pi * (n1 * ((N1n+1)//2 + tau) + n2 * ((N2n+1)//2 + tau)))
    cphaseKernel = np.exp(-1j * 2 * np.pi * (n1 * ((N1n+1)//2) + n2 * ((N2n+1)//2)))

    x = np.fft.fft2(X, s=(N1n, N2n), axes=(0,1)) * phaseKernel[:, :, np.newaxis]

    PhP = np.zeros((patchSize, patchSize, Nc, Nc), dtype=complex)
    for q in range(Nc):
        x_rest = x[:, :, q:]
        x_q = x[:, :, q] 
        prod = np.conj(x_rest) * x_q[:, :, np.newaxis] * cphaseKernel[:, :, np.newaxis]
        b = np.fft.ifft2(prod, axes=(0,1)) 

        b = b.reshape(-1, Nc - q, order='F') 
        b_selected = b[inds.flatten(order='F'), :] 
        b_selected = b_selected.reshape(patchSize, patchSize, Nc - q, order='F')

        PhP[:, :, q:, q] = b_selected
        if q < Nc - 1:
            PhP[:, :, q, q+1:] = np.conj(PhP[:, :, q+1:, q].transpose(1, 0, 2))

    PhP = PhP.transpose(0, 2, 1, 3) 
    PhP = PhP.reshape(patchSize * Nc, patchSize * Nc, order='F')
    return PhP

def nullspace_vectors_C_matrix(kCal, tau, threshold, kernel_shape):
    # 直接使用高效的 FFT 卷积计算 C^H * C
    ChC = ChC_FFT_convolutions(kCal, kCal.shape[0], kCal.shape[1], kCal.shape[2], tau, 1, kernel_shape)
    _, s, vh = np.linalg.svd(ChC, full_matrices=False)
    sing = np.sqrt(np.abs(s))
    sing = sing / sing[0]
    Nvect = np.where(sing >= threshold * sing[0])[0][-1]
    U = vh.conj().T[:, Nvect+1:]  # 提取零空间对应的右奇异向量
    return U

def G_matrices(kCal, N1, N2, tau, U, kernel_shape):
    N1_cal, N2_cal, Nc = kCal.shape
    in1, in2 = np.meshgrid(np.arange(-tau, tau + 1), np.arange(-tau, tau + 1))
    
    flat_in1 = in1.flatten(order='F')
    flat_in2 = in2.flatten(order='F')
    if kernel_shape == 0:
        ind = np.arange(len(flat_in1))
    else:
        mask = in1**2 + in2**2 <= tau**2
        ind = np.where(mask.flatten(order='F'))[0]
    in1 = flat_in1[ind].astype(int)
    in2 = flat_in2[ind].astype(int)
    
    patchSize = len(in1)
    eind = np.arange(patchSize, 0, -1) - 1
    total_size = 2 * (2 * tau + 1)
    G_flat = np.zeros(( (2*(2*tau+1)) * (2*(2*tau+1)), Nc, Nc), dtype=complex)
    
    W = U @ U.conj().T
    W = W.reshape(patchSize, Nc, patchSize, Nc, order='F').transpose(0, 1, 3, 2)
    
    for s in range(patchSize):
        r0 = 2 * tau + 1 + in1[eind] + in1[s] 
        c0 = 2 * tau + 1 + in2[eind] + in2[s] 
        r0 = np.clip(r0, 0, total_size-1)
        c0 = np.clip(c0, 0, total_size-1)
        linear_idx = c0 * total_size + r0 
        G_flat[linear_idx, :, :] += W[:, :, :, s]

    G = G_flat.reshape(total_size, total_size, Nc, Nc, order='F')  
    
    # 采用 FFT Interpolation 扩展到全图大小 (N1, N2)
    N1_g, N2_g = N1, N2 
    n1 = np.fft.fftshift(np.fft.fftfreq(N1_g))
    n2 = np.fft.fftshift(np.fft.fftfreq(N2_g))
    n2, n1 = np.meshgrid(n2, n1, indexing='xy')
    phaseKernel = np.exp(-1j * 2 * np.pi * (n1 * (N1_g - 2*tau - 1) + n2 * (N2_g - 2*tau - 1)))
    
    G = np.fft.fft2(np.conj(G), s=(N1_g, N2_g), axes=(0,1), norm='ortho') * phaseKernel[:, :, np.newaxis, np.newaxis]
    G = np.fft.fftshift(G, axes=(0,1))
    
    return G

def compute_G_for_fastmri_slice(kspace_slice, cal_length=32, tau=3, threshold=0.08, kernel_shape=1):
    """
    针对 FastMRI 的预处理包装函数。
    输入:
        kspace_slice: numpy array, 形状为 (Nc, N1, N2)，即 (通道数, 宽, 高)
    输出:
        G_tensor: PyTorch Complex Tensor, 形状为 (N1, N2, Nc, Nc)
    """
    # 1. 调整维度 (Nc, N1, N2) -> (N1, N2, Nc) 以适配 PISCO 逻辑
    kData = np.transpose(kspace_slice, (1, 2, 0))
    N1, N2, Nc = kData.shape
    
    # 2. 提取 ACS 校准区 (Calibration Data)
    center_x = int(np.ceil(N1 / 2)) + even_pisco(N1)
    center_y = int(np.ceil(N2 / 2)) + even_pisco(N2)
    
    cal_index_x = np.arange(center_x - int(np.floor(cal_length / 2)), 
                            center_x + int(np.floor(cal_length / 2)) - even_pisco(cal_length))
    cal_index_y = np.arange(center_y - int(np.floor(cal_length / 2)), 
                            center_y + int(np.floor(cal_length / 2)) - even_pisco(cal_length))
    
    kCal = kData[cal_index_x[:, np.newaxis], cal_index_y, :]
    
    # 3. 核心计算：获取零空间 U，然后计算 G
    U = nullspace_vectors_C_matrix(kCal, tau, threshold, kernel_shape)
    G_matrix = G_matrices(kCal, N1, N2, tau, U, kernel_shape)
    
    # 4. 转化为 PyTorch Tensor 输出，方便直接存入 DataLoader
    # G_matrix 形状已经是 (N1, N2, Nc, Nc)
    G_tensor = torch.from_numpy(G_matrix).to(torch.complex64)
    
    return G_tensor
