import torch

class MRPhysicsOperators:
    @staticmethod
    def fft2c(x):
        """对最后两个维度(H, W)进行中心化二维正向傅里叶变换"""
        x = torch.fft.ifftshift(x, dim=(-2, -1))
        x = torch.fft.fft2(x, norm="ortho")
        x = torch.fft.fftshift(x, dim=(-2, -1))
        return x

    @staticmethod
    def ifft2c(x):
        """对最后两个维度(H, W)进行中心化二维逆傅里叶变换"""
        x = torch.fft.ifftshift(x, dim=(-2, -1))
        x = torch.fft.ifft2(x, norm="ortho")
        x = torch.fft.fftshift(x, dim=(-2, -1))
        return x
