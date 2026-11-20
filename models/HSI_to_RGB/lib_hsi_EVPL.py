import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from .invertible_isp_model.model import InvISPNet

# LIB-HSI 数据集波长数组 (204通道, 400-1000nm VNIR范围)
wavelengths = np.linspace(300, 700, 31)


class EVPL(nn.Module):
    """
    EVPL (Evolutionary Vision Perception Layer) for LIB-HSI dataset

    针对建筑立面高光谱数据集优化的视蛋白层模块
    - 204 光谱通道
    - 波长范围: 400-1000nm
    - 可学习的视蛋白波长参数
    """

    def __init__(self, lamda_max_1, lamda_max_2, lamda_max_3, device, use_isp=False):
        """
        Args:
            lamda_max_1: 第一个视蛋白的峰值波长 (nm)
            lamda_max_2: 第二个视蛋白的峰值波长 (nm)
            lamda_max_3: 第三个视蛋白的峰值波长 (nm)
            device: 运行设备
            use_isp: 是否使用可逆ISP网络
        """
        super(EVPL, self).__init__()
        self.filter_spectral_response = Gaussian_kernel_conv2d(
            lamda_max_1, lamda_max_2, lamda_max_3, device
        )
        self.two_to_rgb = nn.Conv2d(2, 3, kernel_size=(1, 1))
        self.use_isp = use_isp
        if use_isp:
            self.InvISPNet = InvISPNet()

    def forward(self, x):
        x = self.filter_spectral_response(x)
        if self.use_isp:
            x = self.InvISPNet(x)
        else:
            # 输出归一化到 [0, 1] 范围，确保 SegFormer 输入稳定
            # 使用 per-channel min-max normalization
            B, C, H, W = x.shape
            x_flat = x.view(B, C, -1)
            x_min = x_flat.min(dim=2, keepdim=True)[0].unsqueeze(-1)
            x_max = x_flat.max(dim=2, keepdim=True)[0].unsqueeze(-1)
            x = (x - x_min) / (x_max - x_min + 1e-6)
        x = self.two_to_rgb(x)
        return x

    def get_lambda_max(self):
        """返回当前学习到的视蛋白波长参数"""
        return {
            'lambda_1': self.filter_spectral_response.lamda_max_1.item(),
            'lambda_2': self.filter_spectral_response.lamda_max_2.item(),
            'lambda_3': self.filter_spectral_response.lamda_max_3.item(),
            'std': self.filter_spectral_response.std,  # 固定值
        }


class Gaussian_kernel_conv2d(nn.Module):
    """
    高斯核卷积层，模拟视蛋白的光谱响应曲线

    使用可学习的高斯核将高光谱图像转换为三通道信号
    """

    def __init__(self, lamda_max_1, lamda_max_2, lamda_max_3, device, trainable=True):
        """
        Args:
            lamda_max_1-3: 三个视蛋白的峰值波长
            device: 运行设备
            trainable: 波长参数是否可训练
        """
        super(Gaussian_kernel_conv2d, self).__init__()

        # 可学习的波长参数
        self.lamda_max_1 = nn.Parameter(
            torch.tensor(lamda_max_1, device=device, dtype=torch.float32), requires_grad=trainable
        )
        self.lamda_max_2 = nn.Parameter(
            torch.tensor(lamda_max_2, device=device, dtype=torch.float32), requires_grad=trainable
        )
        self.lamda_max_3 = nn.Parameter(
            torch.tensor(lamda_max_3, device=device, dtype=torch.float32), requires_grad=trainable
        )

        # 波长数组 (204 通道)
        self.register_buffer('wavelengths', torch.tensor(
            wavelengths, dtype=torch.float32
        ))

        # 固定的高斯核标准差 (控制光谱响应带宽)
        # 60nm：对于 400-1000nm 范围，覆盖约 10% 的波长，比之前的 30nm 更宽
        self.std = 20.0

        self.device = device

    def forward(self, input):
        """
        前向传播

        Args:
            input: 高光谱图像 [B, 204, H, W]

        Returns:
            output: 三通道输出 [B, 3, H, W]
        """
        # 动态计算高斯核权重 (支持波长参数的梯度更新)
        kernel_weight_1 = self._gaussian_kernel(self.lamda_max_1)
        kernel_weight_2 = self._gaussian_kernel(self.lamda_max_2)
        # kernel_weight_3 = self._gaussian_kernel(self.lamda_max_3)

        # 合并为 [3, 204, 1, 1] 的卷积核
        # kernel = torch.cat([kernel_weight_1, kernel_weight_2, kernel_weight_3], dim=0)
        kernel = torch.cat([kernel_weight_1, kernel_weight_2], dim=0)

        # 应用卷积
        output = F.conv2d(input, kernel)

        return output

    def _gaussian_kernel(self, lamda_max):
        """计算单个高斯核"""
        weight = torch.exp(-((self.wavelengths - lamda_max) ** 2) / (2 * self.std ** 2))
        weight = weight / (weight.sum() + 1e-6)
        return weight.unsqueeze(0).unsqueeze(2).unsqueeze(3)

    def print_lambda(self):
        """打印当前波长参数"""
        print(f'lamda_max_1: {self.lamda_max_1.data.item():.2f} nm, '
              f'lamda_max_2: {self.lamda_max_2.data.item():.2f} nm, '
              f'lamda_max_3: {self.lamda_max_3.data.item():.2f} nm, '
              f'std: {self.std:.2f} nm (fixed)')


class EVPL_Tetrachromatic(nn.Module):
    """
    四色视觉 EVPL 模块 (模拟鸟类等四色视觉生物)
    """

    def __init__(self, lamda_max_1, lamda_max_2, lamda_max_3, lamda_max_4, device):
        super(EVPL_Tetrachromatic, self).__init__()
        self.filter_spectral_response = Gaussian_kernel_conv2d_4(
            lamda_max_1, lamda_max_2, lamda_max_3, lamda_max_4, device
        )

    def forward(self, x):
        return self.filter_spectral_response(x)


class Gaussian_kernel_conv2d_4(nn.Module):
    """四通道高斯核卷积层"""

    def __init__(self, lamda_max_1, lamda_max_2, lamda_max_3, lamda_max_4, device, trainable=True):
        super(Gaussian_kernel_conv2d_4, self).__init__()

        self.lamda_max_1 = nn.Parameter(torch.tensor(lamda_max_1, device=device), requires_grad=trainable)
        self.lamda_max_2 = nn.Parameter(torch.tensor(lamda_max_2, device=device), requires_grad=trainable)
        self.lamda_max_3 = nn.Parameter(torch.tensor(lamda_max_3, device=device), requires_grad=trainable)
        self.lamda_max_4 = nn.Parameter(torch.tensor(lamda_max_4, device=device), requires_grad=trainable)

        self.wavelengths = torch.tensor(wavelengths, dtype=torch.float32, device=device, requires_grad=True)
        self.std = 10.0
        self.device = device

    def forward(self, input):
        kernel_weight_1 = self._gaussian_kernel(self.lamda_max_1)
        kernel_weight_2 = self._gaussian_kernel(self.lamda_max_2)
        kernel_weight_3 = self._gaussian_kernel(self.lamda_max_3)
        kernel_weight_4 = self._gaussian_kernel(self.lamda_max_4)

        kernel = torch.cat([kernel_weight_1, kernel_weight_2, kernel_weight_3, kernel_weight_4], dim=0)
        output = F.conv2d(input, kernel)
        return output

    def _gaussian_kernel(self, lamda_max):
        weight = (1 / (self.std * (2 * math.pi) ** 0.5) *
                  torch.exp(-((self.wavelengths - lamda_max) ** 2) / (2 * self.std ** 2)))
        return weight.unsqueeze(0).unsqueeze(2).unsqueeze(3)


if __name__ == "__main__":
    # 测试 EVPL 模块
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # 人类三色视觉初始化波长 (蓝、绿、红)
    model = EVPL(
        lamda_max_1=450.0,  # S锥 (蓝)
        lamda_max_2=530.0,  # M锥 (绿)
        lamda_max_3=560.0,  # L锥 (红)
        device=device
    )
    model.to(device)

    # 测试输入
    input = torch.randn((4, 204, 256, 256)).to(device)
    output = model(input)
    print(f"Input shape: {input.shape}")
    print(f"Output shape: {output.shape}")
    print(f"Lambda max values: {model.get_lambda_max()}")
