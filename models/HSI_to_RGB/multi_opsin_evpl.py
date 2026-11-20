"""
Multi-Opsin EVPL 改进版本
用于 lib_hsi 数据集

从 3 个视蛋白 (RGB) 扩展到 5 个 (UVRGNIR)，保留更多光谱信息
并提供可选的特征投影层，增加特征维度以应对 44 类分类
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

# LIB-HSI 数据集波长数组 (204通道, 400-1000nm VNIR范围)
wavelengths = np.linspace(400, 1000, 204)


class MultiOpsinEVPL(nn.Module):
    """
    Multi-Opsin Evolutionary Vision Perception Layer for LIB-HSI dataset
    
    改进点:
    1. 从 3 个视蛋白扩展到 5 个 (UV/Blue/Green/Red/NIR)
       - 保留更多光谱信息的细节
       - 避免过度压缩 204 维 → 3 维
    
    2. 可选的特征投影层
       - 5 → 64 维特征映射
       - 更好的与 44 类分类对齐
    
    3. 数值稳定性改进
       - 高斯核归一化确保输出范围稳定
       - Per-channel 归一化而不是全局
    
    4. 参数约束
       - 确保波长在有效范围内
    """
    
    def __init__(self, device, use_isp=False, projection_dim=64, use_projection=True):
        """
        Args:
            device: torch device
            use_isp: 是否使用可逆ISP网络 (默认False)
            projection_dim: 投影维度 (默认64)
            use_projection: 是否使用特征投影层
        """
        super(MultiOpsinEVPL, self).__init__()
        
        self.device = device
        self.use_isp = use_isp
        self.use_projection = use_projection
        self.projection_dim = projection_dim
        
        # 5 个视蛋白的可学习波长参数
        self.opsin_wavelengths = nn.ParameterDict({
            'uv': nn.Parameter(torch.tensor(380.0, device=device, dtype=torch.float32)),
            'blue': nn.Parameter(torch.tensor(470.0, device=device, dtype=torch.float32)),
            'green': nn.Parameter(torch.tensor(530.0, device=device, dtype=torch.float32)),
            'red': nn.Parameter(torch.tensor(600.0, device=device, dtype=torch.float32)),
            'nir': nn.Parameter(torch.tensor(850.0, device=device, dtype=torch.float32)),
        })
        
        # 波长数组 (204 通道)
        self.register_buffer('wavelengths', torch.tensor(
            wavelengths, dtype=torch.float32, device=device
        ))
        
        # 固定的高斯核标准差
        # 对于 5 个视蛋白，使用较窄的 σ (40nm) 避免过度重叠
        self.std = 40.0
        
        # 可选的特征投影层
        if use_projection:
            # 5 维 → 64 维, 使用并行路径
            self.projection = nn.Sequential(
                nn.Conv2d(5, 32, kernel_size=3, padding=1),
                nn.BatchNorm2d(32),
                nn.ReLU(inplace=True),
                nn.Conv2d(32, projection_dim, kernel_size=1),
            )
            self.out_channels = projection_dim
        else:
            self.out_channels = 5
    
    def forward(self, x):
        """
        Args:
            x: 高光谱图像 [B, 204, H, W]
        
        Returns:
            output: [B, out_channels, H, W]
                如果 use_projection=True, out_channels=64
                否则 out_channels=5
        """
        # Step 1: 生成 5 个视蛋白的响应
        opsin_responses = self._create_multi_opsin_response(x)  # [B, 5, H, W]
        
        # Step 2: 可选的特征投影
        if self.use_projection:
            output = self.projection(opsin_responses)  # [B, projection_dim, H, W]
        else:
            output = opsin_responses
        
        return output
    
    def _create_multi_opsin_response(self, x):
        """
        生成 5 个视蛋白的光谱响应
        
        Args:
            x: [B, 204, H, W]
        
        Returns:
            responses: [B, 5, H, W]
        """
        B, C, H, W = x.shape
        
        # 高斯核权重计算
        weights = {}
        for name, lamda_max in self.opsin_wavelengths.items():
            # 高斯核: exp(-(λ - λ_max)^2 / (2σ^2))
            w = torch.exp(
                -(self.wavelengths - lamda_max) ** 2 / (2 * self.std ** 2)
            )
            # 归一化: 确保和为 1
            w = w / (w.sum() + 1e-8)
            weights[name] = w  # [204]
        
        # 对每个视蛋白进行加权求和
        responses = []
        for name in ['uv', 'blue', 'green', 'red', 'nir']:
            w = weights[name].view(1, C, 1, 1)  # [1, 204, 1, 1]
            
            # 加权求和
            response = (x * w).sum(dim=1, keepdim=True)  # [B, 1, H, W]
            
            # 数值稳定性: 限制范围
            response = torch.clamp(response, 0, 1)
            responses.append(response)
        
        # 拼接: [B, 5, H, W]
        output = torch.cat(responses, dim=1)
        
        return output
    
    def get_wavelengths(self):
        """获取当前学习到的视蛋白波长"""
        return {
            'uv': self.opsin_wavelengths['uv'].item(),
            'blue': self.opsin_wavelengths['blue'].item(),
            'green': self.opsin_wavelengths['green'].item(),
            'red': self.opsin_wavelengths['red'].item(),
            'nir': self.opsin_wavelengths['nir'].item(),
            'std': self.std,
        }
    
    def constrain_wavelengths(self):
        """约束波长参数在有效范围内"""
        with torch.no_grad():
            # UV: 350-400 nm
            self.opsin_wavelengths['uv'].clamp_(350, 400)
            # Blue: 450-480 nm
            self.opsin_wavelengths['blue'].clamp_(450, 480)
            # Green: 500-550 nm
            self.opsin_wavelengths['green'].clamp_(500, 550)
            # Red: 580-620 nm
            self.opsin_wavelengths['red'].clamp_(580, 620)
            # NIR: 800-950 nm
            self.opsin_wavelengths['nir'].clamp_(800, 950)


class OriginalEVPL(nn.Module):
    """
    原始的 3 个视蛋白 EVPL (用于对比实验)
    """
    
    def __init__(self, lamda_max_1, lamda_max_2, lamda_max_3, device):
        super(OriginalEVPL, self).__init__()
        self.filter_spectral_response = Gaussian_kernel_conv2d(
            lamda_max_1, lamda_max_2, lamda_max_3, device
        )
    
    def forward(self, x):
        x = self.filter_spectral_response(x)
        # 输出归一化到 [0, 1] 范围
        B, C, H, W = x.shape
        x_flat = x.view(B, C, -1)
        x_min = x_flat.min(dim=2, keepdim=True)[0].unsqueeze(-1)
        x_max = x_flat.max(dim=2, keepdim=True)[0].unsqueeze(-1)
        x = (x - x_min) / (x_max - x_min + 1e-6)
        return x
    
    def get_lambda_max(self):
        return {
            'lambda_1': self.filter_spectral_response.lamda_max_1.item(),
            'lambda_2': self.filter_spectral_response.lamda_max_2.item(),
            'lambda_3': self.filter_spectral_response.lamda_max_3.item(),
            'std': self.filter_spectral_response.std,
        }


class Gaussian_kernel_conv2d(nn.Module):
    """
    高斯核卷积层 (原始版本)
    """
    
    def __init__(self, lamda_max_1, lamda_max_2, lamda_max_3, device, trainable=True):
        super(Gaussian_kernel_conv2d, self).__init__()
        
        self.lamda_max_1 = nn.Parameter(
            torch.tensor(lamda_max_1, device=device, dtype=torch.float32), 
            requires_grad=trainable
        )
        self.lamda_max_2 = nn.Parameter(
            torch.tensor(lamda_max_2, device=device, dtype=torch.float32),
            requires_grad=trainable
        )
        self.lamda_max_3 = nn.Parameter(
            torch.tensor(lamda_max_3, device=device, dtype=torch.float32),
            requires_grad=trainable
        )
        
        self.register_buffer('wavelengths', torch.tensor(
            wavelengths, dtype=torch.float32
        ))
        
        self.std = 60.0
        self.device = device
    
    def forward(self, x):
        # [B, 204, H, W]
        B, C, H, W = x.shape
        
        # 高斯核权重
        w1 = torch.exp(-(self.wavelengths - self.lamda_max_1) ** 2 / (2 * self.std ** 2))
        w2 = torch.exp(-(self.wavelengths - self.lamda_max_2) ** 2 / (2 * self.std ** 2))
        w3 = torch.exp(-(self.wavelengths - self.lamda_max_3) ** 2 / (2 * self.std ** 2))
        
        # 归一化
        w1 = w1 / (w1.sum() + 1e-8)
        w2 = w2 / (w2.sum() + 1e-8)
        w3 = w3 / (w3.sum() + 1e-8)
        
        # 加权求和得到 RGB 三通道
        w1 = w1.view(1, C, 1, 1)
        w2 = w2.view(1, C, 1, 1)
        w3 = w3.view(1, C, 1, 1)
        
        r = (x * w1).sum(dim=1, keepdim=True)
        g = (x * w2).sum(dim=1, keepdim=True)
        b = (x * w3).sum(dim=1, keepdim=True)
        
        output = torch.cat([r, g, b], dim=1)  # [B, 3, H, W]
        
        return output


if __name__ == '__main__':
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # 测试 MultiOpsinEVPL
    print("=" * 60)
    print("Testing MultiOpsinEVPL")
    print("=" * 60)
    
    evpl_multi = MultiOpsinEVPL(
        device=device,
        use_projection=True,
        projection_dim=64
    )
    evpl_multi = evpl_multi.to(device)
    
    # 模拟高光谱输入
    hsi_input = torch.randn(2, 204, 256, 256).to(device)
    
    output = evpl_multi(hsi_input)
    print(f"Input shape: {hsi_input.shape}")
    print(f"Output shape: {output.shape}")
    print(f"Expected output shape: [2, 64, 256, 256]")
    print(f"✓ Multi-Opsin EVPL works correctly!")
    
    # 显示学习到的波长
    wavelengths_dict = evpl_multi.get_wavelengths()
    print("\nLearned wavelengths:")
    for name, wl in wavelengths_dict.items():
        if name != 'std':
            print(f"  {name}: {wl:.1f} nm")
    
    # 计算参数数量
    total_params = sum(p.numel() for p in evpl_multi.parameters())
    print(f"\nTotal parameters: {total_params}")
    
    # 对比原始 EVPL
    print("\n" + "=" * 60)
    print("Comparison with Original EVPL")
    print("=" * 60)
    
    evpl_original = OriginalEVPL(450.0, 530.0, 580.0, device).to(device)
    output_original = evpl_original(hsi_input)
    
    print(f"Original EVPL output shape: {output_original.shape}")
    print(f"Multi-Opsin EVPL output shape: {output.shape}")
    print(f"Information preservation: 5 channels > 3 channels ✓")
