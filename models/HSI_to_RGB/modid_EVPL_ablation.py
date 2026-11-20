"""
modid_EVPL_ablation.py

MODID 数据集 EVPL 对比实验（消融）模块
针对 Reviewer Weakness：
    "Missing baselines: comparisons are only against RGB,
     no evaluation of existing band selection or filter optimization methods"

数据集参数：
    num_bands      = 16
    wavelengths    = [460.48, 465.22, 474.12, 482.58, 492.60, 504.37,
                      512.47, 521.99, 533.93, 541.20, 551.99, 559.72,
                      570.13, 579.73, 585.39, 595.02]  单位 nm
    std (原始固定) = 70.0 nm（波长范围窄~135nm，宽 std 保证充分覆盖）

论证链：
    Band Selection (PCA / SoftBand)  ← 无/弱学习
        ↓ 差距说明"学习光谱响应有价值"
    Free 1×1 Conv (无形状约束)       ← 完全数据驱动
        ↓ 差距说明"高斯形状约束有价值"
    Gaussian Unconstrained           ← 可训练高斯，无范围/间隔约束
        ↓ 差距说明"固定 λ 先验 + 宽 std 是否已足够"
    EVPL (固定 λ_max，固定 std=70)  ← 本文方法

注意：原始 modid_EVPL.EVPL 的 λ_max 是 requires_grad=False，
      GaussianUnconstrained 额外引入"可训练 λ"这一消融维度。
      由于 MODID 波段仅 16 个、范围窄（135nm），
      FreeConv 参数量仅 3×16=48，远少于一般数据集，过拟合风险较低。

训练脚本替换：
    ablation_map = {
        'evpl':        lambda: EVPL(490., 540., 570., device),
        'gauss_nocon': lambda: EVPL_GaussianUnconstrained(device),
        'free':        lambda: EVPL_FreeConv(device),
        'pca':         lambda: EVPL_PCA(device),
        'softband':    lambda: EVPL_SoftBandSelection(device),
    }
    model_evpl = ablation_map[args.ablation]()
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

try:
    from scipy.optimize import curve_fit
    HAS_SCIPY = True
except ImportError:
    HAS_SCIPY = False

from .modid_EVPL import EVPL

# MODID 数据集参数（16个不均匀波段）
NUM_BANDS = 16
_WAVELENGTHS_NP = np.array([
    460.480990, 465.215098, 474.119400, 482.578653, 492.599330, 504.367465,
    512.469275, 521.987657, 533.930667, 541.203169, 551.996143, 559.719978,
    570.133100, 579.729623, 585.387574, 595.018809
], dtype=np.float32)
SPECTRAL_RANGE = (float(_WAVELENGTHS_NP.min()), float(_WAVELENGTHS_NP.max()))


# ====================================================================
# FreeSpectralConv（内部辅助）
# ====================================================================

class _FreeSpectralConv(nn.Module):
    def __init__(self, device, num_bands=NUM_BANDS, wl_np=_WAVELENGTHS_NP,
                 out_channels=3, use_softplus=True):
        super().__init__()
        self.num_bands = num_bands
        self.out_channels = out_channels
        self.use_softplus = use_softplus
        self.conv = nn.Conv2d(num_bands, out_channels, kernel_size=1, bias=False)
        self.register_buffer('wavelengths', torch.tensor(wl_np, dtype=torch.float32))

    def forward(self, x):
        return self.conv(x)

    def _get_response(self):
        w = self.conv.weight.squeeze(-1).squeeze(-1)
        return F.softplus(w) if self.use_softplus else w.abs()

    def get_lambda_peak(self):
        r = self._get_response().detach().cpu()
        wl = self.wavelengths.cpu().numpy()
        return [float(wl[r[c].argmax().item()]) for c in range(self.out_channels)]

    def get_lambda_centroid(self):
        r = self._get_response().detach()
        wl = self.wavelengths
        return [float(((r[c] * wl).sum() / (r[c].sum() + 1e-8)).item())
                for c in range(self.out_channels)]

    def get_lambda_gaussian_fit(self):
        if not HAS_SCIPY:
            return None
        r = self._get_response().detach().cpu().numpy()
        wl = self.wavelengths.cpu().numpy()
        centroids = self.get_lambda_centroid()

        def _gauss(x, amp, mu, sigma):
            return amp * np.exp(-((x - mu) ** 2) / (2 * sigma ** 2))

        results = []
        for c in range(self.out_channels):
            rc = r[c]
            try:
                popt, _ = curve_fit(
                    _gauss, wl, rc,
                    p0=[float(rc.max()), centroids[c], 20.0],
                    bounds=([0, float(wl.min()) - 5, 1.0],
                            [np.inf, float(wl.max()) + 5, 200.0]),
                    maxfev=5000
                )
                r_pred = _gauss(wl, *popt)
                ss_res = np.sum((rc - r_pred) ** 2)
                ss_tot = np.sum((rc - rc.mean()) ** 2) + 1e-10
                results.append({'mu': float(popt[1]), 'sigma': float(abs(popt[2])),
                                 'r2': float(1.0 - ss_res / ss_tot)})
            except Exception:
                results.append({'mu': centroids[c], 'sigma': float('nan'), 'r2': float('nan')})
        return results

    def get_spectral_entropy(self):
        r = self._get_response().detach().cpu()
        res = []
        for c in range(self.out_channels):
            p = r[c] / (r[c].sum() + 1e-8)
            res.append(float(-(p * torch.log(p + 1e-10)).sum().item()))
        return res

    def print_lambda(self):
        peaks = self.get_lambda_peak()
        cents = self.get_lambda_centroid()
        ents = self.get_spectral_entropy()
        fits = self.get_lambda_gaussian_fit()
        for c in range(self.out_channels):
            fit_str = ''
            if fits:
                fit_str = (f", gauss μ={fits[c]['mu']:.2f} σ={fits[c]['sigma']:.2f}"
                           f" R²={fits[c]['r2']:.3f}")
            print(f"  ch{c+1}: peak={peaks[c]:.2f}nm centroid={cents[c]:.2f}nm"
                  f" H={ents[c]:.3f}{fit_str}")


# ====================================================================
# EVPL_GaussianUnconstrained
# ====================================================================

class EVPL_GaussianUnconstrained(nn.Module):
    """
    Filter Optimization Baseline — 可训练高斯核，无任何约束

    与原始 EVPL 的消融差异：
      ① λ_max 随机初始化（原始使用固定先验且不训练）
      ② λ_max requires_grad=True（原始为 False）
      ③ std 可训练（原始固定 70.0）
      ④ 无范围 clamp、无间隔约束

    注意：MODID 波段范围仅约 135nm（460-595nm），
          std 初始化为 20.0 以体现窄范围特性；
          原始固定 std=70.0 几乎覆盖全部波段，此处解除该约束。
    """

    def __init__(self, device,
                 num_bands: int = NUM_BANDS,
                 wl_np: np.ndarray = _WAVELENGTHS_NP,
                 out_channels: int = 3,
                 init_std: float = 20.0,
                 random_seed: int = 42):
        super().__init__()
        self.num_bands = num_bands
        self.out_channels = out_channels

        rng = np.random.RandomState(random_seed)
        init_lambdas = rng.uniform(float(wl_np.min()), float(wl_np.max()), out_channels).tolist()

        self.lamda_max = nn.ParameterList([
            nn.Parameter(torch.tensor(lam, device=device, dtype=torch.float32), requires_grad=True)
            for lam in init_lambdas
        ])
        self.log_std = nn.Parameter(
            torch.tensor(math.log(init_std), device=device, dtype=torch.float32),
            requires_grad=True
        )
        self.register_buffer('wavelengths', torch.tensor(wl_np, dtype=torch.float32))

        print(f"[EVPL_GaussianUnconstrained/MODID] init_λ={[f'{l:.1f}' for l in init_lambdas]}, "
              f"init_std={init_std} (可训练，无约束；原始 std=70.0 固定)")

    def forward(self, x):
        std = torch.exp(self.log_std)
        kernels = []
        for lam in self.lamda_max:
            w = torch.exp(-((self.wavelengths - lam) ** 2) / (2 * std ** 2))
            w = w / (w.sum() + 1e-6)
            kernels.append(w.unsqueeze(0).unsqueeze(2).unsqueeze(3))
        kernel = torch.cat(kernels, dim=0)
        out = F.conv2d(x, kernel)
        B, C, H, W = out.shape
        flat = out.view(B, C, -1)
        out_min = flat.min(dim=2, keepdim=True)[0].unsqueeze(-1)
        out_max = flat.max(dim=2, keepdim=True)[0].unsqueeze(-1)
        return (out - out_min) / (out_max - out_min + 1e-6)

    def get_lambda_max(self):
        std = float(torch.exp(self.log_std).detach().item())
        result = {'std': std}
        for i, lam in enumerate(self.lamda_max):
            result[f'lambda_{i+1}'] = float(lam.detach().item())
        return result

    def print_lambda(self):
        info = self.get_lambda_max()
        vals = [f"λ_{i+1}={info[f'lambda_{i+1}']:.2f}" for i in range(self.out_channels)]
        print(f"[EVPL_GaussianUnconstrained/MODID] {', '.join(vals)}, std={info['std']:.2f} nm")


# ====================================================================
# EVPL_FreeConv
# ====================================================================

class EVPL_FreeConv(nn.Module):
    """
    Filter Optimization Baseline — 无约束自由 1×1 光谱卷积
    接口与 EVPL 完全一致。
    MODID 仅 16 波段，参数量 3×16=48，过拟合风险低。
    """

    def __init__(self, device,
                 num_bands: int = NUM_BANDS,
                 wl_np: np.ndarray = _WAVELENGTHS_NP):
        super().__init__()
        self.num_bands = num_bands
        self.filter_spectral_response = _FreeSpectralConv(
            device, num_bands=num_bands, wl_np=wl_np, out_channels=3
        )
        print(f"[EVPL_FreeConv/MODID] num_bands={num_bands} (3×{num_bands}={3*num_bands} 参数)")

    def forward(self, x):
        out = self.filter_spectral_response(x)
        B, C, H, W = out.shape
        flat = out.view(B, C, -1)
        out_min = flat.min(dim=2, keepdim=True)[0].unsqueeze(-1)
        out_max = flat.max(dim=2, keepdim=True)[0].unsqueeze(-1)
        return (out - out_min) / (out_max - out_min + 1e-6)

    def get_lambda_max(self):
        fsc = self.filter_spectral_response
        peaks = fsc.get_lambda_peak()
        cents = fsc.get_lambda_centroid()
        ents = fsc.get_spectral_entropy()
        fits = fsc.get_lambda_gaussian_fit()
        n = fsc.out_channels
        result = {}
        for i in range(n):
            result[f'lambda_{i+1}'] = cents[i]
            result[f'lambda_{i+1}_peak'] = peaks[i]
            result[f'entropy_{i+1}'] = ents[i]
        if fits:
            for i, fit in enumerate(fits):
                result[f'gauss_mu_{i+1}'] = fit['mu']
                result[f'gauss_sigma_{i+1}'] = fit['sigma']
                result[f'gauss_r2_{i+1}'] = fit['r2']
        return result

    def print_lambda(self):
        print("[EVPL_FreeConv/MODID]")
        self.filter_spectral_response.print_lambda()


# ====================================================================
# EVPL_PCA
# ====================================================================

class EVPL_PCA(nn.Module):
    """
    Band Selection Baseline — PCA（投影已在 ModidDataset_PCA 层完成）
    MODID 仅 16 波段，前 3 主成分通常能覆盖 >95% 方差。

    本模块仅接收已投影的 out_channels 通道输入，
    做通道归一化后输出。参数量 = 0。

    配套使用：
        from dataloader.modid import ModidDataset_PCA
        train_dataset = ModidDataset_PCA(root_dir, input_shape, train=True)
        # Dataset.__getitem__ 直接返回 [3, H, W] 已投影数据
        evpl_pca = EVPL_PCA(device).to(device)
    """

    def __init__(self, device, out_channels: int = 3, **kwargs):
        super().__init__()
        self.out_channels = out_channels
        print(f"[EVPL_PCA/MODID] 就绪，期望 {out_channels} 通道输入"
              f"（PCA 投影已在 ModidDataset_PCA 层完成）")

    def forward(self, x):
        # x: [B, out_channels, H, W] — 已由 Dataset_PCA 层完成 PCA 投影
        B, C, H, W = x.shape
        flat = x.view(B, C, -1)
        out_min = flat.min(dim=2, keepdim=True)[0].unsqueeze(-1)
        out_max = flat.max(dim=2, keepdim=True)[0].unsqueeze(-1)
        return (x - out_min) / (out_max - out_min + 1e-6)

    def get_lambda_max(self):
        return {f'lambda_{i+1}': float('nan') for i in range(self.out_channels)}

    def print_lambda(self):
        print("[EVPL_PCA/MODID] PCA 投影已在 ModidDataset_PCA 层完成，无模型内等效波长信息")


# ====================================================================
# EVPL_SoftBandSelection
# ====================================================================

class EVPL_SoftBandSelection(nn.Module):
    """
    Band Selection Baseline — Gumbel-Softmax 可微分波段选择
    推理时 hard argmax → λ_max = wavelengths[argmax(logits)]。
    MODID 仅 16 个候选波段，每 epoch 结束调用 anneal_temperature()。
    """

    def __init__(self, device,
                 num_bands: int = NUM_BANDS,
                 wl_np: np.ndarray = _WAVELENGTHS_NP,
                 out_channels: int = 3,
                 init_temperature: float = 1.0,
                 min_temperature: float = 0.1,
                 anneal_rate: float = 0.003):
        super().__init__()
        self.num_bands = num_bands
        self.out_channels = out_channels
        self.temperature = init_temperature
        self.min_temperature = min_temperature
        self.anneal_rate = anneal_rate
        self.logits = nn.Parameter(
            torch.randn(out_channels, num_bands, device=device, dtype=torch.float32)
        )
        self.register_buffer('wavelengths', torch.tensor(wl_np, dtype=torch.float32))
        print(f"[EVPL_SoftBandSelection/MODID] num_bands={num_bands}, T={init_temperature}")

    def anneal_temperature(self):
        self.temperature = max(self.min_temperature,
                               self.temperature * math.exp(-self.anneal_rate))

    def _get_weights(self):
        if self.training:
            return F.gumbel_softmax(self.logits, tau=self.temperature, hard=False, dim=-1)
        else:
            idx = self.logits.argmax(dim=-1, keepdim=True)
            return torch.zeros_like(self.logits).scatter_(-1, idx, 1.0)

    def forward(self, x):
        weights = self._get_weights()
        kernel = weights.unsqueeze(-1).unsqueeze(-1)
        out = F.conv2d(x, kernel)
        B, C, H, W = out.shape
        flat = out.view(B, C, -1)
        out_min = flat.min(dim=2, keepdim=True)[0].unsqueeze(-1)
        out_max = flat.max(dim=2, keepdim=True)[0].unsqueeze(-1)
        return (out - out_min) / (out_max - out_min + 1e-6)

    def get_lambda_max(self):
        wl = self.wavelengths
        hard_idx = self.logits.detach().argmax(dim=-1)
        soft_w = F.softmax(self.logits.detach(), dim=-1)
        result = {'temperature': self.temperature}
        for i in range(self.out_channels):
            result[f'lambda_{i+1}'] = float(wl[hard_idx[i]].item())
            result[f'lambda_{i+1}_soft_centroid'] = float(
                ((soft_w[i] * wl).sum() / (soft_w[i].sum() + 1e-8)).item()
            )
        return result

    def print_lambda(self):
        info = self.get_lambda_max()
        print(f"[EVPL_SoftBandSelection/MODID] T={info['temperature']:.4f}")
        for i in range(self.out_channels):
            print(f"  ch{i+1}: hard λ={info[f'lambda_{i+1}']:.2f} nm  "
                  f"soft_centroid={info[f'lambda_{i+1}_soft_centroid']:.2f} nm")


# ====================================================================
# compare_all_baselines
# ====================================================================

def compare_all_baselines(models: dict,
                          wl_np: np.ndarray = _WAVELENGTHS_NP,
                          num_bands: int = NUM_BANDS,
                          save_path: str = None) -> dict:
    """横向对比 MODID 所有 baseline 的等效 λ_max 与光谱响应。"""
    sep = "=" * 80
    DISPLAY = {
        'evpl':        'EVPL (Fixed Gaussian)       [本文]',
        'gauss_nocon': 'Gaussian Unconstrained      [消融①]',
        'free':        'Free Spectral Conv          [消融②]',
        'pca':         'PCA Projection (fixed)      [Band Sel.]',
        'softband':    'Gumbel SoftBand Selection   [Band Sel.]',
    }
    print(sep)
    print("MODID 全基线光谱响应对比")
    print(sep)
    all_info = {}
    for name, model in models.items():
        print(f"\n[{DISPLAY.get(name, name)}]")
        model.print_lambda()
        all_info[name] = model.get_lambda_max()

    _sample_info = next(iter(all_info.values())) if all_info else {}
    n_ch = max(sum(1 for k in _sample_info if k.startswith('lambda_') and k[7:].isdigit()), 1)
    hdr = ' '.join(f"{'Ch'+str(i+1)+' (nm)':>12}" for i in range(n_ch))
    print(f"\n{'':30s} {hdr}")
    print("-" * (32 + 13 * n_ch))
    for name, info in all_info.items():
        disp = DISPLAY.get(name, name)[:28]
        lams = [info.get(f'lambda_{i+1}', float('nan')) for i in range(n_ch)]
        row = ' '.join(f"{l:12.2f}" for l in lams)
        print(f"  {disp:28s} {row}")

    if save_path is not None:
        try:
            import matplotlib
            matplotlib.use('Agg')
            import matplotlib.pyplot as plt
            n_models = len(models)
            _all_colors = ['royalblue', 'seagreen', 'tomato', 'darkorchid',
                           'darkorange', 'teal', 'crimson', 'olive']
            ch_colors = _all_colors[:n_ch]
            ch_titles = [f'Ch{i+1}' for i in range(n_ch)]
            fig, axes = plt.subplots(n_models, n_ch, figsize=(5 * n_ch, 3.5 * n_models), sharex=True)
            if n_models == 1:
                axes = axes[np.newaxis, :]
            if n_ch == 1:
                axes = axes[:, np.newaxis]
            for row, (name, model) in enumerate(models.items()):
                resps = _extract_spectral_response(model, wl_np, num_bands)
                for ch in range(n_ch):
                    ax = axes[row, ch]
                    if resps[ch] is not None:
                        r = resps[ch]
                        ax.plot(wl_np, r / (r.max() + 1e-8),
                                color=ch_colors[ch], linewidth=2, marker='o', markersize=4)
                    lam = all_info[name].get(f'lambda_{ch+1}', None)
                    if lam and not np.isnan(lam):
                        ax.axvline(lam, color='black', linestyle='--', alpha=0.7,
                                   label=f'λ={lam:.1f}nm')
                    if row == 0:
                        ax.set_title(ch_titles[ch], fontsize=10)
                    if ch == 0:
                        ax.set_ylabel(DISPLAY.get(name, name)[:20], fontsize=8)
                    if row == n_models - 1:
                        ax.set_xlabel('Wavelength (nm)')
                    ax.legend(fontsize=7)
                    ax.grid(True, alpha=0.3)
            fig.suptitle('MODID Spectral Response — All Baselines', fontsize=13)
            plt.tight_layout()
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            plt.close(fig)
            print(f"\n  [图像已保存] {save_path}")
        except ImportError:
            print("  (matplotlib 不可用，跳过可视化)")
    print(sep)
    return all_info


def _extract_spectral_response(model, wl_np, num_bands):
    try:
        if isinstance(model, EVPL):
            fsr = model.filter_spectral_response
            std = fsr.std
            wl_t = fsr.wavelengths
            resps = []
            for lam in [fsr.lamda_max_1, fsr.lamda_max_2, fsr.lamda_max_3]:
                g = torch.exp(-((wl_t - lam) ** 2) / (2 * std ** 2))
                g = g / (g.sum() + 1e-6)
                resps.append(g.detach().cpu().numpy())
            return resps
        elif isinstance(model, EVPL_GaussianUnconstrained):
            std = torch.exp(model.log_std)
            wl_t = model.wavelengths
            resps = []
            for lam in model.lamda_max:
                g = torch.exp(-((wl_t - lam) ** 2) / (2 * std ** 2))
                g = g / (g.sum() + 1e-6)
                resps.append(g.detach().cpu().numpy())
            return resps
        elif isinstance(model, EVPL_FreeConv):
            r = model.filter_spectral_response._get_response()
            n = model.filter_spectral_response.out_channels
            return [r[j].detach().cpu().numpy() for j in range(n)]
        elif isinstance(model, EVPL_PCA):
            return [None, None, None]  # 投影在数据集层，无模型内光谱响应
        elif isinstance(model, EVPL_SoftBandSelection):
            soft_w = F.softmax(model.logits.detach(), dim=-1)
            return [soft_w[j].cpu().numpy() for j in range(model.out_channels)]
    except Exception:
        pass
    return [None, None, None]


# ====================================================================
# 快速测试入口
# ====================================================================

if __name__ == "__main__":
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"device: {device}\n")
    print(f"MODID 波段数: {NUM_BANDS}, 波长范围: {SPECTRAL_RANGE[0]:.1f}~{SPECTRAL_RANGE[1]:.1f} nm\n")

    evpl = EVPL(490.0, 540.0, 570.0, device)

    gauss_nocon = EVPL_GaussianUnconstrained(device)
    free_conv   = EVPL_FreeConv(device)
    # PCA 投影在 ModidDataset_PCA 层完成，模型不需要投影矩阵
    pca_model   = EVPL_PCA(device)
    softband    = EVPL_SoftBandSelection(device)

    models = {
        'evpl': evpl, 'gauss_nocon': gauss_nocon,
        'free': free_conv, 'pca': pca_model, 'softband': softband,
    }

    print("[前向形状验证]")
    x = torch.randn(2, NUM_BANDS, 64, 64).to(device)
    x_pca = torch.randn(2, 3, 64, 64).to(device)   # PCA 已在 ModidDataset_PCA 层完成
    for name, m in models.items():
        x_in = x_pca if name == 'pca' else x
        out = m(x_in)
        print(f"  {name:14s}: {tuple(x_in.shape)} → {tuple(out.shape)}")

    print("\n[参数量]")
    for name, m in models.items():
        n = sum(p.numel() for p in m.parameters() if p.requires_grad)
        print(f"  {name:14s}: {n:6,}")

    compare_all_baselines(models, save_path="ablation_modid_baselines_init.png")
