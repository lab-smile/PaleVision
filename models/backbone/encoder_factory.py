"""
Post-SegFormer segmentation encoders via timm (features_only).

Selected encoders (published after / competitive with SegFormer 2021):
  - pvt_v2_b2   : PVT v2 (NeurIPS 2022)
  - convnext_tiny: ConvNeXt (CVPR 2022)
  - swinv2_tiny_window8_256: Swin Transformer V2 (2022)
  - poolformer_s12: PoolFormer / MetaFormer (CVPR 2022)
  - twins_svt_small: Twins-SVT (NeurIPS 2021, widely used post-SegFormer baseline)

Each wrapper exposes `.embed_dims` (4-stage channel dims) and returns
[C1, C2, C3, C4] feature maps compatible with SegFormerHead.
"""

import timm
import torch.nn as nn

# mit_b0 embed_dims for reference: [32, 64, 160, 256]
ENCODER_SPECS = {
    'mit_b0': {'type': 'mit', 'embed_dims': [32, 64, 160, 256]},
    'mit_b1': {'type': 'mit', 'embed_dims': [64, 128, 320, 512]},
    'pvt_v2_b2': {'type': 'timm', 'model': 'pvt_v2_b2', 'embed_dims': [64, 128, 320, 512]},
    'convnext_tiny': {'type': 'timm', 'model': 'convnext_tiny', 'embed_dims': [96, 192, 384, 768]},
    'swinv2_tiny': {'type': 'timm', 'model': 'swinv2_tiny_window8_256', 'embed_dims': [96, 192, 384, 768], 'fixed_size': 256},
    'poolformer_s12': {'type': 'timm', 'model': 'poolformer_s12', 'embed_dims': [64, 128, 320, 512]},
    'twins_svt_small': {'type': 'timm', 'model': 'twins_svt_small', 'embed_dims': [64, 128, 256, 512]},
}

POST_SEGFORMER_ENCODERS = [
    'pvt_v2_b2', 'convnext_tiny', 'swinv2_tiny', 'poolformer_s12', 'twins_svt_small'
]


class TimmEncoderBackbone(nn.Module):
    """Wrap timm features_only model to match MiT output interface."""

    def __init__(self, model_name, embed_dims, pretrained=False, fixed_size=None):
        super().__init__()
        self.embed_dims = embed_dims
        self.model_name = model_name
        self.fixed_size = fixed_size  # e.g. 256 for SwinV2 window models
        self.backbone = timm.create_model(
            model_name,
            features_only=True,
            out_indices=(0, 1, 2, 3),
            pretrained=pretrained,
            in_chans=3,
        )

    def forward(self, x):
        import torch.nn.functional as F
        b, c, h, w = x.shape
        if self.fixed_size is not None:
            x_in = F.interpolate(x, size=(self.fixed_size, self.fixed_size),
                                 mode='bilinear', align_corners=False)
        else:
            x_in = x
        feats = self.backbone(x_in)
        outs = []
        strides = [4, 8, 16, 32]
        for i, f in enumerate(feats):
            # Swin / some ViT backbones return NHWC
            if f.dim() == 4 and f.shape[-1] == self.embed_dims[i] and f.shape[1] != self.embed_dims[i]:
                f = f.permute(0, 3, 1, 2).contiguous()
            if self.fixed_size is not None:
                th, tw = max(1, h // strides[i]), max(1, w // strides[i])
                f = F.interpolate(f, size=(th, tw), mode='bilinear', align_corners=False)
            outs.append(f)
        return outs


def build_encoder(encoder_type, pretrained=False):
    """Build segmentation encoder; returns nn.Module with .embed_dims."""
    if encoder_type not in ENCODER_SPECS:
        raise ValueError(
            f"Unknown encoder '{encoder_type}'. "
            f"Choose from: {list(ENCODER_SPECS.keys())}"
        )
    spec = ENCODER_SPECS[encoder_type]
    if spec['type'] == 'mit':
        from models.backbone.mix_transformer import mit_b0, mit_b1, mit_b3, mit_b5
        factories = {
            'mit_b0': mit_b0, 'mit_b1': mit_b1,
            'mit_b3': mit_b3, 'mit_b5': mit_b5,
        }
        return factories[encoder_type]()
    return TimmEncoderBackbone(
        spec['model'],
        spec['embed_dims'],
        pretrained=pretrained,
        fixed_size=spec.get('fixed_size'),
    )
