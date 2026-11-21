"""
MinnApple detection & counting with Opsin + MiT + decoder.

Detection : Opsin -> MiT -> SegFormerHead (semantic seg, mIoU/F1)
Counting  : Opsin -> MiT -> CountingHead (global count regression, MAE/RMSE)

Opsin modes:
  --opsin_mode fixed     : RGBOpsinLayer with frozen λ_max
  --opsin_mode trainable : RGBOpsinLayer with learnable λ_max

Example:
  torchrun --nproc_per_node=1 minneapple_det_count_exp.py \\
      --task detection --opsin_mode fixed --workdir work_dir/minne_det_fixed
"""

import argparse
import datetime
import json
import logging
import os

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from omegaconf import OmegaConf
from torch.utils.data import DataLoader

from dataloader.minneapple_ninja import (
    MinneAppleDetectionDataset,
    MinneAppleCountingDataset,
    detection_metrics,
    counting_metrics,
)
from models.backbone.mix_transformer import mit_b0
from models.decoder.segformer_head import SegFormerHead
from models.HSI_to_RGB.rgb_opsin import RGBOpsinLayer


class CountingHead(nn.Module):
    """Global average pooling + MLP count regressor on MiT C4 features."""

    def __init__(self, in_channels=256, hidden=128):
        super().__init__()
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.mlp = nn.Sequential(
            nn.Linear(in_channels, hidden),
            nn.ReLU(inplace=True),
            nn.Linear(hidden, 1),
        )

    def forward(self, features):
        x = features[-1]
        x = self.pool(x).flatten(1)
        return self.mlp(x).squeeze(-1)


class MinneAppleModel(nn.Module):
    def __init__(self, task, opsin_trainable=False, num_classes=2):
        super().__init__()
        self.opsin = RGBOpsinLayer(trainable=opsin_trainable)
        self.backbone = mit_b0()
        self.task = task
        if task == 'detection':
            self.head = SegFormerHead(
                feature_strides=[4, 8, 16, 32],
                in_channels=self.backbone.embed_dims,
                embedding_dim=256,
                num_classes=num_classes,
            )
        else:
            self.head = CountingHead(in_channels=self.backbone.embed_dims[-1])

    def forward(self, x):
        feat = self.backbone(self.opsin(x))
        if self.task == 'detection':
            return self.head(feat)
        return self.head(feat)


def get_args():
    p = argparse.ArgumentParser()
    p.add_argument('--task', required=True, choices=['detection', 'counting'])
    p.add_argument('--opsin_mode', default='fixed', choices=['fixed', 'trainable'])
    p.add_argument('--data_root', default=os.path.join('datasets', 'minneapple'))
    p.add_argument('--config', default='configs/minneapple.yaml')
    p.add_argument('--workdir', default='work_dir/minneapple_det_count')
    p.add_argument('--counting_max_train', type=int, default=10000,
                   help='Cap counting-train samples for feasible runtime')
    p.add_argument('--seed', type=int, default=42)
    return p.parse_args()


def setup_logger(path):
    fmt = logging.Formatter('%(asctime)s - %(levelname)s: %(message)s')
    log = logging.getLogger()
    log.setLevel(logging.INFO)
    log.handlers.clear()
    for h in (logging.FileHandler(path, mode='w'), logging.StreamHandler()):
        h.setFormatter(fmt)
        log.addHandler(h)


@torch.no_grad()
def evaluate_detection(model, loader, device):
    model.eval()
    stats = {'iou': [], 'f1': []}
    for img, mask in loader:
        img = img.to(device)
        logits = model(img)
        logits = F.interpolate(logits, size=mask.shape[1:], mode='bilinear', align_corners=False)
        pred = logits.argmax(dim=1).cpu().numpy()
        gt = mask.numpy()
        for i in range(pred.shape[0]):
            m = detection_metrics(pred[i], gt[i])
            stats['iou'].append(m['iou'])
            stats['f1'].append(m['f1'])
    return {'mIoU': float(np.mean(stats['iou'])), 'F1': float(np.mean(stats['f1']))}


@torch.no_grad()
def evaluate_counting(model, loader, device):
    model.eval()
    preds, gts = [], []
    for img, count in loader:
        img = img.to(device)
        out = model(img)
        preds.extend(out.cpu().numpy().tolist())
        gts.extend(count.numpy().tolist())
    return counting_metrics(preds, gts)


def train(args, cfg):
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    os.makedirs(args.workdir, exist_ok=True)
    log_path = os.path.join(args.workdir, f'{args.task}_{args.opsin_mode}.log')
    setup_logger(log_path)
    logging.info('Task=%s opsin=%s device=%s', args.task, args.opsin_mode, device)

    if args.task == 'detection':
        train_ds = MinneAppleDetectionDataset(args.data_root, 'train', (512, 512))
        val_ds = MinneAppleDetectionDataset(args.data_root, 'val', (512, 512))
        bs = 4
    else:
        train_ds = MinneAppleCountingDataset(
            args.data_root, 'train', (128, 128), max_samples=args.counting_max_train)
        val_ds = MinneAppleCountingDataset(args.data_root, 'val', (128, 128))
        bs = 32

    train_loader = DataLoader(train_ds, batch_size=bs, shuffle=True, num_workers=4, drop_last=True)
    val_loader = DataLoader(val_ds, batch_size=bs, shuffle=False, num_workers=4)

    model = MinneAppleModel(
        task=args.task,
        opsin_trainable=(args.opsin_mode == 'trainable'),
    ).to(device)

    if args.task == 'detection':
        criterion = nn.CrossEntropyLoss()
        optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4, weight_decay=1e-4)
    else:
        criterion = nn.SmoothL1Loss()
        optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4, weight_decay=1e-4)

    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=cfg.train.num_epoch)
    best = {}
    t0 = datetime.datetime.now()

    for epoch in range(cfg.train.num_epoch):
        model.train()
        losses = []
        for img, target in train_loader:
            img = img.to(device)
            optimizer.zero_grad()
            if args.task == 'detection':
                target = target.to(device)
                out = model(img)
                out = F.interpolate(out, size=target.shape[1:], mode='bilinear', align_corners=False)
                loss = criterion(out, target.long())
            else:
                target = target.to(device)
                out = model(img)
                loss = criterion(out, target)
            loss.backward()
            optimizer.step()
            losses.append(loss.item())

        scheduler.step()
        if (epoch + 1) % cfg.train.eval_epoch == 0 or epoch == 0:
            if args.task == 'detection':
                metrics = evaluate_detection(model, val_loader, device)
            else:
                metrics = evaluate_counting(model, val_loader, device)
            key = 'mIoU' if args.task == 'detection' else 'MAE'
            score = metrics[key]
            better = (best.get(key) is None or
                      (key == 'mIoU' and score > best.get(key, 0)) or
                      (key == 'MAE' and score < best.get(key, 1e9)))
            if better:
                best = metrics
                torch.save(model.state_dict(), os.path.join(args.workdir, 'best.pth'))
            elapsed = datetime.datetime.now() - t0
            logging.info('Epoch %d/%d loss=%.4f elapsed=%s metrics=%s best=%s',
                         epoch + 1, cfg.train.num_epoch, np.mean(losses), elapsed, metrics, best)

    result = {
        'task': args.task,
        'opsin_mode': args.opsin_mode,
        'decoder': 'SegFormerHead' if args.task == 'detection' else 'CountingHead',
        'encoder': 'mit_b0',
        'metrics': best,
    }
    with open(os.path.join(args.workdir, 'result.json'), 'w') as f:
        json.dump(result, f, indent=2)
    logging.info('Final result: %s', result)
    return result


if __name__ == '__main__':
    args = get_args()
    cfg = OmegaConf.load(args.config)
    train(args, cfg)
