"""MinnApple DatasetNinja loaders for detection (seg) and counting tasks."""

import base64
import json
import os
import zlib

import cv2
import numpy as np
import torch
from torch.utils.data import Dataset


def _decode_bitmap(obj, h, w):
    """Decode DatasetNinja bitmap mask to [H, W] uint8."""
    if obj.get('geometryType') != 'bitmap':
        return None
    origin = obj['bitmap']['origin']
    ox, oy = int(origin[0]), int(origin[1])
    raw = base64.b64decode(obj['bitmap']['data'])
    try:
        raw = zlib.decompress(raw)
    except zlib.error:
        pass
    bm = cv2.imdecode(np.frombuffer(raw, np.uint8), cv2.IMREAD_UNCHANGED)
    if bm is None:
        return None
    if bm.ndim == 3:
        bm = bm[:, :, 0]
    mask = np.zeros((h, w), dtype=np.uint8)
    bh, bw = bm.shape[:2]
    y1, x1 = oy, ox
    y2, x2 = min(oy + bh, h), min(ox + bw, w)
    mask[y1:y2, x1:x2] = np.maximum(mask[y1:y2, x1:x2], bm[:y2 - y1, :x2 - x1])
    return mask


class MinneAppleDetectionDataset(Dataset):
    """Detection via instance bitmap -> semantic mask.

    Note: detection-test in DatasetNinja has empty annotations.
    We hold out 15% of detection-train as validation.
    """

    TRAIN_RATIO = 0.85

    def __init__(self, root, split='train', img_size=(512, 512), seed=42):
        self.root = root
        self.img_size = img_size
        img_dir = os.path.join(root, 'detection-train', 'img')
        ann_dir = os.path.join(root, 'detection-train', 'ann')
        all_files = sorted([f for f in os.listdir(img_dir)
                            if f.lower().endswith(('.png', '.jpg', '.jpeg'))])
        rng = np.random.RandomState(seed)
        idx = rng.permutation(len(all_files))
        n_train = int(len(all_files) * self.TRAIN_RATIO)
        if split == 'train':
            self.files = [all_files[i] for i in idx[:n_train]]
        elif split in ('val', 'test'):
            self.files = [all_files[i] for i in idx[n_train:]]
        else:
            raise ValueError(f"split must be train/val/test, got {split}")
        self.img_dir = img_dir
        self.ann_dir = ann_dir

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        name = self.files[idx]
        img = cv2.imread(os.path.join(self.img_dir, name))
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        h0, w0 = img.shape[:2]
        ann_path = os.path.join(self.ann_dir, name + '.json')
        mask = np.zeros((h0, w0), dtype=np.uint8)
        if os.path.isfile(ann_path):
            ann = json.load(open(ann_path))
            for obj in ann.get('objects', []):
                bm = _decode_bitmap(obj, h0, w0)
                if bm is not None:
                    mask = np.maximum(mask, (bm > 0).astype(np.uint8))
        h, w = self.img_size
        img = cv2.resize(img, (w, h), interpolation=cv2.INTER_LINEAR)
        mask = cv2.resize(mask, (w, h), interpolation=cv2.INTER_NEAREST)
        img = torch.from_numpy(img.astype(np.float32) / 255.0).permute(2, 0, 1)
        mask = torch.from_numpy(mask.astype(np.uint8))
        return img, mask


class MinneAppleCountingDataset(Dataset):
    """Apple counting on fixed-size patches."""

    SPLITS = {
        'train': 'counting-train',
        'val': 'counting-val',
        'test': 'counting-test',
    }

    def __init__(self, root, split='train', img_size=(128, 128), max_samples=-1):
        self.root = root
        sub = self.SPLITS[split]
        self.img_dir = os.path.join(root, sub, 'img')
        self.ann_dir = os.path.join(root, sub, 'ann')
        self.img_size = img_size
        self.files = sorted([f for f in os.listdir(self.img_dir)
                             if f.lower().endswith('.png')])
        if max_samples > 0:
            self.files = self.files[:max_samples]

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        name = self.files[idx]
        img = cv2.imread(os.path.join(self.img_dir, name))
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        ann = json.load(open(os.path.join(self.ann_dir, name + '.json')))
        count = 0
        for t in ann.get('tags', []):
            if t.get('name') == 'count':
                count = int(t.get('value', 0))
        h, w = self.img_size
        img = cv2.resize(img, (w, h), interpolation=cv2.INTER_LINEAR)
        img = torch.from_numpy(img.astype(np.float32) / 255.0).permute(2, 0, 1)
        return img, torch.tensor(count, dtype=torch.float32)


def detection_metrics(pred_mask, gt_mask):
    """Binary seg metrics for detection task."""
    pred = (pred_mask > 0).astype(np.uint8)
    gt = (gt_mask > 0).astype(np.uint8)
    inter = (pred & gt).sum()
    union = ((pred | gt)).sum()
    iou = inter / (union + 1e-8)
    tp = inter
    fp = pred.sum() - tp
    fn = gt.sum() - tp
    prec = tp / (tp + fp + 1e-8)
    rec = tp / (tp + fn + 1e-8)
    f1 = 2 * prec * rec / (prec + rec + 1e-8)
    return {'iou': float(iou), 'precision': float(prec), 'recall': float(rec), 'f1': float(f1)}


def counting_metrics(pred_counts, gt_counts):
    pred = np.asarray(pred_counts, dtype=np.float64)
    gt = np.asarray(gt_counts, dtype=np.float64)
    mae = np.abs(pred - gt).mean()
    rmse = np.sqrt(((pred - gt) ** 2).mean())
    return {'MAE': float(mae), 'RMSE': float(rmse)}
