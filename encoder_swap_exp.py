"""
Unified encoder-swap + opsin-mode experiment for MODID and MHSI.

Pipeline: HSI -> EVPL(opsin) -> Encoder -> SegFormerHead -> segmentation

Example:
  torchrun --nproc_per_node=4 encoder_swap_exp.py \\
      --dataset mhsi --encoder pvt_v2_b2 --opsin_mode free \\
      --config configs/modid.yaml --workdir work_dir/enc_mhsi_pvt_free
"""

import argparse
import datetime
import json
import logging
import os
import random
import sys

import numpy as np
import torch
import torch.distributed as dist
import torch.nn.functional as F
from omegaconf import OmegaConf
from torch.nn.parallel import DistributedDataParallel
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
from tqdm import tqdm

from dataloader.modid import ModidDataset
from dataloader.Data_Generate import Data_Generate_Bile
from models.backbone.encoder_factory import build_encoder, POST_SEGFORMER_ENCODERS, ENCODER_SPECS
from models.HSI_to_RGB.evpl_factory import build_evpl, OPSIN_MODES
from models.decoder.segformer_head import SegFormerHead
from util.optimizer import PolyWarmupAdamW
from util.validation import segmentation_validation, segmentation_validation_fr


def get_args():
    p = argparse.ArgumentParser()
    p.add_argument('--dataset', required=True, choices=['modid', 'mhsi'])
    p.add_argument('--encoder', default='mit_b0',
                   choices=list(ENCODER_SPECS.keys()))
    p.add_argument('--opsin_mode', default='opsin_train', choices=list(OPSIN_MODES))
    p.add_argument('--config', default='configs/modid.yaml')
    p.add_argument('--workdir', default='work_dir/encoder_swap')
    p.add_argument('--local_rank', default=os.getenv('LOCAL_RANK', 0), type=int)
    p.add_argument('--backend', default='nccl')
    p.add_argument('--seed', default=42, type=int)
    p.add_argument('--data_root', default=None,
                   help='Dataset directory. Defaults to datasets/MODID or datasets/MHSI.')
    p.add_argument('--pretrained_encoder', action='store_true')
    return p.parse_args()


def setup_logger(path):
    fmt = logging.Formatter('%(asctime)s - %(levelname)s: %(message)s')
    log = logging.getLogger()
    log.setLevel(logging.INFO)
    log.handlers.clear()
    fh = logging.FileHandler(path, mode='w')
    fh.setFormatter(fmt)
    log.addHandler(fh)
    sh = logging.StreamHandler()
    sh.setFormatter(fmt)
    log.addHandler(sh)


def cal_eta(t0, cur, total):
    now = datetime.datetime.now().replace(microsecond=0)
    scale = (total - cur) / float(cur)
    eta = (now - t0) * scale
    return str(now - t0), str(eta)


def build_dataloaders(dataset_name, local_rank, data_root=None):
    nw = 4
    if dataset_name == 'modid':
        root = data_root or os.path.join('datasets', 'MODID')
        shape = [270, 510]
        train_ds = ModidDataset(root_dir=root, input_shape=shape, train=True)
        val_ds = ModidDataset(root_dir=root, input_shape=shape, train=False)
        train_sampler = DistributedSampler(train_ds)
        train_loader = DataLoader(train_ds, batch_size=4, shuffle=False,
                                  num_workers=nw, pin_memory=True, drop_last=True,
                                  sampler=train_sampler, prefetch_factor=4)
        val_loader = DataLoader(val_ds, batch_size=4, shuffle=False,
                                num_workers=nw, pin_memory=True, drop_last=False)
        return train_loader, val_loader, train_sampler, 2

    root = data_root or os.path.join('datasets', 'MHSI')
    with open(os.path.join(root, 'train_val_test.json')) as f:
        split = json.load(f)
    train_files = split['train'] + split['val']
    test_files = split['test']
    img_root = os.path.join(root, 'MHSI')
    mask_root = os.path.join(root, 'Mask')
    train_imgs = [os.path.join(img_root, x) for x in train_files]
    train_masks = [os.path.join(mask_root, f'{x[:-4]}.png') for x in train_files]
    test_imgs = [os.path.join(img_root, x) for x in test_files]
    test_masks = [os.path.join(mask_root, f'{x[:-4]}.png') for x in test_files]
    train_ds = Data_Generate_Bile(train_imgs, train_masks, transform=None,
                                  principal_bands_num=-1, cutting=-1)
    test_ds = Data_Generate_Bile(test_imgs, test_masks, transform=None,
                                 principal_bands_num=-1, cutting=-1)
    train_sampler = DistributedSampler(train_ds)
    test_sampler = DistributedSampler(test_ds, shuffle=False)
    train_loader = DataLoader(train_ds, sampler=train_sampler, batch_size=8,
                              num_workers=nw, drop_last=True)
    val_loader = DataLoader(test_ds, sampler=test_sampler, batch_size=8,
                            shuffle=False, num_workers=nw, drop_last=False)
    return train_loader, val_loader, train_sampler, 2


def train(args, cfg):
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    random.seed(args.seed)

    torch.cuda.set_device(args.local_rank)
    dist.init_process_group(backend=args.backend,
                            timeout=datetime.timedelta(minutes=60))
    device = torch.device('cuda', args.local_rank)
    t0 = datetime.datetime.now().replace(microsecond=0)

    train_loader, val_loader, train_sampler, num_classes = build_dataloaders(
        args.dataset, args.local_rank, args.data_root)

    evpl = build_evpl(args.dataset, args.opsin_mode, device)
    evpl.to(device)
    evpl = DistributedDataParallel(evpl, device_ids=[args.local_rank],
                                   find_unused_parameters=True)

    backbone = build_encoder(args.encoder, pretrained=args.pretrained_encoder)
    backbone.to(device)
    backbone = DistributedDataParallel(backbone, device_ids=[args.local_rank],
                                       find_unused_parameters=True)

    embed_dims = backbone.module.embed_dims
    head = SegFormerHead(feature_strides=[4, 8, 16, 32],
                         in_channels=embed_dims,
                         embedding_dim=256,
                         num_classes=num_classes)
    head.to(device)
    head = DistributedDataParallel(head, device_ids=[args.local_rank],
                                   find_unused_parameters=True)

    criterion = torch.nn.CrossEntropyLoss(ignore_index=255).to(device)
    bb_opt = torch.optim.Adam(backbone.parameters(), lr=6e-5)
    hd_opt = torch.optim.SGD(head.parameters(), lr=5e-4, momentum=0.9, weight_decay=5e-4)
    ev_opt = torch.optim.Adam(filter(lambda p: p.requires_grad, evpl.parameters()), lr=3e-2)
    bb_sched = torch.optim.lr_scheduler.CosineAnnealingLR(bb_opt, T_max=cfg.train.num_epoch)
    hd_sched = torch.optim.lr_scheduler.CosineAnnealingLR(hd_opt, T_max=cfg.train.num_epoch)
    ev_sched = torch.optim.lr_scheduler.CosineAnnealingLR(ev_opt, T_max=cfg.train.num_epoch)

    validate_fn = (segmentation_validation_fr if args.dataset == 'modid'
                   else segmentation_validation)
    best_miou = 0.0

    for epoch in range(cfg.train.num_epoch):
        backbone.train()
        head.train()
        evpl.train()
        train_sampler.set_epoch(epoch)
        for step, data in enumerate(train_loader):
            hsi, label = data
            hsi = hsi.to(device, non_blocking=True)
            if args.dataset == 'modid':
                label = label.to(device, non_blocking=True)
            else:
                label = label.squeeze(1).type(torch.uint8).to(device, non_blocking=True)

            feat = backbone(evpl(hsi))
            out = head(feat)
            out = F.interpolate(out, size=label.shape[1:], mode='bilinear', align_corners=False)
            loss = criterion(out, label.long())

            bb_opt.zero_grad()
            hd_opt.zero_grad()
            ev_opt.zero_grad()
            loss.backward()
            bb_opt.step()
            hd_opt.step()
            ev_opt.step()

            if args.local_rank == 0 and step == 0:
                d, eta = cal_eta(t0, epoch + 1, cfg.train.num_epoch)
                logging.info('Epoch %d/%d  loss=%.4f  elapsed=%s  eta=%s',
                             epoch + 1, cfg.train.num_epoch, loss.item(), d, eta)

        bb_sched.step()
        hd_sched.step()
        ev_sched.step()

        if (epoch + 1) % cfg.train.eval_epoch == 0:
            metrics = validate_fn(backbone=backbone, seg_decoder=head,
                                  EVPL_model=evpl, criterion=criterion,
                                  data_loader=val_loader, num_classes=num_classes)
            if args.local_rank == 0:
                logging.info('Epoch %d validate: %s', epoch + 1, metrics)
                miou = float(metrics.get('Mean IoU', 0))
                if miou > best_miou:
                    best_miou = miou
                    torch.save({
                        'encoder': args.encoder,
                        'opsin_mode': args.opsin_mode,
                        'dataset': args.dataset,
                        'best_miou': best_miou,
                        'epoch': epoch + 1,
                        'backbone': backbone.module.state_dict(),
                        'head': head.module.state_dict(),
                        'evpl': evpl.module.state_dict(),
                    }, os.path.join(args.workdir, 'best.pth'))
                    logging.info('New best mIoU=%.4f at epoch %d', best_miou, epoch + 1)

    if args.local_rank == 0:
        with open(os.path.join(args.workdir, 'result.json'), 'w') as f:
            json.dump({
                'dataset': args.dataset,
                'encoder': args.encoder,
                'opsin_mode': args.opsin_mode,
                'best_miou': best_miou,
                'num_epoch': cfg.train.num_epoch,
            }, f, indent=2)
        logging.info('Done. best mIoU=%.4f', best_miou)


if __name__ == '__main__':
    args = get_args()
    cfg = OmegaConf.load(args.config)
    os.makedirs(args.workdir, exist_ok=True)
    tag = f'{args.dataset}_{args.encoder}_{args.opsin_mode}'
    log_path = os.path.join(args.workdir, f'{tag}.log')
    if args.local_rank == 0:
        setup_logger(log_path)
        logging.info('Experiment: dataset=%s encoder=%s opsin=%s',
                     args.dataset, args.encoder, args.opsin_mode)
        logging.info('Post-SegFormer encoders: %s', POST_SEGFORMER_ENCODERS)
        logging.info('Config: %s', cfg)
    train(args, cfg)
