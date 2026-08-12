#!/usr/bin/env python3
"""
用法示例:
    python tools/visualize_dataset.py --config configs/biref_matting.yaml \
        --split train --num_samples 5 --output_dir vis_samples -seed 26
"""

import os
import sys
import argparse
import numpy as np
import random
import cv2

sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from maggie.dataloader import build_dataset
from maggie.utils import CONFIG


def denormalize(img_tensor, mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)):
    """将归一化的图像张量 (C, H, W) 转换为 [0,255] 的 numpy 数组 (H, W, C)"""
    img = img_tensor.cpu().clone()
    if img.dim() == 4:
        img = img.squeeze(0)
    for t, m, s in zip(img, mean, std):
        t.mul_(s).add_(m)
    img = img.clamp(0, 1).numpy()
    img = (img * 255).astype(np.uint8)
    img = np.transpose(img, (1, 2, 0))  # CHW -> HWC
    return img


def get_base_name_from_path(image_path):
    """从图像路径提取文件名（不含扩展名），作为保存文件的前缀"""
    basename = os.path.basename(image_path)
    return os.path.splitext(basename)[0]


def save_original(image_path, alpha_path, output_dir, base_name):
    """保存原始图像和Alpha（未经任何预处理）"""
    img_orig = cv2.imread(image_path)
    img_orig = cv2.cvtColor(img_orig, cv2.COLOR_BGR2RGB)
    # alpha_orig = cv2.imread(alpha_path, cv2.IMREAD_GRAYSCALE)
    
    cv2.imwrite(os.path.join(output_dir, f"{base_name}_orig_image.png"), 
                cv2.cvtColor(img_orig, cv2.COLOR_RGB2BGR))
    # cv2.imwrite(os.path.join(output_dir, f"{base_name}_orig_alpha.png"), alpha_orig)


def save_augmented(sample, output_dir, base_name):
    """保存经过数据集transforms处理（增强/裁剪/归一化）后的图像和Alpha"""
    image = sample['image']
    alpha = sample['alpha']
    
    img_aug = denormalize(image)
    
    if alpha.dim() == 3:
        alpha_aug = alpha.squeeze(0).cpu().numpy()
    else:
        alpha_aug = alpha.cpu().numpy()
    alpha_aug = (alpha_aug * 255).astype(np.uint8)
    
    cv2.imwrite(os.path.join(output_dir, f"{base_name}_aug_image.png"), 
                cv2.cvtColor(img_aug, cv2.COLOR_RGB2BGR))
    cv2.imwrite(os.path.join(output_dir, f"{base_name}_aug_alpha.png"), alpha_aug)
    
    # 可选：白底合成预览
    # img_float = img_aug.astype(np.float32) / 255.0
    # alpha_float = alpha_aug.astype(np.float32) / 255.0
    # comp = img_float * alpha_float[..., None] + 1.0 * (1 - alpha_float[..., None])
    # comp = (comp * 255).astype(np.uint8)
    # cv2.imwrite(os.path.join(output_dir, f"{base_name}_aug_composite_white.png"), 
    #             cv2.cvtColor(comp, cv2.COLOR_RGB2BGR))


def visualize_dataset(cfg, split='train', num_samples=5, output_dir='vis_samples', seed=None):
    os.makedirs(output_dir, exist_ok=True)

    if seed is not None:
        random.seed(seed)
        np.random.seed(seed)
        
    dataset_cfg = cfg.dataset.train if split == 'train' else cfg.dataset.test
    is_train = (split == 'train')
    dataset = build_dataset(dataset_cfg, is_train=is_train)
    total = len(dataset)
    print(f"Dataset size: {total}")
    
    # 随机选择样本索引（不重复）
    if num_samples >= total:
        indices = list(range(total))
    else:
        indices = random.sample(range(total), num_samples)

    print(f"Selected indices: {sorted(indices)}")

    for idx in indices:
        image_path, alpha_path = dataset.data[idx]
        base_name = get_base_name_from_path(image_path)
        
        # 保存原始图
        save_original(image_path, alpha_path, output_dir, base_name)
        # 保存增强/预处理后的图
        sample = dataset[idx]
        save_augmented(sample, output_dir, base_name)
        
        print(f"Saved original & augmented for {base_name} (index {idx})")

    print(f"Visualization completed. Check {output_dir}")


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True, help="Path to config file")
    parser.add_argument("--split", default="train", choices=["train", "test"])
    parser.add_argument("--num_samples", type=int, default=10)
    parser.add_argument("--output_dir", default="vis_samples")
    parser.add_argument("--seed", type=int, default=None, help="Random seed for reproducibility")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    CONFIG.merge_from_file(args.config)
    visualize_dataset(CONFIG, args.split, args.num_samples, args.output_dir, args.seed)