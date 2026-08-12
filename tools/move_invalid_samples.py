#!/usr/bin/env python3
"""
将训练集中 alpha 蒙版无有效前景（像素 >127 的数量为0）的样本移动到单独文件夹。

用法示例:
    python tools/move_invalid_samples.py \
        --root_dir /data/xiaoshuai/human_matting/train_0729_resized/ \
        --output_dir /data/xiaoshuai/human_matting/train_invalid_removed/
"""

import os
import sys
import argparse
import shutil
import cv2
from tqdm import tqdm


def move_invalid_samples(root_dir, output_dir, threshold=0,
                         valid_image_extensions=('.jpg', '.jpeg', '.png', '.JPG', '.JPEG', '.PNG')):
    """
    检查训练集中所有图像对应的 alpha，若有效像素数 <= threshold，则移动该样本。

    Args:
        root_dir (str): 数据集根目录，应包含 'images/' 和 'alphas/' 子目录。
        output_dir (str): 用于存放无效样本的目标目录，将自动创建 images/ 和 alphas/ 子目录。
        threshold (int): 有效像素阈值，默认为 0（即至少有一个 >127 的像素才算有效）。
        valid_image_extensions (tuple): 图像文件扩展名列表。
    """
    image_dir = os.path.join(root_dir, 'images')
    alpha_dir = os.path.join(root_dir, 'alphas')

    if not os.path.isdir(image_dir):
        print(f"错误: 找不到图像目录 {image_dir}")
        return
    if not os.path.isdir(alpha_dir):
        print(f"错误: 找不到 alpha 目录 {alpha_dir}")
        return

    # 收集所有图像文件
    all_files = os.listdir(image_dir)
    images = [f for f in all_files if f.lower().endswith(valid_image_extensions)]
    total = len(images)
    print(f"扫描到 {total} 个图像文件。")

    # 创建输出目录
    out_image_dir = os.path.join(output_dir, 'images')
    out_alpha_dir = os.path.join(output_dir, 'alphas')
    os.makedirs(out_image_dir, exist_ok=True)
    os.makedirs(out_alpha_dir, exist_ok=True)

    moved_count = 0
    invalid_count = 0

    for img_name in tqdm(images, desc="处理样本"):
        # 构造对应的 alpha 文件名（假设扩展名为 .png）
        base_name = os.path.splitext(img_name)[0]
        alpha_name = base_name + '.png'
        alpha_path = os.path.join(alpha_dir, alpha_name)

        if not os.path.isfile(alpha_path):
            print(f"警告: alpha 文件不存在 {alpha_path}，跳过该样本。")
            continue

        # 读取 alpha 灰度图
        alpha_img = cv2.imread(alpha_path, cv2.IMREAD_GRAYSCALE)
        if alpha_img is None:
            print(f"警告: 无法读取 alpha 文件 {alpha_path}，跳过。")
            continue

        # 统计有效像素（>127）的数量
        valid_pixels = (alpha_img > 127).sum()
        if valid_pixels <= threshold:
            invalid_count += 1
            # 移动图像和 alpha 到新目录
            src_img = os.path.join(image_dir, img_name)
            dst_img = os.path.join(out_image_dir, img_name)
            src_alpha = alpha_path
            dst_alpha = os.path.join(out_alpha_dir, alpha_name)

            # 使用 shutil.move 移动文件（若想复制，可改为 shutil.copy2）
            shutil.move(src_img, dst_img)
            shutil.move(src_alpha, dst_alpha)
            moved_count += 1

    print("\n统计结果:")
    print(f"总图像数: {total}")
    print(f"无效样本数 (有效像素 <= {threshold}): {invalid_count}")
    print(f"已移动的样本数: {moved_count}")
    print(f"输出目录: {output_dir}")


def parse_args():
    parser = argparse.ArgumentParser(description="移除 alpha 中无前景的无效样本")
    parser.add_argument('--root_dir', required=True,
                        help='训练集根目录（包含 images/ 和 alphas/ 子目录）')
    parser.add_argument('--output_dir', required=True,
                        help='存放无效样本的目标目录（会自动创建 images/ 和 alphas/）')
    parser.add_argument('--threshold', type=int, default=0,
                        help='有效像素阈值（<=此值视为无效），默认为 0')
    return parser.parse_args()


if __name__ == '__main__':
    args = parse_args()
    move_invalid_samples(args.root_dir, args.output_dir, args.threshold)