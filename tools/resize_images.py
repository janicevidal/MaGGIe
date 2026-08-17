#!/usr/bin/env python3
"""
文件夹图像缩放工具
将输入目录中的所有图像按比例缩放，使短边（宽或高中较小者）等于目标尺寸（默认 640px）。
输出目录将保持与输入目录相同的子文件夹结构。
"""

import os
import argparse
from PIL import Image

# 支持的图像扩展名（不区分大小写）
SUPPORTED_EXTENSIONS = {'.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.tif', '.webp'}


def resize_image_to_short_side(input_path: str, output_path: str, target_size: int):
    """
    处理单张图片：打开、按短边缩放到 target_size、保存到 output_path
    """
    try:
        with Image.open(input_path) as img:
            # 获取原始宽高
            w, h = img.size
            # 计算缩放比例，使短边 = target_size
            if w <= h:
                # 宽度是短边
                scale = target_size / w
                new_w = target_size
                new_h = int(h * scale)
            else:
                # 高度是短边
                scale = target_size / h
                new_h = target_size
                new_w = int(w * scale)

            # 如果原始尺寸已经等于目标尺寸，可跳过缩放（但为了统一还是执行 resize）
            if (w, h) != (new_w, new_h):
                img = img.resize((new_w, new_h), Image.Resampling.LANCZOS)

            # 确保输出目录存在
            os.makedirs(os.path.dirname(output_path), exist_ok=True)

            # 保存图片，保持原格式（根据扩展名自动选择）
            img.save(output_path, quality=95)  # quality 对 JPEG 有效，其他格式忽略
            print(f"已处理: {input_path} -> {output_path} ({new_w}x{new_h})")
    except Exception as e:
        print(f"处理失败: {input_path} - {e}")


def process_directory(input_dir: str, output_dir: str, target_size: int):
    """
    递归遍历 input_dir，处理所有支持的图像文件，保持目录结构保存到 output_dir
    """
    if not os.path.isdir(input_dir):
        print(f"错误: 输入目录 '{input_dir}' 不存在")
        return

    # 统计
    total = 0
    processed = 0

    for root, dirs, files in os.walk(input_dir):
        for file in files:
            # 检查扩展名
            ext = os.path.splitext(file)[1].lower()
            if ext not in SUPPORTED_EXTENSIONS:
                continue

            total += 1
            # 构造输入和输出完整路径
            input_path = os.path.join(root, file)
            # 计算相对路径，用于保持目录结构
            rel_path = os.path.relpath(input_path, input_dir)
            output_path = os.path.join(output_dir, rel_path)

            resize_image_to_short_side(input_path, output_path, target_size)
            processed += 1

    print(f"\n完成！共处理 {processed} 个图像文件（共扫描 {total} 个支持格式文件）")


def main():
    parser = argparse.ArgumentParser(
        description="将文件夹中的所有图像按比例缩放，使短边等于指定尺寸（默认 640）"
    )
    parser.add_argument("input_dir", help="输入文件夹路径")
    parser.add_argument("-o", "--output_dir", help="输出文件夹路径（默认在输入目录同级创建带 '_scaled' 后缀的文件夹）")
    parser.add_argument("-s", "--size", type=int, default=640, help="目标短边长度（像素），默认 640")
    args = parser.parse_args()

    # 自动生成输出目录
    if args.output_dir is None:
        base = os.path.dirname(args.input_dir.rstrip(os.sep))
        name = os.path.basename(args.input_dir.rstrip(os.sep))
        args.output_dir = os.path.join(base, name + "_scaled")

    print(f"输入目录: {args.input_dir}")
    print(f"输出目录: {args.output_dir}")
    print(f"目标短边尺寸: {args.size} px")

    process_directory(args.input_dir, args.output_dir, args.size)


if __name__ == "__main__":
    main()