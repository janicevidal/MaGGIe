#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
将 512x1024 的图像及对应 mask 裁剪为 512x512（只保留上半部分）。

用法示例：
    python crop_top.py \
        --image_dir  data/images \
        --mask_dir   data/masks \
        --out_image_dir data/images_top \
        --out_mask_dir  data/masks_top

目录结构要求：
    image_dir/xxx.png  与  mask_dir/xxx.png  文件名（不含扩展名）一致，扩展名可不同。
"""

import argparse
import sys
from pathlib import Path

from PIL import Image

# 支持的图片扩展名
IMAGE_EXTS = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff", ".webp"}


def parse_args():
    p = argparse.ArgumentParser(
        description="把 512x1024 的图像和对应 mask 裁剪为 512x512（只保留上半部分）。"
    )
    p.add_argument("--image_dir", required=True, help="原图文件夹")
    p.add_argument("--mask_dir", required=True, help="mask 文件夹")
    p.add_argument("--out_image_dir", required=True, help="裁剪后图像输出文件夹")
    p.add_argument("--out_mask_dir", required=True, help="裁剪后 mask 输出文件夹")

    # 裁剪区域（默认：从左上角开始，宽=原图宽，高=512）
    p.add_argument("--top", type=int, default=0, help="裁剪起始 y 坐标，默认 0")
    p.add_argument("--height", type=int, default=512, help="裁剪高度，默认 512")
    p.add_argument("--width", type=int, default=None,
                   help="裁剪宽度，默认取原图宽度（512x1024 -> 512）")

    p.add_argument("--mask_out_ext", type=str, default=None,
                   help="mask 输出扩展名（如 .png）。默认保持原扩展名")
    p.add_argument("--strict", action="store_true",
                   help="图像与 mask 尺寸不一致时直接报错退出")
    p.add_argument("--dry_run", action="store_true",
                   help="只打印将要处理的文件，不实际写出")
    return p.parse_args()


def index_by_stem(folder: Path) -> dict:
    """建立 {文件名(不含扩展名): 路径} 的索引，方便图像与 mask 按名字配对。"""
    files = {}
    for p in sorted(folder.iterdir()):
        if p.is_file() and p.suffix.lower() in IMAGE_EXTS:
            # 同名不同扩展名时保留第一个
            files.setdefault(p.stem, p)
    return files


def crop_one(src, dst, top, height, width, out_ext=None):
    """
    裁剪单张图片并保存。
    返回 (原始尺寸, 裁剪后尺寸)。
    """
    with Image.open(src) as im:
        w, h = im.size

        right = w if width is None else min(width, w)
        bottom = min(top + height, h)
        if top >= h:
            raise ValueError(f"top={top} 超出图像高度 {h}")

        cropped = im.crop((0, top, right, bottom))

        dst = dst.with_suffix(out_ext) if out_ext else dst
        dst.parent.mkdir(parents=True, exist_ok=True)

        # 根据扩展名决定保存参数
        suffix = dst.suffix.lower()
        if suffix in {".jpg", ".jpeg"}:
            cropped.convert("RGB").save(dst, quality=95)
        else:
            cropped.save(dst)

    return (w, h), cropped.size


def main():
    args = parse_args()

    image_dir = Path(args.image_dir)
    mask_dir = Path(args.mask_dir)
    out_image_dir = Path(args.out_image_dir)
    out_mask_dir = Path(args.out_mask_dir)

    for d in (image_dir, mask_dir):
        if not d.is_dir():
            sys.exit(f"[错误] 目录不存在：{d}")

    images = index_by_stem(image_dir)
    masks = index_by_stem(mask_dir)

    if not images:
        sys.exit(f"[错误] 在 {image_dir} 中未找到任何图像文件")

    matched = sorted(set(images) & set(masks))
    only_img = sorted(set(images) - set(masks))
    only_mask = sorted(set(masks) - set(images))

    print(f"原图 {len(images)} 张，mask {len(masks)} 个，成功配对 {len(matched)} 对")
    if only_img:
        print(f"[警告] {len(only_img)} 张原图没有对应 mask，已跳过（示例：{only_img[:3]}）")
    if only_mask:
        print(f"[警告] {len(only_mask)} 个 mask 没有对应原图，已跳过（示例：{only_mask[:3]}）")

    if not matched:
        sys.exit("[错误] 没有任何配对成功的图像/mask")

    ok = 0
    for stem in matched:
        img_path, mask_path = images[stem], masks[stem]

        # 尺寸检查
        with Image.open(img_path) as im:
            img_size = im.size
        with Image.open(mask_path) as mk:
            mask_size = mk.size
        if img_size != mask_size:
            msg = f"{img_path.name}{img_size} 与 {mask_path.name}{mask_size} 尺寸不一致"
            if args.strict:
                sys.exit(f"[错误] {msg}")
            print(f"[警告] {msg}，将各自按自身尺寸裁剪")

        out_img = out_image_dir / img_path.name
        out_msk = out_mask_dir / mask_path.name

        if args.dry_run:
            print(f"[dry-run] {img_path.name} -> {out_img.name} | {mask_path.name} -> {out_msk.name}")
            continue

        try:
            _, new_size = crop_one(img_path, out_img, args.top, args.height, args.width)
            crop_one(mask_path, out_msk, args.top, args.height, args.width,
                     out_ext=args.mask_out_ext)
        except Exception as e:
            print(f"[错误] 处理 {stem} 失败：{e}")
            continue

        ok += 1
        print(f"[{ok}/{len(matched)}] {stem}: {img_size} -> {new_size}")

    if args.dry_run:
        print("dry-run 结束，未写出任何文件")
    else:
        print(f"完成！共处理 {ok} 对。\n  图像输出：{out_image_dir}\n  mask 输出：{out_mask_dir}")


if __name__ == "__main__":
    main()