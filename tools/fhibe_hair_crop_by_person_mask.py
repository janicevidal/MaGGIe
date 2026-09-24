#!/usr/bin/env python3
"""
根据人像 mask 的正方形外接框面积比例，对图像和头发 mask 进行裁切。
输入：图像文件夹、人像 mask 文件夹、头发 mask 文件夹。
输出：裁切后的图像和头发 mask（均保存为 PNG）。
"""

import os
import argparse
import cv2
import numpy as np
from glob import glob

# 支持的图像扩展名
IMG_EXTS = ('.jpg', '.jpeg', '.png', '.bmp', '.tif', '.tiff')

def get_person_bbox(mask):
    """返回人像最小外接矩形 (x, y, w, h)。若无则返回 None。"""
    coords = cv2.findNonZero(mask)
    if coords is None:
        return None
    return cv2.boundingRect(coords)

def bbox_to_containing_square(bbox, img_shape):
    """
    生成一个正方形 (x, y, side, side)，完全包含给定的矩形 bbox，
    且尽量不超出图像边界。如果无法完全包含（如 side 大于图像尺寸），返回 None。
    """
    x, y, w, h = bbox
    side = max(w, h)
    img_h, img_w = img_shape[:2]

    if side > img_w or side > img_h:
        return None

    left_min = max(0, x + w - side)
    left_max = min(x, img_w - side)
    if left_min > left_max:
        return None
    left = (left_min + left_max) // 2

    top_min = max(0, y + h - side)
    top_max = min(y, img_h - side)
    if top_min > top_max:
        return None
    top = (top_min + top_max) // 2

    return left, top, side, side

def square_area_ratio(square_bbox, img_shape):
    """计算正方形面积与整图面积的比值。"""
    _, _, side, _ = square_bbox
    img_h, img_w = img_shape[:2]
    sq_area = side * side
    img_area = img_w * img_h
    return sq_area / img_area if img_area > 0 else 0

def expand_and_fit(square_bbox, img_shape, expand_ratio=0.3):
    """
    将正方形向外扩展 expand_ratio，然后平移使其尽可能落在图像内，
    同时确保扩展后的矩形完全覆盖原始正方形。
    返回 (new_x, new_y, new_w, new_h)。
    """
    x, y, side, _ = square_bbox
    dw = int(side * expand_ratio)
    dh = dw
    new_w = side + dw
    new_h = side + dh
    cx = x + side // 2
    cy = y + side // 2
    new_x = cx - new_w // 2
    new_y = cy - new_h // 2
    img_h, img_w = img_shape[:2]

    new_x_min = max(0, x + side - new_w)
    new_x_max = min(x, img_w - new_w)
    if new_x_min <= new_x_max:
        new_x = (new_x_min + new_x_max) // 2
    else:
        new_x = max(new_x_min, new_x_max)
        new_x = min(max(new_x, 0), img_w - new_w) if new_w <= img_w else 0

    new_y_min = max(0, y + side - new_h)
    new_y_max = min(y, img_h - new_h)
    if new_y_min <= new_y_max:
        new_y = (new_y_min + new_y_max) // 2
    else:
        new_y = min(max(new_y, 0), img_h - new_h) if new_h <= img_h else 0

    if new_w > img_w:
        new_w = img_w
        new_x = 0
    if new_h > img_h:
        new_h = img_h
        new_y = 0

    if new_x > x:
        new_x = x
    if new_y > y:
        new_y = y
    if new_x + new_w < x + side:
        new_w = (x + side) - new_x
    if new_y + new_h < y + side:
        new_h = (y + side) - new_y

    return new_x, new_y, new_w, new_h

def ensure_same_size(img, mask):
    """将 mask 缩放到与 img 相同的尺寸。使用最近邻插值。"""
    h, w = img.shape[:2]
    mh, mw = mask.shape[:2]
    if (h, w) != (mh, mw):
        mask = cv2.resize(mask, (w, h), interpolation=cv2.INTER_NEAREST)
    return mask

def resize_image(img, max_long_edge):
    """缩放图像，长边不超过 max_long_edge。使用 INTER_AREA 插值。"""
    h, w = img.shape[:2]
    long_edge = max(h, w)
    if long_edge <= max_long_edge:
        return img
    scale = max_long_edge / long_edge
    new_w = int(w * scale)
    new_h = int(h * scale)
    return cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_AREA)

def resize_mask(mask, max_long_edge):
    """缩放 mask，长边不超过 max_long_edge。使用 INTER_NEAREST 插值。"""
    h, w = mask.shape[:2]
    long_edge = max(h, w)
    if long_edge <= max_long_edge:
        return mask
    scale = max_long_edge / long_edge
    new_w = int(w * scale)
    new_h = int(h * scale)
    return cv2.resize(mask, (new_w, new_h), interpolation=cv2.INTER_NEAREST)

def process_one(image_path, person_mask_path, hair_mask_path,
                output_image_path, output_hair_mask_path,
                expand_ratio=0.3, max_long_edge=1440, ratio_threshold=0.6):
    # 读取图像
    img = cv2.imread(image_path, cv2.IMREAD_COLOR)
    if img is None:
        print(f"  Cannot read image: {image_path}")
        return False

    # 读取人像 mask
    person_mask = cv2.imread(person_mask_path, cv2.IMREAD_GRAYSCALE)
    if person_mask is None:
        print(f"  Cannot read person mask: {person_mask_path}")
        return False
    _, person_mask = cv2.threshold(person_mask, 127, 255, cv2.THRESH_BINARY)

    # 读取头发 mask
    hair_mask = cv2.imread(hair_mask_path, cv2.IMREAD_GRAYSCALE)
    if hair_mask is None:
        print(f"  Cannot read hair mask: {hair_mask_path}")
        return False
    _, hair_mask = cv2.threshold(hair_mask, 127, 255, cv2.THRESH_BINARY)

    # 确保 mask 与图像尺寸一致
    person_mask = ensure_same_size(img, person_mask)
    hair_mask = ensure_same_size(img, hair_mask)

    # 获取人像 bbox
    bbox = get_person_bbox(person_mask)
    if bbox is None:
        print(f"  No person found in mask, keep full image.")
        crop_x, crop_y, crop_w, crop_h = 0, 0, img.shape[1], img.shape[0]
    else:
        square = bbox_to_containing_square(bbox, img.shape)
        if square is None:
            print(f"  Warning: Cannot create square containing the person, keep full image.")
            crop_x, crop_y, crop_w, crop_h = 0, 0, img.shape[1], img.shape[0]
        else:
            ratio = square_area_ratio(square, img.shape)
            print(f"  Square area ratio: {ratio:.2%}")
            if ratio >= ratio_threshold:
                print(f"  Ratio >= {ratio_threshold}, keep full image.")
                crop_x, crop_y, crop_w, crop_h = 0, 0, img.shape[1], img.shape[0]
            else:
                crop_bbox = expand_and_fit(square, img.shape, expand_ratio)
                crop_x, crop_y, crop_w, crop_h = crop_bbox
                print(f"  Cropped to expanded square ({crop_w}x{crop_h}).")

    # 裁切图像和头发 mask
    img_cropped = img[crop_y:crop_y+crop_h, crop_x:crop_x+crop_w]
    hair_mask_cropped = hair_mask[crop_y:crop_y+crop_h, crop_x:crop_x+crop_w]

    # 缩放
    img_resized = resize_image(img_cropped, max_long_edge)
    hair_mask_resized = resize_mask(hair_mask_cropped, max_long_edge)

    # 确保尺寸一致
    if img_resized.shape[:2] != hair_mask_resized.shape[:2]:
        hair_mask_resized = cv2.resize(hair_mask_resized,
                                       (img_resized.shape[1], img_resized.shape[0]),
                                       interpolation=cv2.INTER_NEAREST)

    # 保存
    os.makedirs(os.path.dirname(output_image_path), exist_ok=True)
    os.makedirs(os.path.dirname(output_hair_mask_path), exist_ok=True)
    cv2.imwrite(output_image_path, img_resized)
    cv2.imwrite(output_hair_mask_path, hair_mask_resized)
    print(f"  Saved image: {output_image_path}")
    print(f"  Saved hair mask: {output_hair_mask_path}")
    return True

def main():
    parser = argparse.ArgumentParser(description="根据人像 mask 裁切图像和头发 mask。")
    parser.add_argument('--image_dir', required=True, help='原始图像文件夹')
    parser.add_argument('--person_mask_dir', required=True, help='人像 mask 文件夹')
    parser.add_argument('--hair_mask_dir', required=True, help='头发 mask 文件夹')
    parser.add_argument('--output_image_dir', required=True, help='裁切后图像输出文件夹')
    parser.add_argument('--output_hair_mask_dir', required=True, help='裁切后头发 mask 输出文件夹')
    parser.add_argument('--image_ext', default='.png', help='图像扩展名（默认 .jpg）')
    parser.add_argument('--person_mask_ext', default='.png', help='人像 mask 扩展名（默认 .png）')
    parser.add_argument('--hair_mask_ext', default='.png', help='头发 mask 扩展名（默认 .png）')
    parser.add_argument('--overwrite', action='store_true', help='覆盖已存在的输出文件')
    parser.add_argument('--expand_ratio', type=float, default=0.3, help='正方形扩展比例（默认 0.3）')
    parser.add_argument('--max_long_edge', type=int, default=1440, help='缩放后长边最大值（默认 1440）')
    parser.add_argument('--ratio_threshold', type=float, default=0.7, help='面积比例阈值，大于等于此值则保留全图（默认 0.6）')
    args = parser.parse_args()

    os.makedirs(args.output_image_dir, exist_ok=True)
    os.makedirs(args.output_hair_mask_dir, exist_ok=True)

    # 收集图像文件
    image_files = []
    for ext in IMG_EXTS:
        image_files.extend(glob(os.path.join(args.image_dir, '*' + ext)))
        image_files.extend(glob(os.path.join(args.image_dir, '*' + ext.upper())))
    image_files = sorted(set(image_files))

    if not image_files:
        print(f"在 {args.image_dir} 中未找到图像文件")
        return

    print(f"找到 {len(image_files)} 个图像文件")

    success = 0
    for image_path in image_files:
        base = os.path.basename(image_path)
        name = os.path.splitext(base)[0]

        person_mask_path = os.path.join(args.person_mask_dir, name + args.person_mask_ext)
        hair_mask_path = os.path.join(args.hair_mask_dir, name + args.hair_mask_ext)

        if not os.path.exists(person_mask_path):
            print(f"未找到 {base} 对应的人像 mask，跳过。")
            continue
        if not os.path.exists(hair_mask_path):
            print(f"未找到 {base} 对应的头发 mask，跳过。")
            continue

        out_image_path = os.path.join(args.output_image_dir, name + '.png')
        out_hair_mask_path = os.path.join(args.output_hair_mask_dir, name + '.png')

        if not args.overwrite:
            if os.path.exists(out_image_path) and os.path.exists(out_hair_mask_path):
                print(f"{name} 的输出已存在，跳过。")
                continue

        print(f"\n处理 {base} ...")
        if process_one(image_path, person_mask_path, hair_mask_path,
                       out_image_path, out_hair_mask_path,
                       args.expand_ratio, args.max_long_edge, args.ratio_threshold):
            success += 1

    print(f"\n完成。成功处理 {success} / {len(image_files)} 个文件。")

if __name__ == '__main__':
    main()