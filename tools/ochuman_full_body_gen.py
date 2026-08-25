import json
import os
import numpy as np
from PIL import Image
from pycocotools import mask as maskUtils

def decode_segmentation(seg, image_info):
    """
    解码 segmentation，支持 RLE dict 和 polygon list。
    返回二值掩码 numpy array (H, W) dtype=uint8 (0/1)
    """
    if isinstance(seg, dict) and 'counts' in seg and 'size' in seg:
        # RLE 格式
        return maskUtils.decode(seg)
    elif isinstance(seg, list) and len(seg) > 0:
        # polygon 格式，可能是一个列表 [x1,y1,x2,y2,...] 或多个多边形
        h, w = image_info['height'], image_info['width']
        # 转换为 RLE
        rles = maskUtils.frPyObjects(seg, h, w)
        if isinstance(rles, list):
            rle = maskUtils.merge(rles)
        else:
            rle = rles
        return maskUtils.decode(rle)
    else:
        raise ValueError("Unsupported segmentation format")

def main(json_path, output_dir):
    with open(json_path, 'r') as f:
        data = json.load(f)

    print("Top-level keys:", data.keys())

    # 检查 annotations 是否存在
    annotations = data.get('annotations', [])
    if not annotations:
        print("No 'annotations' found. Trying alternative keys...")
        # 如果 'annotations' 不存在，尝试其他可能的键名
        possible_keys = ['keypoints', 'segments', 'data']
        for key in possible_keys:
            if key in data:
                print(f"Found key '{key}' with length {len(data[key])}")
                annotations = data[key]
                break
        if not annotations:
            print("Could not find any annotation data. Exiting.")
            return

    # 构建 image_id -> (height, width) 映射
    images_info = {img['id']: img for img in data.get('images', [])}
    print(f"Loaded {len(images_info)} images info.")

    image_masks = {}
    skipped = 0

    for ann in annotations:
        image_id = ann.get('image_id')
        if image_id is None:
            print("Annotation missing 'image_id', skipping.")
            skipped += 1
            continue

        seg = ann.get('segmentation')
        if not seg:
            continue

        # 获取图像尺寸（polygon 需要）
        img_info = images_info.get(image_id)
        if img_info is None:
            print(f"Warning: image_id {image_id} not in images info, skipping.")
            skipped += 1
            continue

        try:
            mask = decode_segmentation(seg, img_info)
        except Exception as e:
            print(f"Decode error for ann {ann.get('id', 'unknown')}: {e}")
            skipped += 1
            continue

        h, w = mask.shape
        if image_id not in image_masks:
            image_masks[image_id] = np.zeros((h, w), dtype=np.uint8)

        # 合并（逻辑或）
        image_masks[image_id] = np.maximum(image_masks[image_id], mask)

    print(f"Processed {len(image_masks)} images, skipped {skipped} annotations.")

    os.makedirs(output_dir, exist_ok=True)
    for image_id, mask in image_masks.items():
        mask_img = (mask * 255).astype(np.uint8)
        out_path = os.path.join(output_dir, f"{image_id}.png")
        Image.fromarray(mask_img, mode='L').save(out_path)
        print(f"Saved {out_path}")

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--json', required=True, help='Path to COCO JSON annotation file.')
    parser.add_argument('--output', default='masks', help='Output directory for mask PNGs.')
    args = parser.parse_args()
    main(args.json, args.output)