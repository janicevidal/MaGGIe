import json
import shutil
from pathlib import Path
from PIL import Image
import numpy as np
import cv2

def generate_full_body_mask(image_path: Path, json_path: Path, output_mask_path: Path):
    """
    从 JSON 中提取所有人像分割多边形，生成完整人像二值掩码。
    遍历所有 subject_annotation，将所有主体的 segment 合并为前景。
    """
    # 读取图像尺寸
    with Image.open(image_path) as img:
        img_w, img_h = img.size
    mask = np.zeros((img_h, img_w), dtype=np.uint8)

    with open(json_path, 'r', encoding='utf-8') as f:
        data = json.load(f)

    subjects = data.get('subject_annotation', [])
    print(f"  👤 该图像包含 {len(subjects)} 个主体")

    for idx, subject in enumerate(subjects):
        segments = subject.get('segments', [])
        if not segments:
            continue
        # 可选：只处理 primary 主体（取消下面注释）
        # if not subject.get('is_primary', False):
        #     continue

        for seg in segments:
            polygon = [(int(pt['x']), int(pt['y'])) for pt in seg['polygon']]
            if len(polygon) < 3:
                continue
            # OpenCV 需要形状 (点数, 1, 2)
            pts = np.array(polygon, dtype=np.int32).reshape((-1, 1, 2))
            cv2.fillPoly(mask, [pts], 1)

    # 保存标准二值掩码 (0/1)
    # Image.fromarray(mask).save(output_mask_path)

    # 生成可视化版本 (0/255)
    vis_mask = (mask > 0).astype(np.uint8) * 255
    vis_path = output_mask_path.parent / (output_mask_path.stem + ".png")
    Image.fromarray(vis_mask).save(vis_path)


def collect_images_and_masks(source_root, image_output_dir, mask_output_dir):
    """
    遍历 source_root 下的二级子目录，复制图像并生成掩码。
    """
    source_root = Path(source_root).resolve()
    image_output_dir = Path(image_output_dir).resolve()
    mask_output_dir = Path(mask_output_dir).resolve()

    if not source_root.exists():
        print(f"❌ 源路径不存在: {source_root}")
        return

    image_output_dir.mkdir(parents=True, exist_ok=True)
    mask_output_dir.mkdir(parents=True, exist_ok=True)

    copied_count = 0
    mask_count = 0

    for top_dir in source_root.iterdir():
        if not top_dir.is_dir():
            continue
        top_name = top_dir.name

        for sub_dir in top_dir.iterdir():
            if not sub_dir.is_dir():
                continue
            sub_name = sub_dir.name

            # 复制图像
            img_filename = f"main_{sub_name}.png"
            src_img = sub_dir / img_filename
            if not src_img.is_file():
                continue

            new_img_name = f"{top_name}_main_{sub_name}.png"
            dst_img = image_output_dir / new_img_name
            shutil.copy2(src_img, dst_img)
            copied_count += 1
            print(f"✅ 复制图像: {src_img} -> {dst_img}")

            # 处理 JSON
            json_filename = f"main_annos_{sub_name}.json"
            src_json = sub_dir / json_filename
            if not src_json.is_file():
                print(f"⚠️ JSON 不存在: {src_json}")
                continue

            mask_filename = new_img_name
            dst_mask = mask_output_dir / mask_filename
            try:
                generate_full_body_mask(src_img, src_json, dst_mask)
                mask_count += 1
                print(f"✅ 生成掩码: {dst_mask}")
            except Exception as e:
                print(f"❌ 生成掩码失败 {src_json}: {e}")

    print(f"\n🎯 完成！共复制 {copied_count} 个图像，生成 {mask_count} 个掩码（含可视副本）。")
    print(f"   图像目录: {image_output_dir}")
    print(f"   掩码目录: {mask_output_dir}")


if __name__ == "__main__":
    SOURCE = "/data/xiaoshuai/human_matting/dataset/FHIBE/fhibe.20250716.u.gT5_rFTA_downsampled_public/data/raw/fhibe_downsampled/"
    IMAGE_OUT = "/data/xiaoshuai/human_matting/dichotomous_dataset/fhibe/images"
    MASK_OUT = "/data/xiaoshuai/human_matting/dichotomous_dataset/fhibe/masks"

    collect_images_and_masks(SOURCE, IMAGE_OUT, MASK_OUT)