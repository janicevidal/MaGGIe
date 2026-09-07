import json
import shutil
from pathlib import Path
from PIL import Image, ImageDraw
import numpy as np

# def generate_clothes_mask(image_path: Path, json_path: Path, output_mask_path: Path):
#     """
#     从 JSON 文件中提取上衣和下衣分割多边形，生成单通道掩码并保存。
#     掩码值：0=背景，1=Upper body clothes，2=Lower body clothes。
#     """
#     with open(json_path, 'r', encoding='utf-8') as f:
#         data = json.load(f)

#     # img_w = int(data['image_annotation']['image_width'])
#     # img_h = int(data['image_annotation']['image_height'])
#     # print(f"mask尺寸: {img_w} x {img_h}")
    
#     with Image.open(image_path) as img:
#         img_w, img_h = img.size 

#     mask = np.zeros((img_h, img_w), dtype=np.uint8)

#     segments = data['subject_annotation'][0].get('segments', [])
#     for seg in segments:
#         class_name = seg['class_name']
#         if 'Upper body clothes' in class_name:
#             value = 1
#         elif 'Lower body clothes' in class_name:
#             value = 2
#         else:
#             continue

#         polygon = [(int(pt['x']), int(pt['y'])) for pt in seg['polygon']]
#         pil_mask = Image.fromarray(mask)
#         draw = ImageDraw.Draw(pil_mask)
#         draw.polygon(polygon, fill=value)
#         mask = np.array(pil_mask)
        
#         mask = (mask > 0).astype(np.uint8) * 255

#     Image.fromarray(mask).save(output_mask_path)

def generate_clothes_mask(image_path: Path, json_path: Path, output_mask_path: Path):
    """
    从 JSON 文件中提取所有主体的上衣和下衣分割多边形，生成二值掩码。
    掩码值：0=背景，255=衣服区域。
    """
    with open(json_path, 'r', encoding='utf-8') as f:
        data = json.load(f)

    with Image.open(image_path) as img:
        img_w, img_h = img.size

    mask = np.zeros((img_h, img_w), dtype=np.uint8)

    # 遍历所有主体（包括 primary 和 secondary）
    for subject in data.get('subject_annotation', []):
        segments = subject.get('segments', [])
        for seg in segments:
            class_name = seg['class_name']
            # 只处理上衣和下衣
            if 'Upper body clothes' in class_name or 'Lower body clothes' in class_name:
                # 统一填充为255（二值）
                value = 255
            else:
                continue

            polygon = [(int(pt['x']), int(pt['y'])) for pt in seg['polygon']]
            pil_mask = Image.fromarray(mask)
            draw = ImageDraw.Draw(pil_mask)
            draw.polygon(polygon, fill=value)
            mask = np.array(pil_mask)
            # 保持二值化（避免绘制重叠时出现其他数值）
            mask = (mask > 0).astype(np.uint8) * 255

    Image.fromarray(mask).save(output_mask_path)


def collect_images_and_masks(source_root, image_output_dir, mask_output_dir):
    """
    遍历 source_root 下的二级子目录（顶层UUID/子目录），
    复制 main_{子目录名}.png 到 image_output_dir（文件名添加顶层前缀），
    同时生成对应的衣服分割掩码到 mask_output_dir。
    图像与掩码文件名完全相同（除目录不同）。
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

            # ----- 处理图像 -----
            img_filename = f"main_{sub_name}.png"
            src_img = sub_dir / img_filename
            if not src_img.is_file():
                continue   # 跳过无图像的目录

            new_img_name = f"{top_name}_main_{sub_name}.png"
            dst_img = image_output_dir / new_img_name
            shutil.copy2(src_img, dst_img)
            copied_count += 1
            print(f"✅ 复制图像: {src_img} -> {dst_img}")

            # ----- 处理 JSON 标注 -----
            json_filename = f"main_annos_{sub_name}.json"
            src_json = sub_dir / json_filename
            if not src_json.is_file():
                print(f"⚠️ JSON 文件不存在: {src_json}")
                continue

            # 掩码文件名与图像文件名完全相同（不含 _mask）
            mask_filename = new_img_name   # 同一个名字
            dst_mask = mask_output_dir / mask_filename
            try:
                generate_clothes_mask(src_img, src_json, dst_mask)
                mask_count += 1
                print(f"✅ 生成掩码: {dst_mask}")
            except Exception as e:
                print(f"❌ 生成掩码失败 {src_json}: {e}")

    print(f"\n🎯 完成！共复制 {copied_count} 个图像，生成 {mask_count} 个掩码。")
    print(f"   图像目录: {image_output_dir}")
    print(f"   掩码目录: {mask_output_dir}")


if __name__ == "__main__":
    SOURCE = "/data/xiaoshuai/human_matting/dataset/FHIBE/fhibe.20250716.u.gT5_rFTA_downsampled_public/data/raw/fhibe_downsampled/"
    IMAGE_OUT = "/data/xiaoshuai/clothing_segmentation/FHIBE/images"
    MASK_OUT = "/data/xiaoshuai/clothing_segmentation/FHIBE/masks_all"

    collect_images_and_masks(SOURCE, IMAGE_OUT, MASK_OUT)