import json
import os
import numpy as np
import cv2
from pycocotools import mask as maskUtils
from tqdm import tqdm

# ================== 可配置参数 ==================
# JSON_PATH = "/data/xiaoshuai/clothing_segmentation/OpenDataLab___fashionpedia/raw/instances_attributes_val2020.json"   # COCO JSON 文件
# IMAGE_DIR = "/data/xiaoshuai/clothing_segmentation/OpenDataLab___fashionpedia/raw/test/"                              # 原始图片文件夹
# OUTPUT_ROOT = "/data/xiaoshuai/clothing_segmentation/OpenDataLab___fashionpedia/raw/output_main_clothing"                       # 输出根目录

JSON_PATH = "/data/xiaoshuai/clothing_segmentation/OpenDataLab___fashionpedia/raw/instances_attributes_train2020.json"   # COCO JSON 文件
IMAGE_DIR = "/data/xiaoshuai/clothing_segmentation/OpenDataLab___fashionpedia/raw/train/"                              # 原始图片文件夹
OUTPUT_ROOT = "/data/xiaoshuai/clothing_segmentation/OpenDataLab___fashionpedia/raw/train_output_main_clothing"          

# 设置你要提取的类别 ID（这里默认 0~12，即所有主要服装）
# 如果你只要上装（0,1,2,3,4,5,9,12?）、下装（6,7,8）、连体（10,11）
# 可按需修改这个列表，例如：
# TARGET_CATEGORY_IDS = [0,1,2,3,4,5,6,7,8,9,10,11,12]  # 全部主要服装
# 或只保留：衬衫、T恤、毛衣、开衫、裤子、短裤、裙子、连衣裙、连体裤（不含外套、背心、披肩）
# TARGET_CATEGORY_IDS = [0,1,2,3,6,7,8,10,11]
# TARGET_CATEGORY_IDS = list(range(0, 13))  # 默认 0~12
# TARGET_CATEGORY_IDS = [0,1,2,3,4,5,6,7,8,9,10,11,12,16,19,20,21,25,38]
TARGET_CATEGORY_IDS = [i for i in range(46) if i not in {13,14,15,17,18,20,21,22,23,24,26}]

# 是否保存每个实例的单独掩码
SAVE_INSTANCE_MASKS = False
# 是否保存合并掩码
SAVE_MERGED_MASK = True
# 是否保存可视化叠加图
SAVE_VISUALIZATION = False

# ================== 初始化输出目录 ==================
INSTANCE_DIR = os.path.join(OUTPUT_ROOT, "masks_instance")
MERGED_DIR = os.path.join(OUTPUT_ROOT, "masks_merged")
VIS_DIR = os.path.join(OUTPUT_ROOT, "visualization")
for d in [INSTANCE_DIR, MERGED_DIR, VIS_DIR]:
    os.makedirs(d, exist_ok=True)

# ================== 加载 JSON ==================
print("Loading JSON...")
with open(JSON_PATH, 'r') as f:
    coco_data = json.load(f)

image_info_map = {img['id']: img for img in coco_data['images']}
cat_map = {cat['id']: cat['name'] for cat in coco_data['categories']}

# 过滤出目标类别的标注
target_anns = [ann for ann in coco_data['annotations'] 
               if ann['category_id'] in TARGET_CATEGORY_IDS]
print(f"Total target annotations: {len(target_anns)}")

# 按图像分组
anns_by_image = {}
for ann in target_anns:
    img_id = ann['image_id']
    anns_by_image.setdefault(img_id, []).append(ann)
print(f"Number of images with target clothing: {len(anns_by_image)}")

# ================== 生成颜色 ==================
COLORS = {}
for idx, cat_id in enumerate(TARGET_CATEGORY_IDS):
    hue = (idx * 180 / len(TARGET_CATEGORY_IDS)) % 180
    color = cv2.cvtColor(np.uint8([[[hue, 255, 255]]]), cv2.COLOR_HSV2BGR)[0][0]
    COLORS[cat_id] = tuple(map(int, color))

# ================== 处理每一张图像 ==================
for img_id, ann_list in tqdm(anns_by_image.items(), desc="Processing"):
    img_info = image_info_map.get(img_id)
    if not img_info:
        continue

    # ----- 关键修改：使用原始图像文件名（不含扩展名）作为输出基础名 -----
    orig_filename = img_info['file_name']           # 例如 "0017a7bb...jpg"
    base_name = os.path.splitext(orig_filename)[0]  # 去掉扩展名

    img_path = os.path.join(IMAGE_DIR, orig_filename)
    if not os.path.exists(img_path):
        print(f"Warning: {img_path} not found, skipping")
        continue

    # 读取原图
    img_bgr = cv2.imread(img_path)
    if img_bgr is None:
        continue
    h, w = img_bgr.shape[:2]

    # 合并掩码
    merged_mask = np.zeros((h, w), dtype=np.uint8)
    vis_img = img_bgr.copy()

    for ann in ann_list:
        ann_id = ann['id']
        cat_id = ann['category_id']
        cat_name = cat_map.get(cat_id, f"class_{cat_id}")
        seg = ann['segmentation']

        # ---------- 解码掩码 ----------
        if isinstance(seg, list):  # 多边形
            rles = maskUtils.frPyObjects(seg, h, w)
            if isinstance(rles, list):
                rle = maskUtils.merge(rles)
            else:
                rle = rles
            mask = maskUtils.decode(rle)
        elif isinstance(seg, dict) and 'counts' in seg:  # RLE
            rle = seg
            if 'size' not in rle:
                rle['size'] = [h, w]
            mask = maskUtils.decode(rle)
        else:
            continue
        mask = (mask > 0).astype(np.uint8)
        if mask.sum() == 0:
            continue

        # ---------- 保存实例掩码（使用图像名 + ann_id 区分） ----------
        if SAVE_INSTANCE_MASKS:
            mask_file = os.path.join(INSTANCE_DIR, f"{base_name}_ann{ann_id}.png")
            cv2.imwrite(mask_file, mask * 255)

        # ---------- 更新合并掩码 ----------
        merged_mask = np.logical_or(merged_mask, mask).astype(np.uint8)

        # ---------- 绘制可视化 ----------
        if SAVE_VISUALIZATION:
            color = COLORS.get(cat_id, (0, 255, 0))
            contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            cv2.drawContours(vis_img, contours, -1, color, thickness=cv2.FILLED)
            cv2.drawContours(vis_img, contours, -1, (255, 255, 255), 1)

            moments = cv2.moments(mask)
            if moments['m00'] != 0:
                cx = int(moments['m10'] / moments['m00'])
                cy = int(moments['m01'] / moments['m00'])
                cv2.putText(vis_img, cat_name, (cx, cy), cv2.FONT_HERSHEY_SIMPLEX,
                            0.5, (0, 0, 0), 2, cv2.LINE_AA)
                cv2.putText(vis_img, cat_name, (cx, cy), cv2.FONT_HERSHEY_SIMPLEX,
                            0.5, (255, 255, 255), 1, cv2.LINE_AA)

    # ---------- 保存合并掩码 ----------
    if SAVE_MERGED_MASK:
        merged_path = os.path.join(MERGED_DIR, f"{base_name}.png")
        cv2.imwrite(merged_path, merged_mask * 255)

    # ---------- 保存可视化图 ----------
    if SAVE_VISUALIZATION:
        vis_path = os.path.join(VIS_DIR, f"{base_name}.jpg")
        cv2.imwrite(vis_path, vis_img)

print("All done!")