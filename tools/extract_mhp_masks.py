import os
import numpy as np
from PIL import Image
from tqdm import tqdm

def obtain_ann_dict(img_root, ann_root):
    """返回 {图像名.jpg: [该图的多个 person 标注文件名, ...]}"""
    ann_dict = {}
    assert os.path.isdir(img_root), f'Path does not exist: {img_root}'
    assert os.path.isdir(ann_root), f'Path does not exist: {ann_root}'

    for add in os.listdir(img_root):
        ann_dict[add] = []

    for add in os.listdir(ann_root):
        ann_dict[add[:-10] + '.jpg'].append(add)

    return ann_dict


# ---------- 二值 mask 提取 ----------

def load_foreground(path):
    """
    读取单个 person 的标注文件，返回 bool 数组 (H, W)，True=前景。
    兼容三种情况：
      1) 调色板 P 图：索引 0=背景
      2) 单通道灰度：值 0=背景
      3) 多通道彩色可视化：纯黑 (0,0,0)=背景
    """
    img = Image.open(path)

    if img.mode == 'P':
        return np.array(img) > 0

    arr = np.array(img)
    if arr.ndim == 2:
        return arr > 0

    # RGB / RGBA：背景是纯黑
    rgb = arr[..., :3].astype(np.int32)
    return rgb.sum(axis=-1) > 0


def export_binary_masks(data_root, out_root, splits=('train', 'val'),
                        overwrite=False):
    """
    遍历 splits 中每个划分，把每张图所有 person 的前景合并成一张二值 mask
    (0/255)，保存到 {out_root}/{split}/parsing_binary/{name}.png
    """
    for split in splits:
        list_file = os.path.join(data_root, 'list', split + '.txt')
        if not os.path.exists(list_file):
            print(f'[跳过] 找不到列表文件 {list_file}')
            continue

        img_root = os.path.join(data_root, split, 'images')
        ann_root = os.path.join(data_root, split, 'parsing_annos')
        out_dir  = os.path.join(out_root, split, 'parsing_binary')
        os.makedirs(out_dir, exist_ok=True)

        ann_dict = obtain_ann_dict(img_root, ann_root)

        flist = [line.strip() for line in open(list_file).readlines()]

        for add in tqdm(flist, desc=f'Export {split}'):
            save_path = os.path.join(out_dir, add + '.png')
            if (not overwrite) and os.path.exists(save_path):
                continue

            ann_adds = sorted(ann_dict.get(add + '.jpg', []))
            if not ann_adds:
                print(f'[跳过] {add} 无标注')
                continue

            merged = None
            for ann_add in ann_adds:
                fg = load_foreground(os.path.join(ann_root, ann_add))
                if merged is None:
                    merged = np.zeros(fg.shape, dtype=bool)
                merged |= fg                     # 多 person 取并集

            out = merged.astype(np.uint8) * 255
            Image.fromarray(out, mode='L').save(save_path)


# ---------- 入口 ----------

if __name__ == '__main__':
    data_root = '/data/xiaoshuai/person/LV-MHP-v2-pose/raw/LV-MHP-v2/LV-MHP-v2'
    out_root  = '/data/xiaoshuai/person/LV-MHP-v2-pose/raw/LV-MHP-v2/LV-MHP-v2/binary_masks'

    export_binary_masks(data_root, out_root,
                        splits=('train', 'val'),
                        overwrite=False)