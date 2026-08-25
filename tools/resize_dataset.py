import os
import argparse
from PIL import Image

def get_save_kwargs(filepath):
    """根据文件扩展名返回保存参数，使质量最高"""
    ext = os.path.splitext(filepath)[1].lower()
    if ext in ('.jpg', '.jpeg', '.JPG', '.JPEG'):
        return {'quality': 100, 'subsampling': 0}  # 最高质量，禁用色度子采样
    elif ext == '.png':
        # PNG是无损，但可设置压缩级别为0（最快，文件最大）
        return {'compress_level': 0}
    else:
        # 其他格式默认
        return {}
    
def resize_dataset(image_dir, mask_dir, output_root, target_size=768,
                   img_extensions=('.jpg', '.jpeg', '.png', '.JPG', '.JPEG', '.PNG'),
                   mask_extension='.png', mask_suffix='',
                   resample_method='LANCZOS'):
    """
    缩放图像和对应的同名 mask（matting），使长边不超过 target_size。
    图像和 mask 位于不同目录，输出也分开到 output_root/images/ 和 output_root/masks/。

    Args:
        image_dir (str): 图像输入目录
        mask_dir (str): mask 输入目录
        output_root (str): 输出根目录，将在其下创建 images/ 和 masks/ 子目录
        target_size (int): 长边目标像素数，默认 768
        img_extensions (tuple): 图像文件扩展名（不区分大小写）
        mask_extension (str): mask 文件的扩展名（例如 '.png'）
        mask_suffix (str): mask 文件名后缀（如果 mask 与图像不完全同名，例如 '_matte'）
        resample_method (str): 重采样方法，可选 'LANCZOS', 'BICUBIC', 'BILINEAR' 等，
                                默认 'LANCZOS'，图像和 mask 均使用此方法。
    """
    resample_map = {
        'NEAREST': Image.Resampling.NEAREST,
        'BILINEAR': Image.Resampling.BILINEAR,
        'BICUBIC': Image.Resampling.BICUBIC,
        'LANCZOS': Image.Resampling.LANCZOS,
    }
    resample = resample_map.get(resample_method.upper(), Image.Resampling.LANCZOS)

    # 创建输出子目录
    img_out_dir = os.path.join(output_root, 'images')
    mask_out_dir = os.path.join(output_root, 'alphas')
    os.makedirs(img_out_dir, exist_ok=True)
    os.makedirs(mask_out_dir, exist_ok=True)

    # 收集所有图像文件
    image_files = []
    for f in os.listdir(image_dir):
        if f.lower().endswith(img_extensions):
            image_files.append(f)

    if not image_files:
        print(f"警告：在 {image_dir} 中未找到任何图像文件（扩展名 {img_extensions}）")
        return

    for img_filename in image_files:
        name, ext = os.path.splitext(img_filename)
        img_path = os.path.join(image_dir, img_filename)

        # 在 mask_dir 中构造 mask 文件名
        mask_candidates = [
            os.path.join(mask_dir, name + mask_suffix + mask_extension),
            os.path.join(mask_dir, name + mask_extension)  # 若带后缀没找到，尝试无后缀
        ]
        mask_path = None
        for candidate in mask_candidates:
            if os.path.exists(candidate):
                mask_path = candidate
                break

        if mask_path is None:
            print(f"警告：未找到 {img_filename} 对应的 mask（在 {mask_dir} 中），跳过")
            continue

        try:
            img = Image.open(img_path)
            mask = Image.open(mask_path)

            w, h = img.size
            w, h = img.size
            short_side = min(w, h)
            
            if short_side > target_size:
                scale = target_size / short_side
                new_size = (int(round(w * scale)), int(round(h * scale)))
            else:
                new_size = (w, h)
                
            if new_size != (w, h):
                img_resized = img.resize(new_size, resample)
                # mask_resized = mask.resize(new_size, resample)
                mask_resized = mask.resize(new_size, Image.Resampling.NEAREST)
            else:
                img_resized = img
                mask_resized = mask

            # 保存图像到 images/ 子目录
            out_img_path = os.path.join(img_out_dir, img_filename)
            img_resized.save(out_img_path, **get_save_kwargs(out_img_path))

            # 保存 mask 到 alphas/ 子目录，保留其原始文件名
            mask_basename = os.path.basename(mask_path).split('.')[0] + ".png"
            out_mask_path = os.path.join(mask_out_dir, mask_basename)
            mask_resized.save(out_mask_path, **get_save_kwargs(out_mask_path))

            print(f"已处理：{img_filename} 和 {mask_basename} -> {new_size}，插值：{resample_method}")

        except Exception as e:
            print(f"处理 {img_filename} 时出错：{e}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="缩放图像和 matting mask（分目录输入，分目录输出），使长边不超过指定尺寸")
    parser.add_argument("--image_dir", required=True, help="图像输入目录")
    parser.add_argument("--mask_dir", required=True, help="mask 输入目录")
    parser.add_argument("--output_root", required=True, help="输出根目录，其下将自动创建 images/ 和 masks/ 子目录")
    parser.add_argument("--target_size", type=int, default=640, help="长边目标像素数，默认 640")
    parser.add_argument("--mask_ext", default=".png", help="mask 文件扩展名，如 .png")
    parser.add_argument("--mask_suffix", default="", help="mask 文件名后缀，例如 '_matte'，默认空")
    parser.add_argument("--resample", default="LANCZOS", 
                        choices=['NEAREST', 'BILINEAR', 'BICUBIC', 'LANCZOS'],
                        help="重采样算法，默认 LANCZOS（推荐用于 matting）")
    args = parser.parse_args()

    resize_dataset(
        image_dir=args.image_dir,
        mask_dir=args.mask_dir,
        output_root=args.output_root,
        target_size=args.target_size,
        mask_extension=args.mask_ext,
        mask_suffix=args.mask_suffix,
        resample_method=args.resample
    )