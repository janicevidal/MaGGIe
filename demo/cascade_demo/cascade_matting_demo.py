#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""级联人像 matting 推理 demo：训练模型 → 粗 mask → 级联 ONNX 精修。

一条命令跑完两个阶段，并把两段结果分别落盘：

  阶段 1（训练模型）：复用 demo.py 的推理流程，用 config + 权重在原图上得到
      alpha，保存 alpha 和透明背景合成图（原始模型输出）。
  阶段 2（级联 ONNX）：把阶段 1 的 alpha 当成粗分割 mask，与原图一起喂给
      portrait_matting_v5.1.onnx（image + mask 双输入），再做一次精细
      matting，保存级联 alpha 和级联合成图。

两个阶段都在“原始分辨率”的同一张图上工作：阶段 1 的 alpha 经过反 padding /
反 resize 回到原图尺寸，阶段 2 自己按参考实现做 letterbox（512 居中填充），
因此唯一在两者之间传递的东西就是粗 mask。

用法::

    python demo/cascade_demo/cascade_matting_demo.py \
        --config configs/Matting/focal_matting_3gpu_host50.yaml \
        --weights /path/to/model_iter110500.pth \
        --input-dir demo/examples \
        --output-dir output/cascade_demo \
        --onnx-model demo/cascade_demo/portrait_matting_v5.1.onnx \
        --save-visualization

输出目录::

    model_alpha/        训练模型 alpha（原始分辨率）
    model_composite/    训练模型输出的透明背景合成图
    coarse_mask/        实际送给级联 ONNX 的粗 mask（便于排查前后差异）
    cascade_alpha/      级联 ONNX alpha
    cascade_composite/  级联 ONNX 输出的合成图
    visualization/      [输入 | 模型 alpha | 级联 alpha | 模型合成 | 级联合成]
"""

from __future__ import annotations

import argparse
import logging
import sys
import time
from pathlib import Path

import cv2
import numpy as np
from PIL import Image
import torch
from torch.utils.data import DataLoader

SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parents[1]
for _path in (str(REPO_ROOT), str(SCRIPT_DIR)):
    if _path not in sys.path:
        sys.path.insert(0, _path)

try:  # 同目录的级联 ONNX 参考实现（preprocess/postprocess 与线上一致）
    import portrait_matting_ort as cascade_ort
except (ImportError, SystemExit) as exc:  # onnxruntime 缺失时给出可读的错误
    raise SystemExit(
        "无法导入 demo/cascade_demo/portrait_matting_ort.py：{}\n"
        "请先安装: pip install onnxruntime opencv-python numpy".format(exc)
    ) from None

from demo import (  # noqa: E402  demo.py 中的训练模型推理链路
    CHROMA_GREEN_RGB,
    FixedSizeImageFolderDataset,
    ImageFolderDataset,
    load_model,
    resolve_device,
    reverse_prediction_transform,
)
from maggie.utils import CONFIG  # noqa: E402
from maggie.utils.postprocessing import postprocess  # noqa: E402


def parse_args(argv=None):
    parser = argparse.ArgumentParser(
        description="级联人像 matting demo：训练模型输出粗 mask，再送级联 ONNX 精修"
    )
    # ---- 阶段 1：训练模型 ----
    parser.add_argument("--config", required=True, help="训练用的实验 YAML config")
    parser.add_argument("--input-dir", required=True, help="输入图像文件夹")
    parser.add_argument("--output-dir", required=True, help="结果保存目录")
    parser.add_argument("--weights", default=None, help="覆盖 config 里的 model.weights")
    parser.add_argument(
        "--short-size", type=int, default=None,
        help="覆盖 dataset.test.short_size",
    )
    parser.add_argument(
        "--resize-mode", choices=("short-side", "fixed"), default="short-side",
        help="与 demo.py 一致：short-side 保持训练时预处理，fixed 居中填充成方图",
    )
    parser.add_argument("--device", default="auto", help="训练模型推理设备，如 cuda:0 / cpu")
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument(
        "--non-recursive", action="store_true", help="只读取 input-dir 直接子目录下的图像",
    )
    postprocess_group = parser.add_mutually_exclusive_group()
    postprocess_group.add_argument("--postprocessing", action="store_true")
    postprocess_group.add_argument("--no-postprocessing", action="store_true")

    # ---- 阶段 2：级联 ONNX ----
    parser.add_argument(
        "--onnx-model", default=str(SCRIPT_DIR / "portrait_matting_v5.1.onnx"),
        help="级联 ONNX 模型路径（image + mask 双输入）",
    )
    parser.add_argument(
        "--onnx-device", choices=("auto", "cpu", "cuda"), default="auto",
        help="ONNX Runtime provider；auto 表示能用 CUDA 就用 CUDA",
    )
    parser.add_argument(
        "--onnx-size", type=int, default=0,
        help="强制级联模型输入边长，默认从模型 shape 读取，读不到用 512",
    )
    parser.add_argument(
        "--mask-mode", choices=("soft", "binary"), default="soft",
        help="送给级联模型的粗 mask：soft 直接用模型 alpha，binary 按阈值二值化",
    )
    parser.add_argument(
        "--mask-threshold", type=float, default=0.5,
        help="--mask-mode binary 时的阈值",
    )
    parser.add_argument(
        "--mask-dilate", type=int, default=0,
        help="对粗 mask 做 N 像素膨胀（粗分割 mask 通常比真实前景略大）",
    )

    # ---- 输出 / 调试 ----
    parser.add_argument(
        "--save-visualization", action="store_true",
        help="额外保存对比图 visualization/",
    )
    parser.add_argument("--limit", type=int, default=0, help="只处理前 N 张，0 表示全部")
    return parser.parse_args(argv)


def build_session(model_path, onnx_device):
    """创建 ONNX Runtime session，并解析 image/mask/output 三个张量名。"""
    import onnxruntime as ort

    model_path = Path(model_path).expanduser().resolve()
    if not model_path.is_file():
        raise FileNotFoundError("级联 ONNX 模型不存在: {}".format(model_path))

    available = ort.get_available_providers()
    providers = []
    if onnx_device in ("auto", "cuda") and "CUDAExecutionProvider" in available:
        providers.append("CUDAExecutionProvider")
    providers.append("CPUExecutionProvider")

    session = ort.InferenceSession(str(model_path), providers=providers)
    image_name, mask_name, out_name = cascade_ort.resolve_io_names(session)
    logging.info(
        "级联模型: %s | providers=%s | inputs=%s | output=%s",
        model_path.name,
        session.get_providers(),
        [(i.name, i.shape) for i in session.get_inputs()],
        out_name,
    )
    return session, (image_name, mask_name, out_name)


@torch.inference_mode()
def predict_with_trained_model(model, image_tensor, device):
    """跑训练模型，返回 padding 前（batch 内）的预测张量列表。"""
    batch = {"image": image_tensor.to(device, non_blocking=True)}
    output = model(batch, mem_feat=None)
    alpha = output.get("refined_masks", output.get("alpha_pred"))
    if alpha is None:
        raise KeyError("模型输出里既没有 refined_masks 也没有 alpha_pred")
    return alpha


def alpha_to_map(alpha_tensor, transform_info, do_postprocessing):
    """反 padding / 反 resize 到原图尺寸，返回 (H, W) float32 的 alpha。"""
    alpha = reverse_prediction_transform(alpha_tensor, transform_info).cpu().numpy()

    alpha[alpha <= 1.0 / 255.0] = 0.0
    alpha[alpha >= 254.0 / 255.0] = 1.0
    if do_postprocessing:
        alpha = postprocess(alpha)

    # 展平 frame / instance 维度：无 trimap 的图像模型每张图通常只给一张 matte
    alpha = alpha[0].reshape(-1, *alpha.shape[-2:])
    if alpha.shape[0] != 1:
        logging.warning("模型返回了 %d 个 alpha，只保留第一个", alpha.shape[0])
    return np.clip(alpha[0], 0.0, 1.0).astype(np.float32)


def to_coarse_mask(alpha, mask_mode, threshold, dilate_pixels):
    """把训练模型的 alpha 转成级联 ONNX 需要的 0~255 粗 mask。"""
    if mask_mode == "binary":
        mask = (alpha >= threshold).astype(np.uint8) * 255
    else:
        mask = np.rint(np.clip(alpha, 0.0, 1.0) * 255.0).astype(np.uint8)
    if dilate_pixels > 0:
        kernel = np.ones((2 * dilate_pixels + 1, 2 * dilate_pixels + 1), np.uint8)
        mask = cv2.dilate(mask, kernel)
    return mask


def predict_with_cascade(session, io_names, image_bgr, mask_u8, input_size):
    """image + mask → 级联 ONNX → 原始分辨率 uint8 alpha。"""
    image_name, mask_name, out_name = io_names
    image_nchw, mask_nchw, offsets, orig_hw = cascade_ort.preprocess(
        image_bgr, mask_u8, input_size
    )
    outputs = session.run([out_name], {image_name: image_nchw, mask_name: mask_nchw})
    return cascade_ort.postprocess(outputs[0], offsets, orig_hw)


def read_rgb(image_path):
    with Image.open(image_path) as image:
        return np.array(image.convert("RGB"), dtype=np.uint8)


def save_rgba_composite(image_rgb, alpha_u8, path):
    """各阶段统一的透明背景合成图：原始 RGB + 预测 alpha 通道。"""
    if alpha_u8.shape[:2] != image_rgb.shape[:2]:
        raise ValueError(
            "alpha {} 与原图 {} 尺寸不一致".format(alpha_u8.shape[:2], image_rgb.shape[:2])
        )
    path.parent.mkdir(parents=True, exist_ok=True)
    rgba = np.dstack((image_rgb, alpha_u8))
    Image.fromarray(rgba, mode="RGBA").save(path, format="PNG")


def save_alpha(alpha_u8, path):
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(alpha_u8, mode="L").save(path)


def green_composite(image_rgb, alpha_u8):
    """把 alpha 压在绿幕上，用作对比图的合成预览（与 demo.py 一致）。"""
    alpha = alpha_u8[..., None].astype(np.float32) / 255.0
    foreground = image_rgb.astype(np.float32) / 255.0 * alpha
    green = CHROMA_GREEN_RGB / 255.0
    composite = foreground + (1.0 - alpha) * green
    return (composite.clip(0.0, 1.0) * 255.0).astype(np.uint8)


def save_visualization(image_rgb, model_alpha, cascade_alpha, path):
    """[输入 | 模型 alpha | 级联 alpha | 模型合成 | 级联合成]。"""
    def to_rgb(alpha):
        return np.repeat(alpha[..., None], 3, axis=-1)

    panel = np.concatenate(
        [
            image_rgb,
            to_rgb(model_alpha),
            to_rgb(cascade_alpha),
            green_composite(image_rgb, model_alpha),
            green_composite(image_rgb, cascade_alpha),
        ],
        axis=1,
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(panel, mode="RGB").save(path)


def relative_output_path(relative_path):
    relative_path = Path(relative_path).with_suffix(".png")
    return relative_path


def run(args):
    device = resolve_device(args.device)
    output_dir = Path(args.output_dir).expanduser().resolve()

    cfg = CONFIG.clone()
    cfg.merge_from_file(args.config)
    if args.weights is not None:
        cfg.model.weights = args.weights
    short_size = args.short_size if args.short_size is not None else cfg.dataset.test.short_size
    if short_size <= 0:
        raise ValueError("short-size 必须是正数")

    if args.postprocessing:
        do_postprocessing = True
    elif args.no_postprocessing:
        do_postprocessing = False
    else:
        do_postprocessing = cfg.test.postprocessing

    dataset_class = (
        FixedSizeImageFolderDataset if args.resize_mode == "fixed" else ImageFolderDataset
    )
    dataset = dataset_class(
        args.input_dir,
        short_size=short_size,
        recursive=not args.non_recursive,
        exclude_dir=output_dir,
    )
    data_loader = DataLoader(
        dataset,
        batch_size=1,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=device.type == "cuda",
    )

    model = load_model(cfg, device)
    session, io_names = build_session(args.onnx_model, args.onnx_device)
    onnx_size = args.onnx_size or cascade_ort.infer_input_size(
        session, io_names[0], cascade_ort.DEFAULT_INPUT_SIZE
    )

    total = len(dataset) if not args.limit else min(args.limit, len(dataset))
    logging.info(
        "共 %d 张图，处理 %d 张；训练模型预处理=%s(short_size=%d)，级联输入 %dx%d，mask_mode=%s",
        len(dataset), total, args.resize_mode, short_size, onnx_size, onnx_size, args.mask_mode,
    )

    for index, sample in enumerate(data_loader, start=1):
        if index > total:
            break
        image_path = sample.pop("image_path")[0]
        relative_path = relative_output_path(sample.pop("relative_path")[0])
        transform_info = sample.pop("transform_info")

        start = time.time()
        alpha_tensor = predict_with_trained_model(model, sample["image"], device)
        model_alpha = alpha_to_map(alpha_tensor, transform_info, do_postprocessing)
        model_alpha_u8 = np.rint(model_alpha * 255.0).astype(np.uint8)
        model_time = time.time() - start

        image_bgr = cv2.imread(image_path, cv2.IMREAD_COLOR)
        if image_bgr is None:
            raise FileNotFoundError("无法读取图像: {}".format(image_path))
        image_rgb = read_rgb(image_path)
        if image_rgb.shape[:2] != model_alpha.shape[:2]:
            raise ValueError(
                "训练模型 alpha {} 与原图 {} 尺寸不一致".format(
                    model_alpha.shape[:2], image_rgb.shape[:2]
                )
            )

        coarse_mask = to_coarse_mask(
            model_alpha, args.mask_mode, args.mask_threshold, args.mask_dilate
        )
        start = time.time()
        cascade_alpha_u8 = predict_with_cascade(
            session, io_names, image_bgr, coarse_mask, onnx_size
        )
        cascade_time = time.time() - start
        
        high = 0.9 * 255
        low= 0.23 * 255
        model_alpha_u8 = np.clip((model_alpha_u8 - low) / (high - low + 1e-8) * 255.0, 0, 255).astype(np.uint8)

        save_alpha(model_alpha_u8, output_dir / "model_alpha" / relative_path)
        save_rgba_composite(image_rgb, model_alpha_u8, output_dir / "model_composite" / relative_path)
        save_alpha(coarse_mask, output_dir / "coarse_mask" / relative_path)
        save_alpha(cascade_alpha_u8, output_dir / "cascade_alpha" / relative_path)
        save_rgba_composite(
            image_rgb, cascade_alpha_u8, output_dir / "cascade_composite" / relative_path
        )
        if args.save_visualization:
            save_visualization(
                image_rgb, model_alpha_u8, cascade_alpha_u8,
                output_dir / "visualization" / relative_path,
            )

        logging.info(
            "[%d/%d] %s | 训练模型 %.2fs | 级联 %.2fs | alpha 均值 %.3f → %.3f",
            index, total, relative_path, model_time, cascade_time,
            float(model_alpha.mean()), float(cascade_alpha_u8.mean() / 255.0),
        )


def main(argv=None):
    args = parse_args(argv)
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s"
    )
    output_dir = Path(args.output_dir).expanduser().resolve()
    run(args)
    logging.info("训练模型 alpha / 合成图: %s , %s", output_dir / "model_alpha", output_dir / "model_composite")
    logging.info("级联 ONNX alpha / 合成图: %s , %s", output_dir / "cascade_alpha", output_dir / "cascade_composite")
    logging.info("送给级联模型的粗 mask: %s", output_dir / "coarse_mask")
    if args.save_visualization:
        logging.info("对比图: %s", output_dir / "visualization")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
