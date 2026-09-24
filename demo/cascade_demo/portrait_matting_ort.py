#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
portrait_matting_v5.1 ONNX Runtime 验证脚本

对齐 YXSegMatting/source/seg_matting.cpp 中 matting 分支：
  - 输入名: image / mask
  - 布局: NCHW (CAFFE)
  - 默认输入边长: 512（letterbox 居中填充）
  - 图像归一化: (x - mean) * scale，RGB
  - mask: 0~255 粗分割 → letterbox 后 * 0.00392

用法（与模型同目录或指定路径）:
  pip install onnxruntime opencv-python numpy
  python portrait_matting_ort.py --image person.jpg --model portrait_matting_v5.1.onnx
  python portrait_matting_ort.py --image person.jpg --mask coarse_mask.png --out out_rgba.png

说明:
  - --mask 可选。不传时用整图 1.0 粗 mask（效果通常偏弱，建议提供人物粗分割 mask）。
  - 输出默认保存: alpha 灰度图 + 前景 RGBA 合成图。
"""

from __future__ import annotations

import argparse
import os
import sys
from typing import Optional, Tuple

import cv2
import numpy as np

try:
    import onnxruntime as ort
except ImportError as exc:
    raise SystemExit(
        "缺少 onnxruntime，请先安装: pip install onnxruntime opencv-python numpy"
    ) from exc


# 与 preprocess_matting_input / ImageNet 风格一致
MEAN_RGB = np.array([123.675, 116.28, 103.53], dtype=np.float32)
SCALE_RGB = np.array([0.017125, 0.0175, 0.01743], dtype=np.float32)
MASK_SCALE = 0.00392  # ≈ 1/255
DEFAULT_INPUT_SIZE = 512


def letterbox_rgb(
    rgb: np.ndarray,
    size: int,
) -> Tuple[np.ndarray, int, int, int, int]:
    """等比缩放到最长边=size，再居中 pad 成 size×size。返回 pad 后图与四边偏移。"""
    h, w = rgb.shape[:2]
    scale = max(w, h) / float(size)
    new_w = max(1, int(w / scale))
    new_h = max(1, int(h / scale))
    resized = cv2.resize(rgb, (new_w, new_h), interpolation=cv2.INTER_LINEAR)

    left = (size - new_w) >> 1
    right = size - new_w - left
    top = (size - new_h) >> 1
    bottom = size - new_h - top

    canvas = np.zeros((size, size, 3), dtype=np.uint8)
    canvas[top : top + new_h, left : left + new_w] = resized
    return canvas, left, right, top, bottom


def letterbox_mask(
    mask: np.ndarray,
    size: int,
    left: int,
    top: int,
    new_w: int,
    new_h: int,
) -> np.ndarray:
    """mask 与 image 使用同一套 letterbox 几何。"""
    resized = cv2.resize(mask, (new_w, new_h), interpolation=cv2.INTER_NEAREST)
    canvas = np.zeros((size, size), dtype=np.float32)
    canvas[top : top + new_h, left : left + new_w] = resized.astype(np.float32)
    return canvas


def preprocess(
    bgr: np.ndarray,
    mask_u8: Optional[np.ndarray],
    input_size: int,
) -> Tuple[np.ndarray, np.ndarray, Tuple[int, int, int, int], Tuple[int, int]]:
    """
    返回:
      image_nchw: (1,3,S,S) float32
      mask_nchw:  (1,1,S,S) float32
      offsets: (left, right, top, bottom)
      orig_hw: (h, w)
    """
    h0, w0 = bgr.shape[:2]
    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)

    canvas, left, right, top, bottom = letterbox_rgb(rgb, input_size)
    new_w = input_size - left - right
    new_h = input_size - top - bottom

    # NCHW + ImageNet-ish normalize
    img_f = canvas.astype(np.float32)
    img_f = (img_f - MEAN_RGB) * SCALE_RGB
    image_nchw = np.transpose(img_f, (2, 0, 1))[None, ...].astype(np.float32)

    if mask_u8 is None:
        mask_src = np.full((h0, w0), 255, dtype=np.uint8)
    else:
        if mask_u8.ndim == 3:
            mask_u8 = cv2.cvtColor(mask_u8, cv2.COLOR_BGR2GRAY)
        if mask_u8.shape[:2] != (h0, w0):
            mask_u8 = cv2.resize(mask_u8, (w0, h0), interpolation=cv2.INTER_NEAREST)
        mask_src = mask_u8

    mask_lb = letterbox_mask(mask_src, input_size, left, top, new_w, new_h)
    mask_nchw = (mask_lb * MASK_SCALE)[None, None, ...].astype(np.float32)

    return image_nchw, mask_nchw, (left, right, top, bottom), (h0, w0)


def postprocess(
    alpha_chw: np.ndarray,
    offsets: Tuple[int, int, int, int],
    orig_hw: Tuple[int, int],
) -> np.ndarray:
    """裁掉 letterbox，缩放到原图，得到 uint8 alpha (H,W)。"""
    left, right, top, bottom = offsets
    h0, w0 = orig_hw
    if alpha_chw.ndim == 4:
        alpha = alpha_chw[0, 0]
    elif alpha_chw.ndim == 3:
        alpha = alpha_chw[0]
    else:
        alpha = alpha_chw

    h, w = alpha.shape[:2]
    crop = alpha[top : h - bottom, left : w - right]
    crop = np.clip(crop, 0.0, 1.0)
    alpha_u8 = (crop * 255.0).astype(np.uint8)
    if alpha_u8.shape[:2] != (h0, w0):
        alpha_u8 = cv2.resize(alpha_u8, (w0, h0), interpolation=cv2.INTER_LINEAR)
    return alpha_u8


def compose_rgba(bgr: np.ndarray, alpha_u8: np.ndarray) -> np.ndarray:
    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
    rgba = np.dstack([rgb, alpha_u8])
    return rgba


def resolve_io_names(session: ort.InferenceSession) -> Tuple[str, str, str]:
    inputs = session.get_inputs()
    outputs = session.get_outputs()
    names = {i.name: i for i in inputs}

    image_name = "image" if "image" in names else inputs[0].name
    mask_name = "mask" if "mask" in names else (inputs[1].name if len(inputs) > 1 else "")
    if not mask_name:
        raise RuntimeError(f"模型需要 image+mask 两个输入，当前: {[i.name for i in inputs]}")
    out_name = outputs[0].name
    return image_name, mask_name, out_name


def infer_input_size(session: ort.InferenceSession, image_name: str, fallback: int) -> int:
    shape = session.get_inputs()[0].shape if image_name == session.get_inputs()[0].name else None
    for inp in session.get_inputs():
        if inp.name == image_name:
            shape = inp.shape
            break
    if shape is None:
        return fallback
    # expect NCHW: [N,C,H,W]
    try:
        h = int(shape[2])
        w = int(shape[3])
        if h > 0 and w > 0 and h == w:
            return h
    except Exception:
        pass
    return fallback


def run(
    model_path: str,
    image_path: str,
    mask_path: Optional[str],
    out_alpha: str,
    out_rgba: str,
    input_size: Optional[int],
) -> None:
    if not os.path.isfile(model_path):
        raise FileNotFoundError(f"模型不存在: {model_path}")
    bgr = cv2.imread(image_path, cv2.IMREAD_COLOR)
    if bgr is None:
        raise FileNotFoundError(f"无法读取图像: {image_path}")

    mask_u8 = None
    if mask_path:
        mask_u8 = cv2.imread(mask_path, cv2.IMREAD_UNCHANGED)
        if mask_u8 is None:
            raise FileNotFoundError(f"无法读取 mask: {mask_path}")

    sess = ort.InferenceSession(model_path, providers=["CPUExecutionProvider"])
    image_name, mask_name, out_name = resolve_io_names(sess)
    size = input_size or infer_input_size(sess, image_name, DEFAULT_INPUT_SIZE)

    print(f"model: {model_path}")
    print(f"inputs: {[ (i.name, i.shape) for i in sess.get_inputs() ]}")
    print(f"output: {[ (o.name, o.shape) for o in sess.get_outputs() ]}")
    print(f"input_size: {size}")

    image_nchw, mask_nchw, offsets, orig_hw = preprocess(bgr, mask_u8, size)
    feeds = {image_name: image_nchw, mask_name: mask_nchw}
    outs = sess.run([out_name], feeds)
    alpha_u8 = postprocess(outs[0], offsets, orig_hw)

    os.makedirs(os.path.dirname(os.path.abspath(out_alpha)) or ".", exist_ok=True)
    cv2.imwrite(out_alpha, alpha_u8)
    rgba = compose_rgba(bgr, alpha_u8)
    cv2.imwrite(out_rgba, cv2.cvtColor(rgba, cv2.COLOR_RGBA2BGRA))
    print(f"saved alpha: {out_alpha}")
    print(f"saved rgba : {out_rgba}")


def parse_args(argv=None):
    here = os.path.dirname(os.path.abspath(__file__))
    default_model = os.path.join(here, "..", "models", "portrait_matting_v5.1.onnx")
    if not os.path.isfile(default_model):
        default_model = "portrait_matting_v5.1.onnx"

    p = argparse.ArgumentParser(description="portrait_matting_v5.1 ONNX Runtime demo")
    p.add_argument("--model", default=default_model, help="portrait_matting_v5.1.onnx 路径")
    p.add_argument("--image", required=True, help="输入人像图 (BGR)")
    p.add_argument("--mask", default="", help="可选粗分割 mask（灰度，与原图同尺寸更佳）")
    p.add_argument("--out-alpha", default="matting_alpha.png", help="输出 alpha")
    p.add_argument("--out-rgba", default="matting_rgba.png", help="输出 RGBA 前景")
    p.add_argument("--size", type=int, default=0, help="强制输入边长，默认从模型读取或 512")
    return p.parse_args(argv)


def main(argv=None) -> int:
    args = parse_args(argv)
    try:
        run(
            model_path=args.model,
            image_path=args.image,
            mask_path=args.mask or None,
            out_alpha=args.out_alpha,
            out_rgba=args.out_rgba,
            input_size=args.size or None,
        )
    except Exception as exc:
        print(f"[FAIL] {exc}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
