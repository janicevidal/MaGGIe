#!/usr/bin/env python3
"""
导出 BiRefNet 模型到 ONNX 格式。

用法:
    python tools/export_onnx.py --config configs/biref_matting_test.yaml --weights /path/to/model.pth --output model.onnx --height 576 --width 576

参数说明:
    --config        配置文件路径 (必须)
    --weights       模型权重文件路径 (必须)
    --output        ONNX 输出路径 (默认: model.onnx)
    --height        固定输入高度 (默认: 576)
    --width         固定输入宽度 (默认: 576)
    --batch_size    固定 batch size (默认: 1, 若设为 -1 则动态)
    --dynamic       启用动态轴 (batch, height, width)
    --opset_version ONNX opset 版本 (默认: 13)
"""

import argparse
import os
import sys
import torch
import yaml
from yacs.config import CfgNode

# 添加项目根目录到路径
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from maggie.network import build_model
from maggie.utils import CONFIG

def parse_args():
    parser = argparse.ArgumentParser(description="Export BiRefNet to ONNX")
    parser.add_argument('--config', required=True, help='path to config file')
    parser.add_argument('--weights', required=True, help='path to model weights (.pth)')
    parser.add_argument('--output', default='model.onnx', help='output onnx file path')
    parser.add_argument('--height', type=int, default=576, help='input height')
    parser.add_argument('--width', type=int, default=576, help='input width')
    parser.add_argument('--batch_size', type=int, default=1, help='fixed batch size, -1 for dynamic')
    parser.add_argument('--dynamic', action='store_true', help='enable dynamic axes for batch, height, width')
    parser.add_argument('--opset_version', type=int, default=13, help='ONNX opset version')
    return parser.parse_args()

def main():
    args = parse_args()

    # 加载配置
    cfg = CONFIG
    cfg.merge_from_file(args.config)
    # 覆盖模型权重路径（如果配置中未指定）
    cfg.model.weights = args.weights

    # 构建模型（不加载权重，之后手动加载）
    model, is_from_hf = build_model(cfg.model)
    model.to('cpu')
    if not is_from_hf:
        # 加载权重
        state_dict = torch.load(args.weights, map_location='cpu')
        model.load_state_dict(state_dict, strict=True)
    else:
        print("Warning: Model loaded from HuggingFace, but weights argument ignored.")
    model.eval()
    model.requires_grad_(False)   # 冻结所有参数梯度
    # 禁用可能包含不支持操作的分支
    if hasattr(model.decoder, 'out_ref'):
        model.decoder.out_ref = False
    if hasattr(model.decoder, 'ms_supervision'):
        model.decoder.ms_supervision = False

    # 确定输入尺寸
    height = args.height
    width = args.width
    batch_size = args.batch_size

    # 创建 dummy 输入
    dummy_input = torch.randn(batch_size, 3, height, width).cpu()

    # 准备动态轴（如果启用）
    dynamic_axes = {}
    if args.dynamic:
        dynamic_axes = {
            'input': {0: 'batch_size', 2: 'height', 3: 'width'},
            'output': {0: 'batch_size', 2: 'height', 3: 'width'}
        }
    elif batch_size == -1:
        # 动态 batch size
        dynamic_axes = {
            'input': {0: 'batch_size'},
            'output': {0: 'batch_size'}
        }
    else:
        # 固定尺寸
        dynamic_axes = None

    # 导出 ONNX
    input_names = ['input']
    output_names = ['alpha_pred']

    # 定义 forward 包装，只取 alpha_pred
    class WrappedModel(torch.nn.Module):
        def forward(self, x):
            # 模型期望的 batch 结构: 输入的图像已经是归一化后的张量
            # 这里假设 x 已经经过 Normalize 预处理（像素值在 [0,1] 并减去均值除以标准差）
            # 如果模型内部没有归一化，则需在外部预处理。
            output = model({'image': x}, mem_feat=None)
            return output['alpha_pred']

    wrapped_model = WrappedModel()
    wrapped_model.eval()

    # 导出
    with torch.no_grad():
        torch.onnx.export(
            wrapped_model,
            dummy_input,
            args.output,
            opset_version=args.opset_version,
            input_names=input_names,
            output_names=output_names,
            dynamic_axes=dynamic_axes,
            do_constant_folding=True,
            verbose=False,
        )
    print(f"ONNX model exported to {args.output}")

if __name__ == '__main__':
    main()