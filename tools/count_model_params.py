"""Print hierarchical parameter counts for a MaGGIe configuration.

Example:
    python tools/count_model_params.py configs/DIS/focal_dis_2gpu.yaml

The report separates encoder, decoder and loss-related modules, then prints
all nested decoder modules so changes to FocalDecoder are easy to audit.
"""

import argparse
import pathlib
import sys
from collections import defaultdict

# When invoked as ``python tools/count_model_params.py``, Python places the
# tools directory (rather than the repository root) on sys.path.
PROJECT_ROOT = pathlib.Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import torch

from maggie.utils.config import CONFIG
from maggie.network.arch import *  # noqa: F401,F403 - architectures are config-selected


def _counts(module):
    total = sum(parameter.numel() for parameter in module.parameters())
    trainable = sum(
        parameter.numel() for parameter in module.parameters()
        if parameter.requires_grad)
    return total, trainable


def _format(value):
    return f"{value:,} ({value / 1e6:.3f}M)"


def _print_tree(module, prefix="", depth=0, max_depth=3):
    if depth > max_depth:
        return
    for name, child in module.named_children():
        total, trainable = _counts(child)
        print(f"{'  ' * depth}{prefix + name}: total={_format(total)}, "
              f"trainable={_format(trainable)}")
        _print_tree(child, prefix + name + '.', depth + 1, max_depth)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('config', help='path to a YAML configuration')
    parser.add_argument('--depth', type=int, default=3,
                        help='maximum nested module depth to print')
    args = parser.parse_args()

    cfg = CONFIG.clone()
    cfg.merge_from_file(args.config)
    model_cfg = cfg.model
    model = eval(model_cfg.arch)(model_cfg)

    total, trainable = _counts(model)
    print(f"model: {model_cfg.arch}")
    print(f"total parameters:     {_format(total)}")
    print(f"trainable parameters: {_format(trainable)}")
    print()

    top_level = {}
    for name, module in model.named_children():
        top_level[name] = _counts(module)
    print("top-level modules:")
    for name, (module_total, module_trainable) in top_level.items():
        ratio = 100.0 * module_total / max(total, 1)
        print(f"  {name}: total={_format(module_total)}, "
              f"trainable={_format(module_trainable)}, "
              f"share={ratio:.2f}%")

    if hasattr(model, 'decoder'):
        print("\ndecoder modules:")
        _print_tree(model.decoder, max_depth=args.depth)

    # Also aggregate parameters by their first name component, which catches
    # loss heads registered directly on the architecture.
    grouped = defaultdict(lambda: [0, 0])
    for name, parameter in model.named_parameters():
        root = name.split('.', 1)[0]
        grouped[root][0] += parameter.numel()
        if parameter.requires_grad:
            grouped[root][1] += parameter.numel()
    print("\nparameter groups:")
    for name, (group_total, group_trainable) in sorted(grouped.items()):
        print(f"  {name}: total={_format(group_total)}, "
              f"trainable={_format(group_trainable)}")
    
    weight_bytes = sum(
        parameter.numel() * parameter.element_size()
        for parameter in model.parameters()
    )
    print(f'weight storage: {weight_bytes / 1024**2:.2f} MB')


if __name__ == '__main__':
    main()
