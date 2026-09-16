#!/usr/bin/env python3
"""Visualize FocalNetMatting unknown regions before the first update.

Example:
    python tools/visualize_initial_unknown.py \
        --config configs/Matting/focal_matting_3gpu.yaml \
        --output-dir output/initial_unknown \
        --num-samples 4 \
        --device cuda:0

The script uses the training dataset/transforms and the configured initial
checkpoint, but it does not create an optimizer or update model parameters.
"""

import argparse
import csv
import logging
import pathlib
import random
import sys

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image, ImageDraw
from torch.utils.data._utils.collate import default_collate


PROJECT_ROOT = pathlib.Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from maggie.dataloader import (  # noqa: E402
    MattingBatchSampler,
    MattingDataset,
    MixedSupervisionBatchSampler,
    MixedSupervisionDataset,
    build_dataset,
)
from maggie.network import build_model  # noqa: E402
from maggie.utils.config import CONFIG  # noqa: E402


FINE_SCALE_NAMES = ('x8', 'x4', 'x2', 'x1')
IMAGE_MEAN = (0.485, 0.456, 0.406)
IMAGE_STD = (0.229, 0.224, 0.225)


def parse_args():
    parser = argparse.ArgumentParser(
        description='Save initial FocalNetMatting unknown-region previews')
    parser.add_argument('--config', required=True, help='training YAML config')
    parser.add_argument(
        '--output-dir', default='output/initial_unknown',
        help='directory used to save PNG previews and coverage.csv')
    parser.add_argument(
        '--num-samples', type=int, default=4,
        help='number of samples drawn from the first training batch')
    parser.add_argument(
        '--device', default='auto',
        help="inference device, for example 'cuda:0', 'cpu', or 'auto'")
    parser.add_argument(
        '--weights', default=None,
        help='optional checkpoint override; defaults to model.weights')
    parser.add_argument('--seed', type=int, default=None)
    return parser.parse_args()


def resolve_device(device):
    if device == 'auto':
        return torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    resolved = torch.device(device)
    if resolved.type == 'cuda' and not torch.cuda.is_available():
        raise RuntimeError(
            f'{device} was requested, but CUDA is not available')
    return resolved


def first_training_indices(dataset, num_samples, seed):
    """Use the same root-aware sampling policy as the training loop."""
    if isinstance(dataset, MixedSupervisionDataset):
        sampler = MixedSupervisionBatchSampler(
            dataset, batch_size=num_samples, num_replicas=1, rank=0,
            seed=seed)
        return next(iter(sampler))
    if (isinstance(dataset, MattingDataset) and
            dataset.uses_root_sampling_rates):
        sampler = MattingBatchSampler(
            dataset, batch_size=num_samples, num_replicas=1, rank=0,
            seed=seed)
        return next(iter(sampler))

    if len(dataset) < 1:
        raise ValueError('The training dataset is empty')
    random_state = np.random.RandomState(seed)
    replace = len(dataset) < num_samples
    return random_state.choice(
        len(dataset), size=num_samples, replace=replace).tolist()


def build_initial_batch(cfg, num_samples, seed):
    dataset = build_dataset(
        cfg.dataset.train, is_train=True, random_seed=seed)
    indices = first_training_indices(dataset, num_samples, seed)
    samples = [dataset[index] for index in indices]
    return default_collate(samples), indices


def unwrap_state_dict(checkpoint):
    if not isinstance(checkpoint, dict):
        raise TypeError(
            f'Checkpoint must contain a dict, got {type(checkpoint).__name__}')
    for key in ('state_dict', 'model'):
        value = checkpoint.get(key)
        if isinstance(value, dict):
            checkpoint = value
            break
    if checkpoint and all(name.startswith('module.') for name in checkpoint):
        checkpoint = {
            name[len('module.'):]: value
            for name, value in checkpoint.items()
        }
    return checkpoint


def load_initial_weights(model, model_cfg, device):
    weight_path = pathlib.Path(str(model_cfg.weights))
    if not weight_path.is_file():
        raise FileNotFoundError(f'Initial checkpoint not found: {weight_path}')

    state_dict = unwrap_state_dict(torch.load(weight_path, map_location=device))
    encoder_only = bool(getattr(model_cfg, 'load_encoder_only', False))
    target_module = model.encoder if encoder_only else model
    prefix = 'encoder.' if encoder_only else ''
    source = {
        name[len(prefix):]: value
        for name, value in state_dict.items()
        if name.startswith(prefix)
    }
    if not source:
        raise ValueError(
            f'No parameters with prefix {prefix!r} found in {weight_path}')

    target_state = target_module.state_dict()
    compatible = {
        name: value for name, value in source.items()
        if name in target_state and value.shape == target_state[name].shape
    }
    mismatch = [
        name for name, value in source.items()
        if name in target_state and value.shape != target_state[name].shape
    ]
    unexpected = [name for name in source if name not in target_state]
    missing = [name for name in target_state if name not in compatible]
    if not compatible:
        raise ValueError(
            f'No shape-compatible parameters found in {weight_path}')

    target_state.update(compatible)
    target_module.load_state_dict(target_state, strict=True)
    print(
        f'Loaded {len(compatible)}/{len(target_state)} tensors from '
        f'{weight_path} (missing={len(missing)}, mismatch={len(mismatch)}, '
        f'unexpected={len(unexpected)}, encoder_only={encoder_only})')


def denormalize_image(image):
    mean = image.new_tensor(IMAGE_MEAN).view(3, 1, 1)
    std = image.new_tensor(IMAGE_STD).view(3, 1, 1)
    image = (image.detach().cpu() * std.cpu() + mean.cpu()).clamp(0, 1)
    return (image.permute(1, 2, 0).numpy() * 255).round().astype(np.uint8)


def grayscale_image(tensor):
    array = tensor.detach().float().cpu().squeeze().clamp(0, 1).numpy()
    return (array * 255).round().astype(np.uint8)


def overlay_unknown(rgb, unknown):
    mask = unknown.detach().float().cpu().squeeze().numpy() > 0.5
    overlay = rgb.astype(np.float32).copy()
    color = np.asarray([255, 32, 32], dtype=np.float32)
    overlay[mask] = 0.45 * overlay[mask] + 0.55 * color
    return np.clip(overlay, 0, 255).astype(np.uint8)


def labeled_tile(array, label, label_height=28):
    image = Image.fromarray(array)
    if image.mode != 'RGB':
        image = image.convert('RGB')
    canvas = Image.new(
        'RGB', (image.width, image.height + label_height), 'white')
    canvas.paste(image, (0, label_height))
    ImageDraw.Draw(canvas).text((8, 7), label, fill='black')
    return canvas


def make_overview(rows):
    tile_width = max(tile.width for row in rows for tile in row)
    tile_height = max(tile.height for row in rows for tile in row)
    columns = max(len(row) for row in rows)
    overview = Image.new(
        'RGB', (columns * tile_width, len(rows) * tile_height), 'white')
    for row_index, row in enumerate(rows):
        for column_index, tile in enumerate(row):
            overview.paste(
                tile, (column_index * tile_width, row_index * tile_height))
    return overview


def visualize_batch(model, batch, output_dir):
    images = batch['image']
    alphas = batch['alpha']
    if images.ndim != 4 or alphas.ndim != 4:
        raise ValueError(
            'Expected image/alpha tensors in BCHW format, got '
            f'{tuple(images.shape)} and {tuple(alphas.shape)}')

    with torch.no_grad():
        features = model.encoder(images)
        predictions = model.decoder(features)
        if len(predictions) != 6:
            raise ValueError(
                'FocalDecoder must return x32/x16/x8/x4/x2/x1 in training '
                f'mode, got {len(predictions)} outputs')

        fine_predictions = []
        unknown_masks = []
        gt_unknown_base = model._build_gt_unknown_base(alphas)
        for index, logits in enumerate(predictions[2:]):
            pred_alpha = logits.sigmoid()
            if pred_alpha.shape[-2:] != alphas.shape[-2:]:
                pred_alpha = F.interpolate(
                    pred_alpha, size=alphas.shape[-2:], mode='bilinear',
                    align_corners=False)
            fine_predictions.append(pred_alpha)
            unknown_masks.append(model._build_unknown_weight(
                pred_alpha, alphas, index, gt_unknown_base))

    coverage_rows = []
    for sample_index in range(images.shape[0]):
        sample_dir = output_dir / f'sample_{sample_index:03d}'
        sample_dir.mkdir(parents=True, exist_ok=True)

        rgb = denormalize_image(images[sample_index])
        gt_alpha = grayscale_image(alphas[sample_index])
        gt_base = grayscale_image(gt_unknown_base[sample_index].float())
        Image.fromarray(rgb).save(sample_dir / 'image.png')
        Image.fromarray(gt_alpha).save(sample_dir / 'gt_alpha.png')
        Image.fromarray(gt_base).save(sample_dir / 'gt_unknown_base.png')

        prediction_tiles = []
        unknown_tiles = []
        for scale_name, pred_alpha, unknown in zip(
                FINE_SCALE_NAMES, fine_predictions, unknown_masks):
            pred_image = grayscale_image(pred_alpha[sample_index])
            unknown_image = grayscale_image(unknown[sample_index])
            overlay = overlay_unknown(rgb, unknown[sample_index])
            Image.fromarray(pred_image).save(
                sample_dir / f'pred_{scale_name}.png')
            Image.fromarray(unknown_image).save(
                sample_dir / f'unknown_{scale_name}.png')
            Image.fromarray(overlay).save(
                sample_dir / f'unknown_overlay_{scale_name}.png')

            coverage = float(unknown[sample_index].float().mean().cpu())
            coverage_rows.append({
                'sample': sample_index,
                'scale': scale_name,
                'unknown_ratio': coverage,
            })
            prediction_tiles.append(labeled_tile(
                pred_image, f'Pred {scale_name}'))
            unknown_tiles.append(labeled_tile(
                overlay, f'Unknown {scale_name}: {coverage:.2%}'))

        overview = make_overview([
            [
                labeled_tile(rgb, 'Augmented RGB'),
                labeled_tile(gt_alpha, 'GT alpha'),
                *prediction_tiles,
            ],
            [
                labeled_tile(gt_base, 'GT unknown base'),
                *unknown_tiles,
            ],
        ])
        overview.save(sample_dir / 'overview.png')

    with (output_dir / 'coverage.csv').open(
            'w', newline='', encoding='utf-8') as file:
        writer = csv.DictWriter(
            file, fieldnames=('sample', 'scale', 'unknown_ratio'))
        writer.writeheader()
        writer.writerows(coverage_rows)
    return coverage_rows


def main():
    args = parse_args()
    if args.num_samples < 1:
        raise ValueError('--num-samples must be positive')

    seed = args.seed
    cfg = CONFIG.clone()
    cfg.merge_from_file(args.config)
    if seed is None:
        seed = int(cfg.train.seed)
    if args.weights is not None:
        cfg.model.weights = args.weights

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    device = resolve_device(args.device)
    output_dir = pathlib.Path(args.output_dir).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    # Avoid the verbose per-parameter initialization messages emitted by the
    # encoder while retaining warnings and this script's explicit summaries.
    logging.disable(logging.INFO)
    batch, indices = build_initial_batch(cfg, args.num_samples, seed)
    model, _ = build_model(cfg.model)
    load_initial_weights(model, cfg.model, device)
    model = model.to(device)
    model.train()

    model_batch = {
        'image': batch['image'].to(device),
        'alpha': batch['alpha'].to(device),
    }
    coverage = visualize_batch(model, model_batch, output_dir)

    print(f'Device: {device}')
    print(f'Training sample indices: {indices}')
    for row in coverage:
        print(
            'sample={sample:03d}, scale={scale}, unknown={unknown_ratio:.2%}'
            .format(**row))
    print(f'Saved initial unknown visualizations to: {output_dir}')


if __name__ == '__main__':
    main()
