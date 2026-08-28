"""Binarize masks and optionally visualize them over paired images."""

import argparse
import logging
from pathlib import Path

import numpy as np
from PIL import Image, ImageOps
from tqdm import tqdm


LOGGER = logging.getLogger(__name__)
IMAGE_EXTENSIONS = {
    '.jpg', '.jpeg', '.png', '.bmp', '.webp', '.tif', '.tiff',
}


def index_images(directory):
    """Index flat image files by stem and reject ambiguous pairs."""
    directory = Path(directory)
    if not directory.is_dir():
        raise NotADirectoryError(directory)

    index = {}
    duplicates = {}
    for path in directory.iterdir():
        if not path.is_file() or path.suffix.lower() not in IMAGE_EXTENSIONS:
            continue
        if path.stem in index:
            duplicates.setdefault(path.stem, [index[path.stem]]).append(path)
        else:
            index[path.stem] = path
    if duplicates:
        examples = '; '.join(
            f'{stem}: {[str(path) for path in paths]}'
            for stem, paths in list(duplicates.items())[:5])
        raise ValueError(
            'Multiple files have the same stem and cannot be paired '
            f'unambiguously: {examples}')
    return index


def load_binary_mask(mask_path, threshold=0):
    """Return 255 where any source channel is strictly above threshold."""
    with Image.open(mask_path) as mask_image:
        mask_image = ImageOps.exif_transpose(mask_image)
        mask = np.asarray(mask_image)
    if mask.ndim == 2:
        foreground = mask > threshold
    elif mask.ndim == 3:
        foreground = np.any(mask > threshold, axis=2)
    else:
        raise ValueError(
            f'Unsupported mask shape {mask.shape} in {mask_path}')
    return foreground.astype(np.uint8) * 255


def build_overlay(image_path, binary_mask, color=(255, 0, 0), alpha=0.5):
    """Overlay foreground pixels on an RGB image."""
    with Image.open(image_path) as image:
        image = ImageOps.exif_transpose(image).convert('RGB')
        image_array = np.asarray(image).copy()
    if image_array.shape[:2] != binary_mask.shape:
        raise ValueError(
            f'Image/mask size mismatch for {image_path}: '
            f'image={image_array.shape[:2]}, mask={binary_mask.shape}')

    foreground = binary_mask > 0
    overlay_color = np.asarray(color, dtype=np.float32)
    blended = (
        image_array[foreground].astype(np.float32) * (1.0 - alpha) +
        overlay_color * alpha)
    image_array[foreground] = np.rint(blended).clip(0, 255).astype(np.uint8)
    return Image.fromarray(image_array, mode='RGB')


def save_png(image, output_path):
    """Write a PNG atomically so interrupted runs do not leave partial files."""
    temporary_path = output_path.with_suffix(output_path.suffix + '.tmp')
    image.save(temporary_path, format='PNG')
    temporary_path.replace(output_path)


def process_mask_folders(image_dir, mask_dir, output_dir,
                         save_overlay=False, overlay_color=(255, 0, 0),
                         overlay_alpha=0.5, threshold=0, overwrite=False,
                         max_samples=0):
    image_dir = Path(image_dir)
    mask_dir = Path(mask_dir)
    output_dir = Path(output_dir)
    if not 0 <= overlay_alpha <= 1:
        raise ValueError('overlay_alpha must be between 0 and 1')
    if len(overlay_color) != 3 or any(
            not 0 <= channel <= 255 for channel in overlay_color):
        raise ValueError('overlay_color must contain three values in [0, 255]')
    if max_samples < 0:
        raise ValueError('max_samples cannot be negative')
    if not np.isfinite(threshold) or threshold < 0:
        raise ValueError('threshold must be a finite non-negative number')

    LOGGER.info('Indexing images in %s', image_dir)
    images = index_images(image_dir)
    LOGGER.info('Indexing masks in %s', mask_dir)
    masks = index_images(mask_dir)
    if not masks:
        raise FileNotFoundError(f'No supported mask images found in {mask_dir}')

    mask_items = sorted(masks.items())
    if max_samples:
        mask_items = mask_items[:max_samples]
    matched = sum(stem in images for stem, _ in mask_items)
    LOGGER.info(
        'Found images=%d, masks=%d, selected_masks=%d, paired=%d, '
        'missing_images=%d, threshold=%s',
        len(images), len(masks), len(mask_items), matched,
        len(mask_items) - matched, threshold)

    binary_dir = output_dir / 'binary_masks'
    overlay_dir = output_dir / 'overlays'
    binary_dir.mkdir(parents=True, exist_ok=True)
    if save_overlay:
        overlay_dir.mkdir(parents=True, exist_ok=True)

    binary_written = 0
    binary_skipped = 0
    overlay_written = 0
    overlay_skipped = 0
    errors = 0
    for stem, mask_path in tqdm(
            mask_items, unit='mask', desc='Binarizing masks'):
        binary_path = binary_dir / f'{stem}.png'
        overlay_path = overlay_dir / f'{stem}.png'
        need_binary = overwrite or not binary_path.exists()
        need_overlay = (
            save_overlay and (overwrite or not overlay_path.exists()))
        if not need_binary and not need_overlay:
            binary_skipped += 1
            if save_overlay:
                overlay_skipped += 1
            continue

        try:
            binary_mask = load_binary_mask(mask_path, threshold=threshold)
            if need_binary:
                save_png(Image.fromarray(binary_mask, mode='L'), binary_path)
                binary_written += 1
            else:
                binary_skipped += 1

            if save_overlay:
                if not need_overlay:
                    overlay_skipped += 1
                elif stem not in images:
                    raise FileNotFoundError(
                        f'No paired image for mask {mask_path.name}')
                else:
                    overlay = build_overlay(
                        images[stem], binary_mask,
                        color=overlay_color, alpha=overlay_alpha)
                    save_png(overlay, overlay_path)
                    overlay_written += 1
        except Exception as error:
            errors += 1
            LOGGER.error('Failed %s: %s', mask_path, error)

    LOGGER.info(
        'Finished: binary_written=%d, binary_skipped=%d, '
        'overlay_written=%d, overlay_skipped=%d, errors=%d, output=%s',
        binary_written, binary_skipped, overlay_written, overlay_skipped,
        errors, output_dir)
    return {
        'total': len(mask_items),
        'binary_written': binary_written,
        'binary_skipped': binary_skipped,
        'overlay_written': overlay_written,
        'overlay_skipped': overlay_skipped,
        'errors': errors,
    }


def parse_args():
    parser = argparse.ArgumentParser(
        description='Binarize masks and optionally create image overlays')
    parser.add_argument('--image-dir', type=Path, required=True)
    parser.add_argument('--mask-dir', type=Path, required=True)
    parser.add_argument('--output-dir', type=Path, required=True)
    parser.add_argument('--save-overlay', action='store_true')
    parser.add_argument(
        '--overlay-color', type=int, nargs=3, default=(255, 0, 0),
        metavar=('R', 'G', 'B'))
    parser.add_argument('--overlay-alpha', type=float, default=0.5)
    parser.add_argument(
        '--threshold', type=float, default=0,
        help='Pixels/channels strictly greater than this source-mask value '
             'are foreground (default: 0)')
    parser.add_argument('--overwrite', action='store_true')
    parser.add_argument('--max-samples', type=int, default=0,
                        help='Process at most N masks; 0 means all')
    parser.add_argument(
        '--log-level', default='INFO',
        choices=['DEBUG', 'INFO', 'WARNING', 'ERROR'])
    return parser.parse_args()


def main():
    args = parse_args()
    logging.basicConfig(
        level=getattr(logging, args.log_level),
        format='%(asctime)s - %(levelname)s - %(message)s')
    result = process_mask_folders(
        image_dir=args.image_dir,
        mask_dir=args.mask_dir,
        output_dir=args.output_dir,
        save_overlay=args.save_overlay,
        overlay_color=tuple(args.overlay_color),
        overlay_alpha=args.overlay_alpha,
        threshold=args.threshold,
        overwrite=args.overwrite,
        max_samples=args.max_samples)
    if result['errors']:
        raise SystemExit(1)


if __name__ == '__main__':
    main()
