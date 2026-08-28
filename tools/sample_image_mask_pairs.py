"""Randomly move paired images and masks from multiple datasets."""

import argparse
import json
import logging
import random
import re
import shutil
from pathlib import Path

from tqdm import tqdm


LOGGER = logging.getLogger(__name__)
IMAGE_EXTENSIONS = (
    '.jpg', '.jpeg', '.png', '.webp', '.bmp', '.tif', '.tiff',
)


def safe_name(value):
    name = re.sub(r'[^0-9A-Za-z._-]+', '_', str(value)).strip('._')
    return name or 'dataset'


def index_by_stem(directory):
    """Index supported files in a flat directory by filename stem."""
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
            f'{stem}: {[path.name for path in paths]}'
            for stem, paths in list(duplicates.items())[:5])
        raise ValueError(
            f'Ambiguous duplicate stems under {directory}: {examples}')
    return index


def infer_image_dir(mask_dir):
    """Infer a sibling images directory from a mask directory."""
    mask_dir = Path(mask_dir)
    candidates = [
        mask_dir.parent / 'images',
        mask_dir.parent / 'image',
        mask_dir.parent / 'imgs',
    ]
    for candidate in candidates:
        if candidate.is_dir():
            return candidate
    raise FileNotFoundError(
        f'Could not infer an image directory for {mask_dir}; expected one of '
        f'{candidates}. Pass --image-dirs explicitly.')


def collect_pairs(mask_dirs, image_dirs=None):
    """Collect image-mask pairs matched by stem from all source datasets."""
    mask_dirs = [Path(path) for path in mask_dirs]
    if image_dirs:
        image_dirs = [Path(path) for path in image_dirs]
        if len(image_dirs) != len(mask_dirs):
            raise ValueError(
                '--image-dirs must have exactly one path for each --mask-dirs '
                f'entry ({len(image_dirs)} vs {len(mask_dirs)})')
    else:
        image_dirs = [infer_image_dir(mask_dir) for mask_dir in mask_dirs]

    pairs = []
    for source_index, (mask_dir, image_dir) in enumerate(
            zip(mask_dirs, image_dirs)):
        masks = index_by_stem(mask_dir)
        images = index_by_stem(image_dir)
        matched_stems = sorted(set(masks) & set(images))
        LOGGER.info(
            'Source %d: image_dir=%s, mask_dir=%s, images=%d, masks=%d, '
            'paired=%d, masks_without_image=%d',
            source_index, image_dir, mask_dir, len(images), len(masks),
            len(matched_stems), len(set(masks) - set(images)))
        source_name = safe_name(mask_dir.parent.name or mask_dir.name)
        for stem in matched_stems:
            pairs.append({
                'source_index': source_index,
                'source_name': source_name,
                'stem': stem,
                'image_path': images[stem],
                'mask_path': masks[stem],
            })
    return pairs


def unique_output_stems(pairs):
    """Prefix source folder names and disambiguate repeated dataset names."""
    bases = [
        safe_name(f"{pair['source_name']}_{pair['stem']}")
        for pair in pairs
    ]
    base_counts = {}
    for base in bases:
        base_counts[base] = base_counts.get(base, 0) + 1

    used = set()
    output_stems = []
    for pair, base in zip(pairs, bases):
        if base_counts[base] > 1:
            base = safe_name(
                f"{pair['source_name']}_{pair['source_index']}_"
                f"{pair['stem']}")
        output_stem = base
        suffix = 2
        while output_stem in used:
            output_stem = f'{base}_{suffix}'
            suffix += 1
        used.add(output_stem)
        output_stems.append(output_stem)
    return output_stems


def move_file(source, destination):
    """Move one file, including across filesystems."""
    destination.parent.mkdir(parents=True, exist_ok=True)
    shutil.move(str(source), str(destination))


def move_pair(image_source, mask_source, image_destination, mask_destination,
              overwrite=False):
    """Move a pair and attempt rollback if moving its mask fails."""
    destinations = (image_destination, mask_destination)
    existing = [path for path in destinations if path.exists()]
    if existing and not overwrite:
        raise FileExistsError(
            f'Destination files already exist: {existing}')
    if overwrite:
        for path in existing:
            path.unlink()

    move_file(image_source, image_destination)
    try:
        move_file(mask_source, mask_destination)
    except Exception:
        try:
            if image_destination.exists() and not image_source.exists():
                move_file(image_destination, image_source)
        except Exception as rollback_error:
            LOGGER.error(
                'Image rollback also failed: %s -> %s: %s',
                image_destination, image_source, rollback_error)
        raise


def sample_pairs(mask_dirs, output_dir, image_dirs=None, num_samples=500,
                 seed=2026, overwrite=False):
    """Randomly sample pairs without replacement and move them by type."""
    if num_samples < 1:
        raise ValueError('num_samples must be positive')
    output_dir = Path(output_dir)
    manifest_path = output_dir / 'sample_manifest.jsonl'
    if manifest_path.exists() and not overwrite:
        raise FileExistsError(
            f'{manifest_path} already exists. Because moving changes the '
            'source population, use a new output directory or explicitly '
            'pass --overwrite.')
    pairs = collect_pairs(mask_dirs, image_dirs=image_dirs)
    if len(pairs) < num_samples:
        raise ValueError(
            f'Requested {num_samples} pairs, but only {len(pairs)} valid '
            'image-mask pairs were found')

    random_generator = random.Random(seed)
    selected = random_generator.sample(pairs, num_samples)
    output_stems = unique_output_stems(selected)
    image_output_dir = output_dir / 'images'
    mask_output_dir = output_dir / 'masks'
    image_output_dir.mkdir(parents=True, exist_ok=True)
    mask_output_dir.mkdir(parents=True, exist_ok=True)
    LOGGER.info(
        'Randomly selected %d/%d pairs with seed=%d; moving to %s. '
        'Selected source files will be removed.',
        num_samples, len(pairs), seed, output_dir)

    manifest_records = []
    moved = 0
    errors = 0
    for pair, output_stem in tqdm(
            zip(selected, output_stems), total=len(selected),
            unit='pair', desc='Moving pairs'):
        image_suffix = pair['image_path'].suffix.lower()
        mask_suffix = pair['mask_path'].suffix.lower()
        image_output = image_output_dir / f'{output_stem}{image_suffix}'
        mask_output = mask_output_dir / f'{output_stem}{mask_suffix}'
        try:
            move_pair(
                pair['image_path'], pair['mask_path'],
                image_output, mask_output, overwrite=overwrite)
            moved += 1
            manifest_records.append({
                'output_stem': output_stem,
                'image': str(Path('images') / image_output.name),
                'mask': str(Path('masks') / mask_output.name),
                'source_index': pair['source_index'],
                'source_image': str(pair['image_path']),
                'source_mask': str(pair['mask_path']),
            })
        except Exception as error:
            errors += 1
            LOGGER.error(
                'Failed pair image=%s mask=%s: %s',
                pair['image_path'], pair['mask_path'], error)

    temporary_manifest = manifest_path.with_suffix('.jsonl.tmp')
    with temporary_manifest.open('w', encoding='utf-8') as manifest:
        for record in manifest_records:
            manifest.write(json.dumps(record, ensure_ascii=False) + '\n')
    temporary_manifest.replace(manifest_path)
    LOGGER.info(
        'Finished: requested=%d, moved=%d, errors=%d, manifest=%s',
        num_samples, moved, errors, manifest_path)
    return {
        'available': len(pairs),
        'requested': num_samples,
        'moved': moved,
        'errors': errors,
        'manifest': manifest_path,
    }


def parse_args():
    parser = argparse.ArgumentParser(
        description='Randomly move paired images and masks from directories')
    parser.add_argument(
        '--mask-dirs', type=Path, nargs='+', required=True,
        help='One or more flat mask directories')
    parser.add_argument(
        '--image-dirs', type=Path, nargs='+',
        help='Image directories aligned with mask-dirs; if omitted, sibling '
             'images directories are inferred')
    parser.add_argument('--output-dir', type=Path, required=True)
    parser.add_argument('--num-samples', type=int, default=500)
    parser.add_argument('--seed', type=int, default=2026)
    parser.add_argument('--overwrite', action='store_true')
    parser.add_argument(
        '--log-level', default='INFO',
        choices=['DEBUG', 'INFO', 'WARNING', 'ERROR'])
    return parser.parse_args()


def main():
    args = parse_args()
    logging.basicConfig(
        level=getattr(logging, args.log_level),
        format='%(asctime)s - %(levelname)s - %(message)s')
    result = sample_pairs(
        mask_dirs=args.mask_dirs,
        image_dirs=args.image_dirs,
        output_dir=args.output_dir,
        num_samples=args.num_samples,
        seed=args.seed,
        overwrite=args.overwrite)
    if result['errors']:
        raise SystemExit(1)


if __name__ == '__main__':
    main()
