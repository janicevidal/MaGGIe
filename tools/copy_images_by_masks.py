"""Copy images whose filename stems are present in a mask directory."""

import argparse
import logging
import shutil
from pathlib import Path

from tqdm import tqdm


LOGGER = logging.getLogger(__name__)
IMAGE_EXTENSIONS = {
    '.jpg', '.jpeg', '.png', '.webp', '.bmp', '.tif', '.tiff',
}


def index_by_stem(directory):
    """Index supported files in a flat directory and reject duplicate stems."""
    directory = Path(directory)
    if not directory.is_dir():
        raise NotADirectoryError(directory)

    files = {}
    duplicates = {}
    for path in directory.iterdir():
        if not path.is_file() or path.suffix.lower() not in IMAGE_EXTENSIONS:
            continue
        if path.stem in files:
            duplicates.setdefault(path.stem, [files[path.stem]]).append(path)
        else:
            files[path.stem] = path
    if duplicates:
        examples = '; '.join(
            f'{stem}: {[path.name for path in paths]}'
            for stem, paths in list(duplicates.items())[:5])
        raise ValueError(
            f'Multiple files with the same stem under {directory}: {examples}')
    return files


def atomic_copy(source, destination, overwrite=False):
    if destination.exists() and not overwrite:
        return False
    temporary = destination.with_suffix(destination.suffix + '.tmp')
    shutil.copy2(source, temporary)
    temporary.replace(destination)
    return True


def copy_images_by_masks(image_dir, mask_dir, output_dir,
                         overwrite=False, strict=False):
    """Copy images matched to masks by stem, preserving image filenames."""
    image_dir = Path(image_dir)
    mask_dir = Path(mask_dir)
    output_dir = Path(output_dir)

    LOGGER.info('Indexing images: %s', image_dir)
    images = index_by_stem(image_dir)
    LOGGER.info('Indexing masks: %s', mask_dir)
    masks = index_by_stem(mask_dir)
    if not masks:
        raise FileNotFoundError(f'No supported mask files found in {mask_dir}')

    missing_stems = sorted(set(masks) - set(images))
    matched_stems = sorted(set(masks) & set(images))
    LOGGER.info(
        'Found images=%d, masks=%d, matched=%d, missing_images=%d',
        len(images), len(masks), len(matched_stems), len(missing_stems))
    if missing_stems:
        LOGGER.warning(
            'No matching image for %d masks; examples: %s',
            len(missing_stems), ', '.join(missing_stems[:20]))
        if strict:
            raise FileNotFoundError(
                f'{len(missing_stems)} masks have no matching image')

    output_dir.mkdir(parents=True, exist_ok=True)
    copied = 0
    skipped = 0
    errors = 0
    for stem in tqdm(
            matched_stems, unit='image', desc='Copying images'):
        source = images[stem]
        destination = output_dir / source.name
        try:
            if atomic_copy(source, destination, overwrite=overwrite):
                copied += 1
            else:
                skipped += 1
        except Exception as error:
            errors += 1
            LOGGER.error('Failed %s -> %s: %s', source, destination, error)

    LOGGER.info(
        'Finished: matched=%d, copied=%d, skipped=%d, missing=%d, '
        'errors=%d, output=%s',
        len(matched_stems), copied, skipped, len(missing_stems),
        errors, output_dir)
    return {
        'images': len(images),
        'masks': len(masks),
        'matched': len(matched_stems),
        'copied': copied,
        'skipped': skipped,
        'missing': len(missing_stems),
        'errors': errors,
    }


def parse_args():
    parser = argparse.ArgumentParser(
        description='Copy images selected by filenames in a mask directory')
    parser.add_argument('--image-dir', type=Path, required=True)
    parser.add_argument('--mask-dir', type=Path, required=True)
    parser.add_argument('--output-dir', type=Path, required=True)
    parser.add_argument('--overwrite', action='store_true')
    parser.add_argument(
        '--strict', action='store_true',
        help='Fail before copying if any mask has no matching image')
    parser.add_argument(
        '--log-level', default='INFO',
        choices=['DEBUG', 'INFO', 'WARNING', 'ERROR'])
    return parser.parse_args()


def main():
    args = parse_args()
    logging.basicConfig(
        level=getattr(logging, args.log_level),
        format='%(asctime)s - %(levelname)s - %(message)s')
    result = copy_images_by_masks(
        image_dir=args.image_dir,
        mask_dir=args.mask_dir,
        output_dir=args.output_dir,
        overwrite=args.overwrite,
        strict=args.strict)
    if result['errors']:
        raise SystemExit(1)


if __name__ == '__main__':
    main()
