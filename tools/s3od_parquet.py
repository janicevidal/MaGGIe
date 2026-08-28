"""Extract S3OD images, masks, and captions from Parquet shards.

The output directory is compatible with ``BinarySegmentationDataset``::

    output_dir/
      images/<image_id>.<source extension>
      masks/<image_id>.<source extension>
      captions/<image_id>.txt
"""

import argparse
import logging
import os
from io import BytesIO
from pathlib import Path

from PIL import Image
import pyarrow.parquet as pq
from tqdm import tqdm


LOGGER = logging.getLogger(__name__)
REQUIRED_COLUMNS = ('image', 'mask', 'image_id', 'caption')


def safe_image_id(value):
    """Return an image ID that cannot escape its output directory."""
    image_id = str(value).strip()
    if not image_id:
        raise ValueError('image_id is empty')
    if image_id in {'.', '..'} or Path(image_id).name != image_id:
        raise ValueError(f'Unsafe image_id: {image_id!r}')
    if os.sep in image_id or (os.altsep and os.altsep in image_id):
        raise ValueError(f'Unsafe image_id: {image_id!r}')
    return image_id


def encoded_bytes(value):
    """Get encoded image bytes from a HuggingFace Image struct."""
    if isinstance(value, dict):
        data = value.get('bytes')
        if data is not None:
            return bytes(data)

        source_path = value.get('path')
        if source_path and Path(source_path).is_file():
            return Path(source_path).read_bytes()
        raise ValueError(
            f'Image struct has neither bytes nor a readable path: {value}')

    if isinstance(value, (bytes, bytearray, memoryview)):
        return bytes(value)
    raise TypeError(f'Unsupported encoded image value: {type(value)}')


def image_suffix(value, data, default_suffix):
    """Prefer the stored path suffix, falling back to encoded image format."""
    if isinstance(value, dict):
        suffix = Path(value.get('path') or '').suffix.lower()
        if suffix:
            return suffix

    try:
        with Image.open(BytesIO(data)) as image:
            suffix = Image.registered_extensions()
            for extension, image_format in suffix.items():
                if image_format == image.format:
                    return extension
    except Exception:
        pass
    return default_suffix


def atomic_write_bytes(path, data):
    temporary_path = path.with_suffix(path.suffix + '.tmp')
    temporary_path.write_bytes(data)
    temporary_path.replace(path)


def atomic_write_text(path, text):
    temporary_path = path.with_suffix(path.suffix + '.tmp')
    temporary_path.write_text(text, encoding='utf-8')
    temporary_path.replace(path)


def validate_schema(parquet_file, file_path):
    columns = set(parquet_file.schema_arrow.names)
    missing = set(REQUIRED_COLUMNS) - columns
    if missing:
        raise ValueError(
            f'{file_path} is missing required columns: {sorted(missing)}')


def extract_row(row, image_dir, mask_dir, caption_dir, overwrite=False):
    image_id = safe_image_id(row['image_id'])
    image_data = encoded_bytes(row['image'])
    mask_data = encoded_bytes(row['mask'])
    image_path = image_dir / (
        image_id + image_suffix(row['image'], image_data, '.jpg'))
    mask_path = mask_dir / (
        image_id + image_suffix(row['mask'], mask_data, '.png'))
    caption_path = caption_dir / f'{image_id}.txt'

    output_paths = (image_path, mask_path, caption_path)
    if not overwrite and all(path.exists() for path in output_paths):
        return False

    if overwrite or not image_path.exists():
        atomic_write_bytes(image_path, image_data)
    if overwrite or not mask_path.exists():
        atomic_write_bytes(mask_path, mask_data)
    if overwrite or not caption_path.exists():
        caption = row['caption']
        atomic_write_text(caption_path, '' if caption is None else str(caption))
    return True


def extract_parquet_directory(input_dir, output_dir, pattern='*.parquet',
                              batch_size=32, overwrite=False,
                              max_samples=0):
    """Extract all matching Parquet shards without loading a shard at once."""
    input_dir = Path(input_dir)
    output_dir = Path(output_dir)
    parquet_files = sorted(input_dir.glob(pattern))
    if not parquet_files:
        raise FileNotFoundError(
            f'No Parquet files matched {pattern!r} under {input_dir}')
    if batch_size < 1:
        raise ValueError('batch_size must be positive')
    if max_samples < 0:
        raise ValueError('max_samples cannot be negative')

    image_dir = output_dir / 'images'
    mask_dir = output_dir / 'masks'
    caption_dir = output_dir / 'captions'
    for directory in (image_dir, mask_dir, caption_dir):
        directory.mkdir(parents=True, exist_ok=True)

    total_rows = sum(pq.ParquetFile(path).metadata.num_rows
                     for path in parquet_files)
    if max_samples:
        total_rows = min(total_rows, max_samples)
    LOGGER.info('Found %d Parquet shards with %d rows',
                len(parquet_files), total_rows)

    processed = 0
    skipped = 0
    errors = 0
    visited = 0
    progress = tqdm(total=total_rows, unit='sample', desc='Extracting S3OD')
    try:
        for file_path in parquet_files:
            if max_samples and visited >= max_samples:
                break
            LOGGER.info('Reading %s', file_path)
            parquet_file = pq.ParquetFile(file_path)
            validate_schema(parquet_file, file_path)

            batches = parquet_file.iter_batches(
                batch_size=batch_size, columns=list(REQUIRED_COLUMNS))
            for batch in batches:
                for row_index, row in enumerate(batch.to_pylist()):
                    if max_samples and visited >= max_samples:
                        break
                    try:
                        written = extract_row(
                            row, image_dir, mask_dir, caption_dir, overwrite)
                        if written:
                            processed += 1
                        else:
                            skipped += 1
                    except Exception as error:
                        errors += 1
                        LOGGER.error(
                            'Failed %s batch row %d (image_id=%r): %s',
                            file_path.name, row_index,
                            row.get('image_id'), error)
                    finally:
                        visited += 1
                        progress.update(1)
                if max_samples and visited >= max_samples:
                    break
    finally:
        progress.close()

    LOGGER.info(
        'Done: processed=%d, skipped=%d, errors=%d, output=%s',
        processed, skipped, errors, output_dir)
    return {
        'processed': processed,
        'skipped': skipped,
        'errors': errors,
        'visited': visited,
    }


def inspect_parquet(file_path):
    parquet_file = pq.ParquetFile(file_path)
    print(f'File: {file_path}')
    print(f'Rows: {parquet_file.metadata.num_rows}')
    print(f'Row groups: {parquet_file.num_row_groups}')
    print(f'Schema:\n{parquet_file.schema_arrow}')

    batch = next(parquet_file.iter_batches(batch_size=1))
    row = batch.to_pylist()[0]
    print(f'First image_id: {row.get("image_id")!r}')
    for column in ('image', 'mask'):
        value = row.get(column)
        data = encoded_bytes(value)
        with Image.open(BytesIO(data)) as image:
            print(
                f'{column}: format={image.format}, mode={image.mode}, '
                f'size={image.size}, path={value.get("path")!r}')
    print(f'First caption: {row.get("caption")!r}')


def parse_args():
    parser = argparse.ArgumentParser(
        description='Extract S3OD images, masks, and per-image captions')
    parser.add_argument('--input-dir', type=Path,
                        help='Directory containing Parquet shards')
    parser.add_argument('--output-dir', type=Path,
                        help='Output root containing images/masks/captions')
    parser.add_argument('--pattern', default='train-*.parquet',
                        help='Parquet filename glob under input-dir')
    parser.add_argument('--batch-size', type=int, default=32,
                        help='Rows decoded from Parquet at a time')
    parser.add_argument('--overwrite', action='store_true',
                        help='Overwrite already extracted files')
    parser.add_argument('--max-samples', type=int, default=0,
                        help='Stop after N rows; 0 extracts everything')
    parser.add_argument('--inspect', type=Path, metavar='PARQUET_FILE',
                        help='Only inspect one Parquet file')
    return parser.parse_args()


def main():
    args = parse_args()
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s')
    if args.inspect:
        inspect_parquet(args.inspect)
        return
    if args.input_dir is None or args.output_dir is None:
        raise SystemExit(
            '--input-dir and --output-dir are required unless --inspect is used')
    extract_parquet_directory(
        input_dir=args.input_dir,
        output_dir=args.output_dir,
        pattern=args.pattern,
        batch_size=args.batch_size,
        overwrite=args.overwrite,
        max_samples=args.max_samples)


if __name__ == '__main__':
    main()
