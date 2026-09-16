#!/usr/bin/env python3
"""Pair and concatenate images and their masks.

The input images and masks are expected to have matching filename stems and
size ``512 x 1024`` (width x height).  Files are sorted by filename stem and consumed
in adjacent, non-overlapping pairs.  Each pair is concatenated horizontally,
producing a ``1024 x 1024`` output image and mask.

Usage:
    python tools/pair_concat_images_masks.py \
        IMAGE_SOURCE MASK_SOURCE IMAGE_OUTPUT MASK_OUTPUT
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import List, Tuple, Union

import numpy as np
from PIL import Image


EXPECTED_SIZE = (512, 1024)  # width, height
IMAGE_EXTENSIONS = {".bmp", ".jpeg", ".jpg", ".png", ".tif", ".tiff", ".webp"}


def _files(folder: Path) -> List[Path]:
    """Return image files in *folder*, sorted by filename."""

    return sorted(
        (
            path
            for path in folder.iterdir()
            if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS
        ),
        key=lambda path: path.name,
    )


def _validate_inputs(image_source: Path, mask_source: Path) -> Tuple[List[Path], List[Path]]:
    if not image_source.is_dir():
        raise NotADirectoryError(f"image source is not a directory: {image_source}")
    if not mask_source.is_dir():
        raise NotADirectoryError(f"mask source is not a directory: {mask_source}")

    image_files = _files(image_source)
    mask_files = _files(mask_source)
    # Match by stem so ``0001.jpg`` can be paired with ``0001.png``.  A
    # duplicate stem is ambiguous and should be fixed instead of silently
    # selecting one file.  The mask directory may intentionally be a subset
    # of the image directory (for example, after filtering Accessories).
    image_by_stem = {path.stem: path for path in image_files}
    mask_by_stem = {path.stem: path for path in mask_files}
    if len(image_by_stem) != len(image_files) or len(mask_by_stem) != len(mask_files):
        raise ValueError("duplicate image or mask filename stems found")
    image_only = set(image_by_stem) - set(mask_by_stem)
    mask_only = set(mask_by_stem) - set(image_by_stem)
    if image_only:
        print(
            f"warning: ignoring {len(image_only)} images without a matching mask",
            file=sys.stderr,
        )
    if mask_only:
        print(
            f"warning: ignoring {len(mask_only)} masks without a matching image",
            file=sys.stderr,
        )

    # Rebuild both lists from the sorted intersection to guarantee pairing.
    stems = sorted(set(image_by_stem) & set(mask_by_stem))
    if not stems:
        raise ValueError("no image/mask filename stems match")
    if len(stems) % 2:
        dropped = stems[-1]
        print(
            f"warning: {len(stems)} matched files is odd; "
            f"ignoring unpaired sample {dropped!r}",
            file=sys.stderr,
        )
        stems = stems[:-1]
    if not stems:
        raise ValueError("fewer than two matching image/mask samples found")
    image_files = [image_by_stem[stem] for stem in stems]
    mask_files = [mask_by_stem[stem] for stem in stems]

    return image_files, mask_files


def _concat(path_a: Path, path_b: Path, convert_rgb: bool = False) -> Image.Image:
    """Load two images and concatenate them left-to-right.

    Source images are converted to RGB by the caller.  Masks retain their
    original mode (RGB or grayscale), so a binary mask does not unnecessarily
    become a three-channel image.
    """

    with Image.open(path_a) as first, Image.open(path_b) as second:
        if first.size != EXPECTED_SIZE or second.size != EXPECTED_SIZE:
            raise ValueError(
                f"expected {EXPECTED_SIZE[0]}x{EXPECTED_SIZE[1]} inputs, got "
                f"{path_a.name}: {first.size}, {path_b.name}: {second.size}"
            )
        first_array = np.asarray(first.convert("RGB") if convert_rgb else first)
        second_array = np.asarray(second.convert("RGB") if convert_rgb else second)
    if (
        first_array.ndim != second_array.ndim
        or first_array.shape[2:] != second_array.shape[2:]
    ):
        raise ValueError(
            f"cannot concatenate files with different channel layouts: "
            f"{path_a.name}: {first_array.shape}, {path_b.name}: {second_array.shape}"
        )
    return Image.fromarray(np.concatenate((first_array, second_array), axis=1))


def pair_and_concat(
    image_source: Union[str, Path],
    mask_source: Union[str, Path],
    image_output: Union[str, Path],
    mask_output: Union[str, Path],
    prefix: str = "pair",
) -> int:
    """Create paired outputs and return the number of pairs written."""

    image_source_path = Path(image_source).expanduser()
    mask_source_path = Path(mask_source).expanduser()
    image_files, mask_files = _validate_inputs(image_source_path, mask_source_path)

    image_output_path = Path(image_output).expanduser()
    mask_output_path = Path(mask_output).expanduser()
    image_output_path.mkdir(parents=True, exist_ok=True)
    mask_output_path.mkdir(parents=True, exist_ok=True)

    pair_count = len(image_files) // 2
    for pair_index in range(pair_count):
        left = pair_index * 2
        right = left + 1
        # Numeric names make ordering independent of the lengths of source
        # filenames and avoid collisions when stems contain unusual chars.
        output_name = f"{prefix}_{pair_index:06d}.png"
        _concat(image_files[left], image_files[right], convert_rgb=True).save(
            image_output_path / output_name
        )
        _concat(mask_files[left], mask_files[right]).save(mask_output_path / output_name)

    return pair_count


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Concatenate sorted image/mask files in non-overlapping pairs."
    )
    parser.add_argument("IMAGE_SOURCE", type=Path, help="folder containing source images")
    parser.add_argument("MASK_SOURCE", type=Path, help="folder containing same-named masks")
    parser.add_argument("IMAGE_OUTPUT", type=Path, help="folder for concatenated images")
    parser.add_argument("MASK_OUTPUT", type=Path, help="folder for concatenated masks")
    parser.add_argument("--prefix", default="pair", help="output filename prefix (default: pair)")
    args = parser.parse_args()

    try:
        pair_count = pair_and_concat(
            args.IMAGE_SOURCE,
            args.MASK_SOURCE,
            args.IMAGE_OUTPUT,
            args.MASK_OUTPUT,
            prefix=args.prefix,
        )
    except (NotADirectoryError, ValueError, OSError) as error:
        parser.error(str(error))

    print(f"Created {pair_count} image/mask pairs")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
