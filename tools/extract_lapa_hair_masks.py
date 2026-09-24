#!/usr/bin/env python3
"""Extract hair (class 10) from LaPa multi-class parsing label maps.

LaPa label images are single-channel index maps (``PIL`` mode ``L``) whose
pixel value is the class id:

===== ================
label class
===== ================
0     background
1     skin
2     left eyebrow
3     right eyebrow
4     left eye
5     right eye
6     nose
7     upper lip
8     inner mouth
9     lower lip
10    hair
===== ================

For every label image a binary mask is written where hair pixels are 255 and
all remaining classes are 0 (``uint8``/``L`` PNG).  Labels that contain no hair
pixel are skipped so no all-zero mask is ever saved.

Usage:
    python tools/extract_lapa_hair_masks.py [SOURCE] [OUTPUT]

SOURCE defaults to the LaPa validation labels and OUTPUT to
``<SOURCE>/../<SOURCE name>_hair``, e.g. ``.../LaPa/val/labels_hair``.
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Dict, Iterable, Optional, Tuple, Union

import numpy as np
from PIL import Image


# Class id -> class name, as defined by the LaPa parsing labels.
CLASS_NAMES: Dict[int, str] = {
    0: "background",
    1: "skin",
    2: "left eyebrow",
    3: "right eyebrow",
    4: "left eye",
    5: "right eye",
    6: "nose",
    7: "upper lip",
    8: "inner mouth",
    9: "lower lip",
    10: "hair",
}
HAIR_CLASS = 10
FOREGROUND_VALUE = 255

DEFAULT_SOURCE = "/data/xiaoshuai/hair_segmentation/dataset/LaPa/val/labels"
IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff", ".webp"}


def _label_files(source: Path) -> Iterable[Path]:
    """Yield label images in deterministic order."""

    return sorted(
        path
        for path in source.iterdir()
        if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS
    )


def read_label_map(label_path: Union[str, Path]) -> np.ndarray:
    """Read a LaPa parsing label as a 2-D array of class ids."""

    with Image.open(label_path) as image:
        # Index (``L``/``I``) and palette (``P``) labels both expose the class
        # id as the raw pixel value, so the array is taken unchanged.
        array = np.asarray(image)
    if array.ndim != 2:
        raise ValueError(
            f"Expected a single-channel index label, got shape {array.shape} "
            f"for {label_path}; RGB label maps are not supported"
        )
    return array


def hair_mask_from_label(label: np.ndarray) -> np.ndarray:
    """Return a 2-D uint8 binary hair mask (255 = hair, 0 = everything else)."""

    return (label == HAIR_CLASS).astype(np.uint8) * FOREGROUND_VALUE


def extract_hair_masks(
    source: Union[str, Path] = DEFAULT_SOURCE,
    output: Optional[Union[str, Path]] = None,
) -> Tuple[int, int]:
    """Extract hair masks and return ``(written_count, skipped_count)``.

    ``skipped_count`` counts labels whose hair mask is empty; those files are
    not written at all.
    """

    source_path = Path(source).expanduser()
    if not source_path.is_dir():
        raise NotADirectoryError(f"SOURCE is not a directory: {source_path}")

    output_path = (
        Path(output).expanduser()
        if output is not None
        else source_path.parent / f"{source_path.name}_hair"
    )
    output_path.mkdir(parents=True, exist_ok=True)

    written = 0
    skipped = 0
    for label_path in _label_files(source_path):
        mask = hair_mask_from_label(read_label_map(label_path))
        if not mask.any():
            skipped += 1
            continue
        # PNG keeps the mask lossless; reusing the stem keeps the output easy
        # to pair with its label and image.
        Image.fromarray(mask).save(output_path / f"{label_path.stem}.png")
        written += 1

    return written, skipped


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Extract binary hair masks (class 10) from LaPa labels."
    )
    parser.add_argument(
        "SOURCE",
        type=Path,
        nargs="?",
        default=Path(DEFAULT_SOURCE),
        help=f"folder containing LaPa label maps (default: {DEFAULT_SOURCE})",
    )
    parser.add_argument(
        "OUTPUT",
        type=Path,
        nargs="?",
        help="output folder (default: <SOURCE>/../<SOURCE name>_hair)",
    )
    args = parser.parse_args()

    try:
        written, skipped = extract_hair_masks(args.SOURCE, args.OUTPUT)
    except NotADirectoryError as error:
        parser.error(str(error))

    output = args.OUTPUT or args.SOURCE.parent / f"{args.SOURCE.name}_hair"
    print(
        f"Extracted {written} hair masks to {output}; "
        f"skipped {skipped} label(s) without hair"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())