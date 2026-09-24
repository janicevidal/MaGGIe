#!/usr/bin/env python3
"""Extract selected classes from multi-value RGB segmentation masks.

Usage:
    python tools/extract_rgb_label_masks.py SOURCE [OUTPUT]

When OUTPUT is omitted, masks are written to ``SOURCE_binary``.
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Dict, Iterable, Optional, Tuple, Union

import numpy as np
from PIL import Image


# Class ID -> RGB color, copied from the dataset label table.
CLASS_COLORS: Dict[int, Tuple[int, int, int]] = {
    2: (255, 0, 0),   # Hair
}

TARGET_COLORS = np.asarray(tuple(CLASS_COLORS.values()), dtype=np.uint8)


def _image_files(source: Path) -> Iterable[Path]:
    """Yield supported image files in deterministic order."""

    extensions = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff", ".webp"}
    return sorted(
        path for path in source.iterdir() if path.is_file() and path.suffix.lower() in extensions
    )


def _read_rgb(image_path: Path) -> np.ndarray:
    with Image.open(image_path) as image:
        # Convert explicitly to RGB so palette, grayscale and RGBA files are
        # handled consistently; only the RGB channels define a class.
        return np.asarray(image.convert("RGB"), dtype=np.uint8)


def _extract_binary_mask(rgb: np.ndarray) -> np.ndarray:
    """Return a 2-D uint8 binary mask for selected RGB classes."""

    selected = np.any(
        np.all(rgb[:, :, None, :] == TARGET_COLORS[None, None, :, :], axis=3),
        axis=2,
    )
    return selected.astype(np.uint8) * 255


def extract_binary_mask(image_path: Path) -> np.ndarray:
    """Read an RGB label image and extract the selected classes."""

    return _extract_binary_mask(_read_rgb(image_path))


def extract_masks(
    source: Union[str, Path], output: Optional[Union[str, Path]] = None
) -> Tuple[int, int]:
    """Extract masks and return ``(written_count, skipped_count)``."""

    source_path = Path(source).expanduser()
    if not source_path.is_dir():
        raise NotADirectoryError(f"SOURCE is not a directory: {source_path}")

    output_path = (
        Path(output).expanduser()
        if output is not None
        else source_path.parent / f"{source_path.name}_binary"
    )
    output_path.mkdir(parents=True, exist_ok=True)

    count = 0
    skipped_count = 0
    for image_path in _image_files(source_path):
        # Do not create an output for samples containing Accessories.  This
        # check intentionally happens before extraction/saving.
        rgb = _read_rgb(image_path)
        
        mask = _extract_binary_mask(rgb)
        # Use PNG to avoid lossy compression changing binary values.  Keeping
        # the original stem makes the output easy to pair with its input.
        Image.fromarray(mask).save(output_path / f"{image_path.stem}.png")
        count += 1

    return count, skipped_count


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Extract classes 2 from RGB masks."
    )
    parser.add_argument("SOURCE", type=Path, help="folder containing multi-value RGB masks")
    parser.add_argument(
        "OUTPUT",
        type=Path,
        nargs="?",
        help="output folder (default: SOURCE_binary)",
    )
    args = parser.parse_args()

    try:
        count, skipped_count = extract_masks(args.SOURCE, args.OUTPUT)
    except NotADirectoryError as error:
        parser.error(str(error))

    output = args.OUTPUT or args.SOURCE.parent / f"{args.SOURCE.name}_binary"
    print(
        f"Extracted {count} masks to {output}; "
        f"skipped {skipped_count} masks containing Accessories"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
