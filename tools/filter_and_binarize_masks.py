#!/usr/bin/env python3
"""Copy masks corresponding to images without modifying their contents.

For every image in ``--image-dir``, the script looks for a mask with the same
relative path and file stem in ``--mask-dir``.  A matched mask is copied as-is:
its format, extension, channels and pixel values are preserved.

Example:
    python tools/filter_and_binarize_masks.py \
        --image-dir /data/images \
        --mask-dir /data/masks \
        --output-dir /data/masks_binary
"""

from __future__ import annotations

import argparse
import shutil
from pathlib import Path
from typing import Dict, Iterable, Optional


IMAGE_EXTENSIONS = {
    ".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"
}


def index_masks(mask_dir: Path) -> Dict[str, Optional[Path]]:
    """Index masks by relative path and stem, allowing different extensions."""
    masks = {}
    for path in mask_dir.rglob("*"):
        if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS:
            key = path.relative_to(mask_dir).with_suffix("").as_posix()
            masks.setdefault(key, path)
            # Also permit flat image/mask directories where only the stem is
            # shared.  A duplicate stem is marked ambiguous and ignored.
            stem_key = "__stem__:" + path.stem
            if stem_key in masks:
                masks[stem_key] = None
            else:
                masks[stem_key] = path
    return masks


def iter_images(image_dir: Path) -> Iterable[Path]:
    for path in image_dir.rglob("*"):
        if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS:
            yield path


def find_mask(image_path: Path, image_dir: Path,
              masks: Dict[str, Optional[Path]]) -> Optional[Path]:
    key = image_path.relative_to(image_dir).with_suffix("").as_posix()
    result = masks.get(key)
    if result is not None:
        return result
    return masks.get("__stem__:" + image_path.stem)


def copy_mask(mask_path: Path, output_path: Path) -> None:
    """Copy a mask without decoding or changing it."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(mask_path, output_path)


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--image-dir", type=Path, required=True,
                        help="folder containing source images")
    parser.add_argument("--mask-dir", type=Path, required=True,
                        help="folder containing masks")
    parser.add_argument("--output-dir", type=Path, required=True,
                        help="folder for selected binary masks")
    return parser.parse_args()


def main():
    args = parse_args()
    if not args.image_dir.is_dir():
        raise SystemExit(f"Image directory does not exist: {args.image_dir}")
    if not args.mask_dir.is_dir():
        raise SystemExit(f"Mask directory does not exist: {args.mask_dir}")

    masks = index_masks(args.mask_dir)
    matched = missing = 0
    for image_path in iter_images(args.image_dir):
        mask_path = find_mask(image_path, args.image_dir, masks)
        if mask_path is None:
            missing += 1
            print(f"[missing mask] {image_path}")
            continue

        # Use the mask's own relative path, rather than the image extension,
        # so the copied file remains byte-for-byte identical.
        relative = mask_path.relative_to(args.mask_dir)
        copy_mask(mask_path, args.output_dir / relative)
        matched += 1

    print(f"Copied {matched} mask(s) without modification; missing masks: {missing}.")


if __name__ == "__main__":
    main()
