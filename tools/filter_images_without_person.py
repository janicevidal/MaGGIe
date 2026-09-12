#!/usr/bin/env python3
"""Copy images whose annotation TXT does not contain the ``person`` class.

Examples
--------
python tools/filter_images_without_person.py \
    --txt-dir "/data/Instance name" \
    --image-dir "/data/images" \
    --output-dir "/data/images_without_person"

TXT and image files are matched by relative path and stem.  For example,
``subdir/a.txt`` is matched to ``subdir/a.jpg`` (or png/jpeg/bmp/tif/webp).
"""

from __future__ import annotations

import argparse
import re
import shutil
from pathlib import Path
from typing import Tuple


IMAGE_EXTENSIONS = (".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp")


def contains_person(text: str) -> bool:
    """Return whether *text* contains the category ``person``.

    A word boundary avoids treating unrelated labels such as
    ``personality`` as ``person``.  Matching is case-insensitive.
    """

    return re.search(r"\bperson\b", text, flags=re.IGNORECASE) is not None


def build_image_index(image_dir: Path):
    """Index images by relative path/stem, retaining the first extension."""

    index = {}
    for image_path in image_dir.rglob("*"):
        if image_path.is_file() and image_path.suffix.lower() in IMAGE_EXTENSIONS:
            key = image_path.relative_to(image_dir).with_suffix("").as_posix()
            index.setdefault(key, image_path)
    return index


def filter_images(txt_dir: Path, image_dir: Path, output_dir: Path) -> Tuple[int, int, int]:
    image_index = build_image_index(image_dir)
    copied = skipped_person = missing_image = 0

    for txt_path in txt_dir.rglob("*.txt"):
        # errors='ignore' keeps one malformed annotation from aborting a batch.
        text = txt_path.read_text(encoding="utf-8", errors="ignore")
        if contains_person(text):
            skipped_person += 1
            continue

        key = txt_path.relative_to(txt_dir).with_suffix("").as_posix()
        image_path = image_index.get(key)
        if image_path is None:
            missing_image += 1
            print(f"[missing image] {txt_path}")
            continue

        destination = output_dir / image_path.relative_to(image_dir)
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(image_path, destination)
        copied += 1

    return copied, skipped_person, missing_image


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--txt-dir", type=Path, required=True,
                        help="directory containing category TXT files")
    parser.add_argument("--image-dir", type=Path, required=True,
                        help="directory containing corresponding images")
    parser.add_argument("--output-dir", type=Path, required=True,
                        help="new directory to receive selected images")
    return parser.parse_args()


def main():
    args = parse_args()
    for path, label in ((args.txt_dir, "TXT"), (args.image_dir, "image")):
        if not path.is_dir():
            raise SystemExit(f"{label} directory does not exist: {path}")

    copied, skipped_person, missing_image = filter_images(
        args.txt_dir, args.image_dir, args.output_dir)
    print(
        f"Copied {copied} image(s); skipped {skipped_person} annotation(s) "
        f"containing person; missing images: {missing_image}."
    )


if __name__ == "__main__":
    main()
