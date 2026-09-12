#!/usr/bin/env python3
"""Export image/mask pairs whose JSONL classification has no person.

``classification.jsonl`` is expected to contain one JSON object per line with
an ``image_id`` field and either ``contains_person`` or ``labels``.  Records
classified as person-only *or* person+animal are excluded; animal-only and
neither-person-nor-animal records are exported.

Example:
    python tools/export_non_person_pairs.py \
        --classification-jsonl /data/classification.jsonl \
        --image-dir /data/all/images \
        --mask-dir /data/all/masks \
        --output-root /data/non_person

The output contains ``output-root/images`` and ``output-root/masks``.  Files
are matched by stem, so image and mask extensions may differ.
"""

from __future__ import annotations

import argparse
import json
import logging
import shutil
from pathlib import Path
from typing import Dict, Optional, Set, Tuple


LOGGER = logging.getLogger(__name__)
IMAGE_EXTENSIONS = {
    ".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"
}


def normalize_id(value) -> str:
    """Normalize JSONL IDs such as ``foo.jpg`` or ``sub/foo`` to a stem key."""
    value = str(value).replace("\\", "/").strip()
    return Path(value).with_suffix("").as_posix()


def result_contains_person(result: dict) -> bool:
    if "contains_person" in result:
        value = result["contains_person"]
        if isinstance(value, str):
            return value.strip().lower() in {"1", "true", "yes", "person", "人"}
        return bool(value)
    labels = result.get("labels") or []
    if isinstance(labels, str):
        labels = [labels]
    labels = {str(label).strip().lower() for label in labels}
    # English labels are produced by filter_s3od_captions. Chinese aliases are
    # accepted for manually generated classification files.
    return bool(labels & {"person", "human", "people", "人"})


def load_non_person_ids(path: Path) -> Tuple[Set[str], int, int]:
    """Return selected IDs, total valid records, and excluded-person count."""
    selected = set()
    total = excluded = 0
    with path.open("r", encoding="utf-8") as stream:
        for line_number, line in enumerate(stream, 1):
            if not line.strip():
                continue
            try:
                result = json.loads(line)
                image_id = result["image_id"]
            except (json.JSONDecodeError, KeyError, TypeError) as error:
                raise ValueError(
                    f"Invalid classification record at line {line_number}: {error}"
                ) from error
            total += 1
            key = normalize_id(image_id)
            if result_contains_person(result):
                excluded += 1
            else:
                selected.add(key)
    return selected, total, excluded


def index_files(directory: Path) -> Dict[str, Optional[Path]]:
    """Index files by relative stem, with a stem fallback for flat folders."""
    indexed = {}
    for path in directory.rglob("*"):
        if not path.is_file() or path.suffix.lower() not in IMAGE_EXTENSIONS:
            continue
        relative_key = normalize_id(path.relative_to(directory))
        indexed.setdefault(relative_key, path)
        # A collision is ambiguous and is removed from the fallback index.
        stem_key = "__stem__:" + path.stem
        if stem_key in indexed:
            indexed[stem_key] = None
        else:
            indexed[stem_key] = path
    return indexed


def lookup(index: Dict[str, Optional[Path]], key: str) -> Optional[Path]:
    path = index.get(key)
    if path is not None:
        return path
    return index.get("__stem__:" + Path(key).name)


def copy_pair(image_path: Path, mask_path: Path, image_dir: Path,
              mask_dir: Path, output_root: Path, overwrite: bool) -> bool:
    image_destination = output_root / "images" / image_path.relative_to(image_dir)
    mask_destination = output_root / "masks" / mask_path.relative_to(mask_dir)
    for source, destination in ((image_path, image_destination),
                                (mask_path, mask_destination)):
        destination.parent.mkdir(parents=True, exist_ok=True)
        if destination.exists() and not overwrite:
            continue
        temporary = destination.with_suffix(destination.suffix + ".tmp")
        shutil.copy2(source, temporary)
        temporary.replace(destination)
    return True


def export_non_person_pairs(classification_jsonl: Path, image_dir: Path,
                            mask_dir: Path, output_root: Path,
                            overwrite: bool = False):
    selected, total, excluded = load_non_person_ids(classification_jsonl)
    image_index = index_files(image_dir)
    mask_index = index_files(mask_dir)
    copied = missing_images = missing_masks = 0

    for key in sorted(selected):
        image_path = lookup(image_index, key)
        if image_path is None:
            missing_images += 1
            LOGGER.warning("Image not found for classification ID: %s", key)
            continue
        mask_path = lookup(mask_index, key)
        if mask_path is None:
            missing_masks += 1
            LOGGER.warning("Mask not found for image: %s", image_path)
            continue
        copy_pair(image_path, mask_path, image_dir, mask_dir, output_root,
                  overwrite)
        copied += 1

    LOGGER.info(
        "Classification records=%d; excluded person/person+animal=%d; "
        "selected non-person=%d; copied pairs=%d; missing images=%d; "
        "missing masks=%d",
        total, excluded, len(selected), copied, missing_images, missing_masks)
    return copied, missing_images, missing_masks


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--classification-jsonl", type=Path, required=True)
    parser.add_argument("--image-dir", type=Path, required=True)
    parser.add_argument("--mask-dir", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--log-level", default="INFO",
                        choices=["DEBUG", "INFO", "WARNING", "ERROR"])
    return parser.parse_args()


def main():
    args = parse_args()
    logging.basicConfig(
        level=getattr(logging, args.log_level),
        format="%(asctime)s - %(levelname)s - %(message)s")
    for path, label in ((args.classification_jsonl, "classification JSONL"),
                        (args.image_dir, "image directory"),
                        (args.mask_dir, "mask directory")):
        if not path.exists():
            raise SystemExit(f"{label} does not exist: {path}")
    copied, missing_images, missing_masks = export_non_person_pairs(
        args.classification_jsonl, args.image_dir, args.mask_dir,
        args.output_root, args.overwrite)
    if missing_images or missing_masks:
        raise SystemExit(1)
    LOGGER.info("Finished: copied %d image/mask pair(s)", copied)


if __name__ == "__main__":
    main()
