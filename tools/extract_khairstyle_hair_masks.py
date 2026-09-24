#!/usr/bin/env python3
"""Build paired image/hair-mask folders from K-Hairstyle annotations.

K-Hairstyle stores one JSON per photo.  The hair region is annotated as a
polygon and saved in the ``polygon1`` field; ``polygon2`` holds the blurred
face region (frequently the empty string ``"[]"``).  Both fields are strings
containing JSON, and their coordinates are pixel coordinates in the *stored*
image frame, i.e. the frame Pillow decodes without applying EXIF rotation
(verified by overlaying rendered polygons on the photos):

    "polygon1": "[[{\"x\": 228.4, \"y\": 58.2}, ...], ...]"
                ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
                list of rings; each ring is a closed polygon

This script walks a K-Hairstyle set directory and writes two folders:

    OUTPUT/images/<photo name>.<ext>   identical copy of the source photo
    OUTPUT/masks/<photo name>.png      binary hair mask, 255 = hair, 0 = rest

Usage:
    python tools/extract_khairstyle_hair_masks.py \\
        /data/xiaoshuai/hair_segmentation/dataset/K-Hairstyle/0002.mqset \\
        --output-dir /data/xiaoshuai/hair_segmentation/dataset/K-Hairstyle/0002_hair

Samples are skipped, never written half-way, when the annotation carries no
hair polygon, when the rendered mask would be empty, or when the photo is
missing from the set (a couple of capture folders ship annotations only).
Photo names are unique across the whole set, so a flat output layout cannot
collide; ``--skip-existing`` makes an interrupted run resumable.

Performance:
    Rasterising with Pillow is fast enough that the run is dominated by
    annotation parsing and photo copying, both of which scale across worker
    processes (``--workers``, default ``min(cpu_count, 32)``).  Use
    ``--image-mode hardlink`` to avoid duplicating photo bytes when OUTPUT
    lives on the same filesystem as the source set.
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
from collections import Counter
from concurrent.futures import ProcessPoolExecutor
from functools import partial
from pathlib import Path
from typing import Dict, List, NamedTuple, Optional, Sequence, Tuple, Union

import numpy as np
from PIL import Image, ImageDraw

try:  # Progress reporting is optional; the extraction itself does not need it.
    from tqdm import tqdm
except ImportError:  # pragma: no cover - tqdm is present in the project envs.
    tqdm = None


HAIR_POLYGON_KEY = "polygon1"
FACE_POLYGON_KEY = "polygon2"
FOREGROUND_VALUE = 255
DEFAULT_COMPRESSION_LEVEL = 1
MAX_DEFAULT_WORKERS = 32
IMAGE_EXTENSIONS = (".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp")

# Per-sample outcomes, kept as small ints because they travel back from workers.
STATUS_WRITTEN = 0
STATUS_NO_HAIR = 1
STATUS_MISSING_IMAGE = 2
STATUS_EXISTING = 3


class ExportStats(NamedTuple):
    """Outcome counts for one export run."""

    written: int
    skipped_no_hair: int
    skipped_missing_image: int
    skipped_existing: int
    rotated_images: int


class _NoProgress:
    """Drop-in stand-in for ``tqdm`` when the package is unavailable."""

    def update(self, _increment: int) -> None:
        pass

    def close(self) -> None:
        pass


def default_workers() -> int:
    """Return a sensible process count for the current machine."""

    return min(os.cpu_count() or 1, MAX_DEFAULT_WORKERS)


def find_annotation_files(input_dir: Union[str, Path]) -> List[Path]:
    """Return every annotation JSON below ``input_dir`` in deterministic order."""

    root = Path(input_dir).expanduser()
    if not root.is_dir():
        raise NotADirectoryError(f"INPUT is not a directory: {root}")
    return sorted(path for path in root.rglob("*.json") if path.is_file())


def load_annotation(annotation_path: Union[str, Path]) -> Dict[str, object]:
    """Read one K-Hairstyle annotation JSON."""

    with open(annotation_path, "rb") as handle:
        return json.load(handle)


def parse_hair_polygons(
    annotation: Dict[str, object],
) -> List[List[Tuple[float, float]]]:
    """Return the hair polygons of one annotation as rings of ``(x, y)`` points.

    The polygon fields are JSON payloads stored as strings, and a sample may
    carry several rings (disconnected hair regions) or none at all.  Rings with
    fewer than three points cannot enclose an area and are dropped.
    """

    payload = annotation.get(HAIR_POLYGON_KEY) or ""
    if not payload.strip():
        return []
    rings = json.loads(payload)
    return [
        [(float(point["x"]), float(point["y"])) for point in ring]
        for ring in rings
        if len(ring) >= 3
    ]


def read_hair_polygons(
    annotation_path: Union[str, Path],
) -> List[List[Tuple[float, float]]]:
    """Read one annotation file and return its hair polygons."""

    return parse_hair_polygons(load_annotation(annotation_path))


def render_hair_mask(
    rings: Sequence[Sequence[Tuple[float, float]]], size: Tuple[int, int]
) -> np.ndarray:
    """Rasterise hair rings into a 2-D uint8 mask of shape ``(height, width)``."""

    mask = Image.new("L", size, 0)
    drawer = ImageDraw.Draw(mask)
    for ring in rings:
        drawer.polygon(list(ring), fill=FOREGROUND_VALUE)
    return np.asarray(mask)


def resolve_image_path(
    annotation_path: Path, annotation: Dict[str, object]
) -> Optional[Path]:
    """Return the photo referenced by an annotation, or ``None`` when absent.

    Photos normally sit next to their JSON with the name stored in ``filename``
    (``CP032677_001.json`` -> ``CP032677-001.jpg``); a few folders use the
    underscore/dash variant or a different image extension, hence the fallback.
    """

    directory = annotation_path.parent
    declared = annotation.get("filename")
    if isinstance(declared, str) and declared:
        candidate = directory / declared
        if candidate.is_file():
            return candidate
        stem, _, _ = declared.rpartition(".")
        variants = {stem, annotation_path.stem, annotation_path.stem.replace("_", "-")}
    else:
        variants = {annotation_path.stem, annotation_path.stem.replace("_", "-")}

    for variant in sorted(v for v in variants if v):
        for extension in IMAGE_EXTENSIONS:
            candidate = directory / f"{variant}{extension}"
            if candidate.is_file():
                return candidate
    return None


def link_or_copy_image(source: Path, target: Path, image_mode: str) -> None:
    """Place the photo into the image folder, honouring ``image_mode``.

    Hard and symbolic links are rolled back to a real copy when the filesystem
    refuses them (cross-device links, unsupported mount options), so a partial
    pair is never left behind.
    """

    if image_mode == "copy":
        shutil.copyfile(source, target)
        return

    try:
        if image_mode == "hardlink":
            os.link(source, target)
        else:
            os.symlink(source, target)
    except OSError:
        target.unlink(missing_ok=True)
        shutil.copyfile(source, target)


def process_annotation(
    annotation_path: Path,
    images_dir: Path,
    masks_dir: Path,
    compression_level: int = DEFAULT_COMPRESSION_LEVEL,
    image_mode: str = "copy",
    skip_existing: bool = False,
) -> Tuple[int, bool]:
    """Export one annotation, returning ``(status, image_has_exif_rotation)``.

    Runs inside worker processes, so every argument is picklable and the
    function stays at module level.  The photo and its mask are only written
    together, and never written at all when the hair mask would be empty.
    """

    annotation = load_annotation(annotation_path)

    image_path = resolve_image_path(annotation_path, annotation)
    if image_path is None:
        return STATUS_MISSING_IMAGE, False

    mask_path = masks_dir / f"{image_path.stem}.png"
    image_target = images_dir / image_path.name
    if skip_existing and mask_path.exists() and image_target.exists():
        return STATUS_EXISTING, False

    with Image.open(image_path) as image:
        size = image.size
        rotated = image.getexif().get(0x0112, 1) not in (1, None)

    mask = render_hair_mask(parse_hair_polygons(annotation), size)
    if not mask.any():
        return STATUS_NO_HAIR, False

    # PNG keeps the mask lossless; the shared stem keeps the pair easy to match.
    Image.fromarray(mask).save(mask_path, compress_level=compression_level)
    link_or_copy_image(image_path, image_target, image_mode)
    return STATUS_WRITTEN, rotated


def export_dataset(
    input_dir: Union[str, Path],
    output_dir: Union[str, Path],
    image_subdir: str = "images",
    mask_subdir: str = "masks",
    workers: int = 0,
    limit: Optional[int] = None,
    compression_level: int = DEFAULT_COMPRESSION_LEVEL,
    image_mode: str = "copy",
    skip_existing: bool = False,
    show_progress: bool = True,
) -> ExportStats:
    """Export every annotation below ``input_dir`` into ``output_dir``."""

    annotations = find_annotation_files(input_dir)
    if limit is not None:
        annotations = annotations[:limit]

    images_dir = Path(output_dir).expanduser() / image_subdir
    masks_dir = Path(output_dir).expanduser() / mask_subdir
    images_dir.mkdir(parents=True, exist_ok=True)
    masks_dir.mkdir(parents=True, exist_ok=True)

    task = partial(
        process_annotation,
        images_dir=images_dir,
        masks_dir=masks_dir,
        compression_level=compression_level,
        image_mode=image_mode,
        skip_existing=skip_existing,
    )
    worker_count = workers if workers > 0 else default_workers()
    if worker_count > 1 and len(annotations) > 1:
        # Big chunks amortize inter-process overhead on many small annotations.
        chunksize = max(1, len(annotations) // (worker_count * 8))
        executor = ProcessPoolExecutor(max_workers=worker_count)
        results = executor.map(task, annotations, chunksize=chunksize)
    else:
        executor = None
        results = map(task, annotations)

    counts: Counter = Counter()
    rotated = 0
    progress = (
        tqdm(total=len(annotations), unit="pair", disable=not show_progress)
        if tqdm is not None
        else _NoProgress()
    )
    try:
        for status, is_rotated in results:
            counts[status] += 1
            rotated += int(is_rotated)
            progress.update(1)
    finally:
        progress.close()
        if executor is not None:
            executor.shutdown(wait=True)

    return ExportStats(
        written=counts[STATUS_WRITTEN],
        skipped_no_hair=counts[STATUS_NO_HAIR],
        skipped_missing_image=counts[STATUS_MISSING_IMAGE],
        skipped_existing=counts[STATUS_EXISTING],
        rotated_images=rotated,
    )


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Export K-Hairstyle photos and their binary hair masks."
    )
    parser.add_argument(
        "input_dir",
        metavar="INPUT",
        type=Path,
        help="K-Hairstyle set folder holding the annotation JSONs and photos",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        help="output folder (default: <INPUT>_hair_export)",
    )
    parser.add_argument("--image-subdir", default="images", help="photo folder inside --output-dir")
    parser.add_argument("--mask-subdir", default="masks", help="mask folder inside --output-dir")
    parser.add_argument(
        "-j",
        "--workers",
        type=int,
        default=0,
        help=f"worker processes (default: min(cpu_count, {MAX_DEFAULT_WORKERS}); 1 = no pool)",
    )
    parser.add_argument(
        "--limit", type=int, default=None, help="export at most N samples (smoke tests)"
    )
    parser.add_argument(
        "--image-mode",
        choices=("copy", "hardlink", "symlink"),
        default="copy",
        help="how photos enter the image folder (default: copy)",
    )
    parser.add_argument(
        "--compress-level",
        type=int,
        default=DEFAULT_COMPRESSION_LEVEL,
        choices=range(0, 10),
        metavar="0-9",
        help=f"PNG compression level (default: {DEFAULT_COMPRESSION_LEVEL})",
    )
    parser.add_argument(
        "--skip-existing",
        action="store_true",
        help="keep pairs that already exist (resume an interrupted run)",
    )
    parser.add_argument(
        "--no-progress", action="store_true", help="disable the progress bar"
    )
    args = parser.parse_args()

    input_dir = args.input_dir.expanduser()
    output_dir = args.output_dir or input_dir.with_name(
        f"{input_dir.name}_hair_export"
    )
    try:
        stats = export_dataset(
            input_dir,
            output_dir,
            image_subdir=args.image_subdir,
            mask_subdir=args.mask_subdir,
            workers=args.workers,
            limit=args.limit,
            compression_level=args.compress_level,
            image_mode=args.image_mode,
            skip_existing=args.skip_existing,
            show_progress=not args.no_progress,
        )
    except NotADirectoryError as error:
        parser.error(str(error))

    print(
        f"Exported {stats.written} image/mask pair(s) to {output_dir} "
        f"({args.image_subdir}/ + {args.mask_subdir}/); "
        f"skipped {stats.skipped_no_hair} without hair, "
        f"{stats.skipped_missing_image} missing photo(s), "
        f"{stats.skipped_existing} already present"
    )
    if stats.rotated_images:
        print(
            f"Note: {stats.rotated_images} photo(s) carry an EXIF orientation tag; "
            "masks follow the stored pixel frame, matching the copied photos."
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
