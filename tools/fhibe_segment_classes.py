#!/usr/bin/env python3
"""Print the segmentation classes used by the FHIBE annotations.

Each item in ``segments`` has a ``class_name`` such as ``"20. Upper body
clothes"``.  The number before the full stop is the segmentation class
label; ``num`` is an instance number and is therefore deliberately not used
here.  The script prints one ``<label>\t<class name>`` pair per line and does
not print per-file details or counts to stdout.

Usage:
    python tools/fhibe_segment_classes.py SOURCE
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path
from typing import Iterator, Set, Tuple, Union


# FHIBE currently uses ``"<integer>. <name>"``.  Accepting ``)`` as well
# makes the reader tolerant of the equivalent notation in future exports.
CLASS_NAME_RE = re.compile(r"^\s*(\d+)\s*[.)]\s*(.*?)\s*$")


def _annotation_files(source_root: Path) -> Iterator[Path]:
    """Yield sample annotation JSON files below *source_root*.

    A source export contains additional annotation products (for example
    ``faces_crop_*``).  ``main_annos_*.json`` is the sample annotation used by
    :mod:`fhibe_clothing_gen`; restricting the normal scan to those files
    avoids reading the same segments several times.  The generic fallback is
    useful for small custom/test directories whose files have arbitrary
    names.
    """

    sample_files = sorted(
        path for path in source_root.rglob("main_annos_*.json") if path.is_file()
    )
    if sample_files:
        yield from sample_files
    else:
        yield from sorted(path for path in source_root.rglob("*.json") if path.is_file())


def collect_segment_classes(source_root: Union[str, Path]) -> Set[Tuple[int, str]]:
    """Collect ``(class_number, class_name)`` pairs from all samples.

    Files that do not contain ``subject_annotation`` (or malformed segment
    entries) are ignored.  A malformed JSON file is reported on stderr and
    does not prevent the remaining samples from being processed.
    """

    root = Path(source_root).expanduser()
    if not root.is_dir():
        raise NotADirectoryError(f"SOURCE is not a directory: {root}")

    classes: Set[Tuple[int, str]] = set()
    for json_path in _annotation_files(root):
        try:
            with json_path.open("r", encoding="utf-8") as file:
                data = json.load(file)
        except (OSError, json.JSONDecodeError) as error:
            print(f"warning: could not read {json_path}: {error}", file=sys.stderr)
            continue

        subjects = data.get("subject_annotation", []) if isinstance(data, dict) else []
        if not isinstance(subjects, list):
            continue
        for subject in subjects:
            if not isinstance(subject, dict):
                continue
            segments = subject.get("segments", [])
            if not isinstance(segments, list):
                continue
            for segment in segments:
                if not isinstance(segment, dict):
                    continue
                class_name = segment.get("class_name")
                if not isinstance(class_name, str):
                    continue
                match = CLASS_NAME_RE.match(class_name)
                if match is None:
                    print(
                        f"warning: class_name has no numeric label in {json_path}: "
                        f"{class_name!r}",
                        file=sys.stderr,
                    )
                    continue
                classes.add((int(match.group(1)), match.group(2).strip()))

    return classes


def print_segment_classes(classes: Set[Tuple[int, str]]) -> None:
    """Print classes as ``label<TAB>name``, sorted by their numeric label."""

    for class_number, class_name in sorted(classes, key=lambda item: (item[0], item[1])):
        print(f"{class_number}\t{class_name}")


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Print all segmentation class labels found below SOURCE."
    )
    parser.add_argument("SOURCE", type=Path, help="FHIBE source directory")
    args = parser.parse_args()

    try:
        classes = collect_segment_classes(args.SOURCE)
    except NotADirectoryError as error:
        print(f"error: {error}", file=sys.stderr)
        return 1

    print_segment_classes(classes)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

#
# 0       Face skin
# 1       Upper body skin
# 2       Left arm skin
# 3       Right arm skin
# 4       Left leg skin
# 5       Right leg skin
# 6       Head hair
# 7       Left eyebrow
# 8       Right eyebrow
# 9       Left eye
# 10      Right eye
# 11      Nose
# 12      Upper lip
# 13      Lower lip
# 14      Inner mouth
# 15      Left shoe
# 16      Right shoe
# 17      Headwear
# 18      Mask
# 19      Eyewear
# 20      Upper body clothes
# 21      Lower body clothes
# 22      Full body clothes
# 23      Sock or legwarmer
# 24      Neckwear
# 25      Bag
# 26      Glove
# 27      Jewelry or timepiece
