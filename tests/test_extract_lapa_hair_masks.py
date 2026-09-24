"""Tests for tools/extract_lapa_hair_masks.py."""

import sys
from pathlib import Path

import numpy as np
import pytest
from PIL import Image


sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "tools"))

from extract_lapa_hair_masks import (  # noqa: E402
    HAIR_CLASS,
    extract_hair_masks,
    hair_mask_from_label,
)


def test_hair_mask_is_binary_and_isolates_class_10():
    label = np.array([[0, 1, HAIR_CLASS], [HAIR_CLASS, 9, 4]], dtype=np.uint8)

    mask = hair_mask_from_label(label)

    assert mask.dtype == np.uint8
    assert mask.tolist() == [[0, 0, 255], [255, 0, 0]]


def test_empty_hair_masks_are_skipped(tmp_path):
    source = tmp_path / "labels"
    source.mkdir()
    output = tmp_path / "hair"
    with_hair = np.full((3, 3), 1, dtype=np.uint8)
    with_hair[0, 0] = HAIR_CLASS
    Image.fromarray(with_hair).save(source / "with_hair.png")
    Image.fromarray(np.full((3, 3), 1, dtype=np.uint8)).save(source / "no_hair.png")

    written, skipped = extract_hair_masks(source, output)

    assert (written, skipped) == (1, 1)
    assert [p.name for p in sorted(output.iterdir())] == ["with_hair.png"]
    assert np.array(Image.open(output / "with_hair.png")).max() == 255


def test_rejects_non_index_labels(tmp_path):
    source = tmp_path / "labels"
    source.mkdir()
    Image.fromarray(np.zeros((3, 3, 3), dtype=np.uint8)).save(source / "rgb.png")

    with pytest.raises(ValueError):
        extract_hair_masks(source, tmp_path / "hair")


def test_missing_source_raises(tmp_path):
    with pytest.raises(NotADirectoryError):
        extract_hair_masks(tmp_path / "does_not_exist", tmp_path / "hair")
