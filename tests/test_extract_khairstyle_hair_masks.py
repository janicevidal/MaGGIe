"""Tests for tools/extract_khairstyle_hair_masks.py."""

import json
import sys
from pathlib import Path

import numpy as np
import pytest
from PIL import Image


sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "tools"))

from extract_khairstyle_hair_masks import (  # noqa: E402
    export_dataset,
    main,
    parse_hair_polygons,
    render_hair_mask,
    resolve_image_path,
)


def _polygon(points):
    """Encode points the way K-Hairstyle stores them: JSON inside a string."""

    ring = [{"x": float(x), "y": float(y)} for x, y in points]
    return json.dumps([ring])


def _capture(tmp_path, name="CP000001-001", points=((1, 1), (4, 1), (4, 4), (1, 4)),
             polygon1=None, with_image=True):
    """Write a miniature capture folder and return its annotation path."""

    capture = tmp_path / "0001.style" / f"0126.{name.split('-')[0]}"
    capture.mkdir(parents=True, exist_ok=True)
    if with_image:
        Image.new("RGB", (8, 8), (20, 20, 20)).save(capture / f"{name}.jpg")
    annotation = {
        "filename": f"{name}.jpg",
        "polygon1": _polygon(points) if polygon1 is None else polygon1,
        "polygon2": "[]",
    }
    path = capture / f"{name.replace('-', '_')}.json"
    path.write_text(json.dumps(annotation), encoding="utf-8")
    return path


def test_cli_exports_pairs_and_defaults_output_beside_input(tmp_path, monkeypatch, capsys):
    _capture(tmp_path)
    source = tmp_path / "0001.style"

    monkeypatch.setattr("sys.argv", ["export", str(source), "--workers", "1", "--no-progress"])
    assert main() == 0

    assert (tmp_path / "0001.style_hair_export" / "masks" / "CP000001-001.png").is_file()
    assert "Exported 1 image/mask pair(s)" in capsys.readouterr().out


def test_render_hair_mask_marks_the_polygon_interior():
    mask = render_hair_mask([[(1, 1), (4, 1), (4, 4), (1, 4)]], (8, 8))

    assert mask.dtype == np.uint8
    assert set(np.unique(mask).tolist()) <= {0, 255}
    assert mask[2, 2] == 255  # inside the square
    assert mask[6, 6] == 0  # outside it


def test_multiple_rings_are_merged_into_one_mask():
    rings = [[(0, 0), (2, 0), (2, 2), (0, 2)], [(5, 5), (7, 5), (7, 7), (5, 7)]]

    mask = render_hair_mask(rings, (8, 8))

    assert mask[1, 1] == 255 and mask[6, 6] == 255
    assert mask[1, 6] == 0


def test_parse_hair_polygons_handles_empty_and_short_rings():
    assert parse_hair_polygons({"polygon1": ""}) == []
    assert parse_hair_polygons({"polygon1": "[]"}) == []
    # Two-point rings cannot enclose an area and must be dropped.
    assert parse_hair_polygons({"polygon1": json.dumps([[[{"x": 0, "y": 0}, {"x": 1, "y": 1}]]])}) == []
    assert parse_hair_polygons({"polygon1": _polygon(((0, 0), (1, 0), (1, 1)))}) == [
        [(0.0, 0.0), (1.0, 0.0), (1.0, 1.0)]
    ]


def test_export_writes_matching_image_and_mask(tmp_path):
    _capture(tmp_path)
    output = tmp_path / "out"

    stats = export_dataset(tmp_path / "0001.style", output, workers=1)

    assert stats.written == 1 and stats.skipped_no_hair == 0
    assert [p.name for p in (output / "images").iterdir()] == ["CP000001-001.jpg"]
    mask = np.asarray(Image.open(output / "masks" / "CP000001-001.png"))
    assert mask[2, 2] == 255 and mask[6, 6] == 0


def test_empty_hair_mask_skips_the_pair(tmp_path):
    _capture(tmp_path, polygon1="[]")
    output = tmp_path / "out"

    stats = export_dataset(tmp_path / "0001.style", output, workers=1)

    assert stats.written == 0 and stats.skipped_no_hair == 1
    assert list((output / "images").iterdir()) == []
    assert list((output / "masks").iterdir()) == []


def test_missing_photo_is_counted_and_skipped(tmp_path):
    _capture(tmp_path, with_image=False)
    output = tmp_path / "out"

    stats = export_dataset(tmp_path / "0001.style", output, workers=1)

    assert stats.written == 0 and stats.skipped_missing_image == 1


def test_image_is_resolved_from_the_underscore_stem_when_filename_is_wrong(tmp_path):
    annotation = _capture(tmp_path)
    document = json.loads(annotation.read_text(encoding="utf-8"))
    document["filename"] = "CP000001.other-name.jpg"
    annotation.write_text(json.dumps(document), encoding="utf-8")

    resolved = resolve_image_path(annotation, document)

    assert resolved is not None and resolved.name == "CP000001-001.jpg"


def test_skip_existing_keeps_finished_pairs(tmp_path):
    _capture(tmp_path)
    output = tmp_path / "out"

    export_dataset(tmp_path / "0001.style", output, workers=1)
    before = (output / "masks" / "CP000001-001.png").stat().st_mtime_ns
    stats = export_dataset(tmp_path / "0001.style", output, workers=1, skip_existing=True)

    assert stats.skipped_existing == 1 and stats.written == 0
    assert (output / "masks" / "CP000001-001.png").stat().st_mtime_ns == before


def test_parallel_workers_match_sequential_output(tmp_path):
    for index in range(12):
        _capture(tmp_path, name=f"CP00000{index}-001")

    sequential, parallel = tmp_path / "seq", tmp_path / "par"
    assert export_dataset(tmp_path / "0001.style", sequential, workers=1).written == 12
    assert export_dataset(tmp_path / "0001.style", parallel, workers=4).written == 12

    for mask_name in sorted(p.name for p in (sequential / "masks").iterdir()):
        first = np.asarray(Image.open(sequential / "masks" / mask_name))
        second = np.asarray(Image.open(parallel / "masks" / mask_name))
        assert np.array_equal(first, second)
    assert sorted(p.name for p in (parallel / "images").iterdir()) == sorted(
        p.name for p in (sequential / "images").iterdir()
    )


def test_missing_input_directory_raises(tmp_path):
    with pytest.raises(NotADirectoryError):
        export_dataset(tmp_path / "nope", tmp_path / "out")
