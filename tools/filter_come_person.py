"""Filter COME images with a person detector.

Images with detections are written with their bounding boxes overlaid. Images
without detections are copied unchanged to a separate directory.
"""

import argparse
import glob
import os
import shutil

import cv2
import numpy as np
from tqdm import tqdm

from rtmlib import YOLOX


DEFAULT_INPUT_DIR = (
    "/data/xiaoshuai/human_matting/dataset/OpenDataLab___COME15K/raw/"
    "non_person_images/"
)
DEFAULT_DETECTED_DIR = (
    "/data/xiaoshuai/human_matting/dataset/OpenDataLab___COME15K/raw/"
    "person_detected/"
)
DEFAULT_NO_PERSON_DIR = (
    "/data/xiaoshuai/human_matting/dataset/OpenDataLab___COME15K/raw/"
    "non_person_images_filtered/"
)
DEFAULT_ONNX_MODEL = (
    "/data/xiaoshuai/facial_lanmark/face_detect/weights/yolox_l.onnx"
)
PERSON_CLASS_ID = 0

IMAGE_PATTERNS = ("*.png", "*.jpg", "*.jpeg", "*.JPG", "*.JPEG", "*.PNG")


def collect_image_names(input_dir):
    """Return supported images once, sorted by path."""
    image_names = {
        image_name
        for pattern in IMAGE_PATTERNS
        for image_name in glob.glob(os.path.join(input_dir, pattern))
    }
    return sorted(image_names)


def normalize_detections(detections, score_threshold=0.0):
    """Normalize RTMLib YOLOX output.

    With ``mode='multiclass'``, RTMLib returns ``(boxes, class_ids)``. It
    deliberately does not return scores. With ``mode='human'`` it returns
    boxes only, but that mode does *not* filter the COCO classes despite its
    name. For compatibility, this helper accepts both forms and returns
    ``(boxes, class_ids, scores)``. Missing scores are represented by NaN.
    """
    scores = None
    class_ids = None
    if isinstance(detections, (tuple, list)):
        # RTMLib YOLOX(mode='multiclass') returns (boxes, class_ids).
        first = np.asarray(detections[0])
        if (len(detections) == 2 and
                (first.ndim >= 2 or first.size == 0)):
            boxes = first
            class_ids = np.asarray(detections[1]).reshape(-1)
            if boxes.ndim == 2 and boxes.shape[1] >= 4:
                boxes = boxes[:, :4]
                if class_ids.size != boxes.shape[0]:
                    raise ValueError(
                        "YOLOX boxes and class IDs have different lengths: "
                        f"{boxes.shape[0]} and {class_ids.size}")
        else:
            boxes = np.asarray(detections)
    else:
        boxes = np.asarray(detections)

    if boxes.size == 0:
        return (
            np.empty((0, 4), dtype=np.float32),
            np.empty((0,), dtype=np.int64),
            np.empty((0,), dtype=np.float32),
        )
    if boxes.ndim == 1:
        boxes = boxes.reshape(1, -1)
    if boxes.ndim != 2 or boxes.shape[1] < 4:
        raise ValueError(
            "Expected YOLOX detections with at least four coordinates, got "
            f"shape {boxes.shape}")

    boxes = boxes.astype(np.float32, copy=False)
    if class_ids is None:
        # Compatibility with mode='human' or a plain Nx4 result. Such a
        # result cannot identify non-person classes, so callers should prefer
        # mode='multiclass' when using a COCO YOLOX model.
        class_ids = np.full(boxes.shape[0], PERSON_CLASS_ID, dtype=np.int64)
        if boxes.shape[1] >= 5:
            scores = boxes[:, 4].copy()
        boxes = boxes[:, :4]
    else:
        class_ids = class_ids.astype(np.int64, copy=False)
        scores = np.full(boxes.shape[0], np.nan, dtype=np.float32)

    if scores is None:
        scores = np.full(boxes.shape[0], np.nan, dtype=np.float32)

    valid = np.isfinite(boxes).all(axis=1)
    valid &= boxes[:, 2] > boxes[:, 0]
    valid &= boxes[:, 3] > boxes[:, 1]
    if scores is not None:
        valid &= np.isnan(scores) | (scores >= float(score_threshold))
    return boxes[valid], class_ids[valid], scores[valid]


def draw_detections(image, detections, score_threshold=0.0):
    """Draw only person detections and return the kept boxes."""
    boxes, class_ids, scores = normalize_detections(
        detections, score_threshold)
    person = class_ids == PERSON_CLASS_ID
    boxes = boxes[person]
    scores = scores[person]
    annotated = image.copy()
    height, width = annotated.shape[:2]
    for (x1, y1, x2, y2), score in zip(boxes, scores):
        x1 = int(np.clip(round(float(x1)), 0, width - 1))
        y1 = int(np.clip(round(float(y1)), 0, height - 1))
        x2 = int(np.clip(round(float(x2)), 0, width - 1))
        y2 = int(np.clip(round(float(y2)), 0, height - 1))
        if x2 <= x1 or y2 <= y1:
            continue
        cv2.rectangle(annotated, (x1, y1), (x2, y2), (0, 255, 0), 2)
        text = "person"
        if np.isfinite(score):
            text += f" {float(score):.2f}"
        cv2.putText(annotated, text, (x1, max(0, y1 - 6)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 255, 0), 2,
                    cv2.LINE_AA)
    return annotated, boxes


def filter_images(
        input_dir,
        detected_dir,
        no_person_dir,
        onnx_model,
        device="cuda",
        backend="onnxruntime",
        score_threshold=0.0):
    """Detect, annotate, and separate images from ``input_dir``."""
    os.makedirs(detected_dir, exist_ok=True)
    os.makedirs(no_person_dir, exist_ok=True)

    human_detector = YOLOX(
        onnx_model=onnx_model,
        # The default ``mode='human'`` in RTMLib returns boxes for every
        # surviving COCO class. Request class IDs so we can keep person (0)
        # only; otherwise a chair/car can be mistaken for a person.
        mode="multiclass",
        score_thr=score_threshold,
        backend=backend,
        device=device,
    )
    image_names = collect_image_names(input_dir)
    print(f"There are {len(image_names)} images.")

    detected_count = 0
    no_person_count = 0
    failed_count = 0
    for image_name in tqdm(image_names):
        input_image = cv2.imread(image_name)
        if input_image is None:
            failed_count += 1
            print(f"Failed to read image: {image_name}")
            continue

        detections = human_detector(input_image)
        annotated, boxes = draw_detections(
            input_image, detections, score_threshold=score_threshold)
        output_name = os.path.join(
            detected_dir if len(boxes) else no_person_dir,
            os.path.basename(image_name),
        )
        if len(boxes):
            if not cv2.imwrite(output_name, annotated):
                raise IOError(f"Failed to write detected image: {output_name}")
            detected_count += 1
        else:
            shutil.copy2(image_name, output_name)
            no_person_count += 1

    print(
        f"Finished: detected={detected_count}, "
        f"no_person={no_person_count}, failed={failed_count}"
    )


def parse_args():
    parser = argparse.ArgumentParser(
        description="Separate COME images with and without detected people.")
    parser.add_argument("--input-dir", default=DEFAULT_INPUT_DIR)
    parser.add_argument("--detected-dir", default=DEFAULT_DETECTED_DIR)
    parser.add_argument("--no-person-dir", default=DEFAULT_NO_PERSON_DIR)
    parser.add_argument("--onnx-model", default=DEFAULT_ONNX_MODEL)
    parser.add_argument("--device", default="cuda", choices=("cuda", "cpu"))
    parser.add_argument("--backend", default="onnxruntime")
    parser.add_argument(
        "--score-threshold", type=float, default=0.7,
        help="Minimum YOLOX score (RTMLib default is 0.7).")
    return parser.parse_args()


def main():
    args = parse_args()
    filter_images(
        input_dir=args.input_dir,
        detected_dir=args.detected_dir,
        no_person_dir=args.no_person_dir,
        onnx_model=args.onnx_model,
        device=args.device,
        backend=args.backend,
        score_threshold=args.score_threshold,
    )


if __name__ == "__main__":
    main()
