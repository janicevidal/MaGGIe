"""Run image matting inference on a folder without ground-truth alpha mattes.

Example:
    python demo.py \
        --config configs/biref_matting_mod_3gpu_channel_pro_test.yaml \
        --input-dir /path/to/images \
        --output-dir output/demo \
        --save-composite
"""

import argparse
import logging
import os
from pathlib import Path

import cv2
import numpy as np
from PIL import Image
import torch
from torch.utils.data import DataLoader, Dataset

from maggie.network import build_model
from maggie.utils import CONFIG
from maggie.utils.postprocessing import postprocess, reverse_transform_tensor


IMAGE_EXTENSIONS = {
    ".bmp", ".jpeg", ".jpg", ".png", ".tif", ".tiff", ".webp"
}
IMAGENET_MEAN = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
IMAGENET_STD = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)


class ImageFolderDataset(Dataset):
    """Load and preprocess images without requiring alpha or mask files."""

    def __init__(self, input_dir, short_size, recursive=True, exclude_dir=None):
        self.input_dir = Path(input_dir).expanduser().resolve()
        self.short_size = short_size

        if not self.input_dir.is_dir():
            raise NotADirectoryError("Input directory does not exist: {}".format(input_dir))

        iterator = self.input_dir.rglob("*") if recursive else self.input_dir.glob("*")
        exclude_dir = Path(exclude_dir).expanduser().resolve() if exclude_dir else None
        excluded_dirs = []
        if exclude_dir == self.input_dir:
            excluded_dirs = [exclude_dir / "alpha", exclude_dir / "composite"]
        elif exclude_dir is not None and self.input_dir in exclude_dir.parents:
            excluded_dirs = [exclude_dir]
        self.image_paths = []
        for path in iterator:
            if not path.is_file() or path.suffix.lower() not in IMAGE_EXTENSIONS:
                continue
            resolved_path = path.resolve()
            if any(
                    directory == resolved_path or directory in resolved_path.parents
                    for directory in excluded_dirs):
                continue
            self.image_paths.append(resolved_path)
        self.image_paths.sort(key=lambda path: str(path.relative_to(self.input_dir)))

        if not self.image_paths:
            raise RuntimeError("No supported images found in {}".format(self.input_dir))

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, index):
        image_path = self.image_paths[index]
        with Image.open(image_path) as image:
            image = np.array(image.convert("RGB"))

        ori_h, ori_w = image.shape[:2]
        ratio = self.short_size / float(min(ori_h, ori_w))
        resized_h, resized_w = int(ori_h * ratio), int(ori_w * ratio)
        if ratio != 1:
            image = cv2.resize(
                image, (resized_w, resized_h), interpolation=cv2.INTER_LINEAR
            )

        pad_h = (32 - resized_h % 32) % 32
        pad_w = (32 - resized_w % 32) % 32
        image = cv2.copyMakeBorder(
            image, 0, pad_h, 0, pad_w, cv2.BORDER_CONSTANT, value=0
        )

        image = torch.from_numpy(np.ascontiguousarray(image))
        image = image.permute(2, 0, 1).contiguous().float() / 255.0
        image = (image - IMAGENET_MEAN) / IMAGENET_STD

        transform_info = [
            {"name": "resize", "ori_size": (ori_h, ori_w), "ratio": ratio},
            {"name": "padding", "pad_size": (pad_h, pad_w)},
        ]
        return {
            "image": image,
            "image_path": str(image_path),
            "relative_path": str(image_path.relative_to(self.input_dir)),
            "transform_info": transform_info,
        }


def resolve_device(device_name):
    if device_name == "auto":
        return torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    device = torch.device(device_name)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested, but no CUDA device is available")
    return device


def load_model(cfg, device):
    logging.info("Building %s", cfg.model.arch)
    model, is_from_hf = build_model(cfg.model)
    model = model.to(device)

    if not is_from_hf:
        if not cfg.model.weights or not os.path.isfile(cfg.model.weights):
            raise FileNotFoundError(
                "Cannot find model weights: {}".format(cfg.model.weights or "<empty>")
            )
        logging.info("Loading weights from %s", cfg.model.weights)
        state_dict = torch.load(cfg.model.weights, map_location=device)
        missing_keys, unexpected_keys = model.load_state_dict(state_dict, strict=False)
        if missing_keys:
            logging.warning("Missing keys: %s", missing_keys)
        if unexpected_keys:
            logging.warning("Unexpected keys: %s", unexpected_keys)

    model.eval()
    return model


def output_paths(output_dir, relative_path, instance_index=None):
    relative_path = Path(relative_path).with_suffix(".png")
    if instance_index is not None:
        relative_path = relative_path.with_name(
            "{}_{:02d}.png".format(relative_path.stem, instance_index)
        )
    alpha_path = output_dir / "alpha" / relative_path
    composite_path = output_dir / "composite" / relative_path
    return alpha_path, composite_path


def save_prediction(alpha, image_path, alpha_path, composite_path=None):
    alpha = np.clip(alpha, 0.0, 1.0)
    alpha_image = np.rint(alpha * 255.0).astype(np.uint8)
    alpha_path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(alpha_image, mode="L").save(alpha_path)

    if composite_path is not None:
        with Image.open(image_path) as image:
            image = np.array(image.convert("RGB"), dtype=np.float32)
        composite = image * alpha[..., None] + 255.0 * (1.0 - alpha[..., None])
        composite = np.rint(composite).clip(0, 255).astype(np.uint8)
        composite_path.parent.mkdir(parents=True, exist_ok=True)
        Image.fromarray(composite, mode="RGB").save(composite_path)


@torch.inference_mode()
def run_inference(model, data_loader, device, output_dir, do_postprocessing,
                  save_composite):
    total = len(data_loader.dataset)
    for index, sample in enumerate(data_loader, start=1):
        image_path = sample.pop("image_path")[0]
        relative_path = sample.pop("relative_path")[0]
        transform_info = sample.pop("transform_info")
        batch = {key: value.to(device, non_blocking=True) for key, value in sample.items()}

        output = model(batch, mem_feat=None)
        alpha = output.get("refined_masks", output.get("alpha_pred"))
        if alpha is None:
            raise KeyError("Model output contains neither 'refined_masks' nor 'alpha_pred'")

        alpha = reverse_transform_tensor(alpha, transform_info).cpu().numpy()
        alpha[alpha <= 1.0 / 255.0] = 0.0
        alpha[alpha >= 254.0 / 255.0] = 1.0
        if do_postprocessing:
            alpha = postprocess(alpha)

        # Flatten frame/instance dimensions. Trimap-free image models normally
        # produce exactly one alpha matte per input image.
        alpha = alpha[0].reshape(-1, *alpha.shape[-2:])
        for instance_index, instance_alpha in enumerate(alpha):
            suffix_index = instance_index if len(alpha) > 1 else None
            alpha_path, composite_path = output_paths(
                output_dir, relative_path, suffix_index
            )
            save_prediction(
                instance_alpha,
                image_path,
                alpha_path,
                composite_path if save_composite else None,
            )

        logging.info("[%d/%d] %s", index, total, relative_path)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Run matting inference on a folder of images without GT"
    )
    parser.add_argument("--config", required=True, help="Experiment YAML config")
    parser.add_argument("--input-dir", required=True, help="Folder containing input images")
    parser.add_argument("--output-dir", required=True, help="Folder used to save results")
    parser.add_argument(
        "--weights", default=None, help="Override model.weights from the config"
    )
    parser.add_argument(
        "--short-size", type=int, default=None,
        help="Override dataset.test.short_size from the config",
    )
    parser.add_argument(
        "--device", default="auto", help="Inference device, e.g. cuda:0 or cpu"
    )
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument(
        "--non-recursive", action="store_true",
        help="Only read images immediately inside input-dir",
    )
    parser.add_argument(
        "--save-composite", action="store_true",
        help="Also composite each prediction over a white background",
    )
    postprocess_group = parser.add_mutually_exclusive_group()
    postprocess_group.add_argument("--postprocessing", action="store_true")
    postprocess_group.add_argument("--no-postprocessing", action="store_true")
    return parser.parse_args()


def main():
    args = parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")

    cfg = CONFIG.clone()
    cfg.merge_from_file(args.config)
    if args.weights is not None:
        cfg.model.weights = args.weights

    short_size = (
        args.short_size
        if args.short_size is not None
        else cfg.dataset.test.short_size
    )
    if short_size <= 0:
        raise ValueError("short-size must be positive")

    if args.postprocessing:
        do_postprocessing = True
    elif args.no_postprocessing:
        do_postprocessing = False
    else:
        do_postprocessing = cfg.test.postprocessing

    device = resolve_device(args.device)
    output_dir = Path(args.output_dir).expanduser().resolve()
    dataset = ImageFolderDataset(
        args.input_dir,
        short_size=short_size,
        recursive=not args.non_recursive,
        exclude_dir=output_dir,
    )
    data_loader = DataLoader(
        dataset,
        batch_size=1,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=device.type == "cuda",
    )

    logging.info("Found %d images; using %s", len(dataset), device)
    model = load_model(cfg, device)
    run_inference(
        model,
        data_loader,
        device,
        output_dir,
        do_postprocessing,
        args.save_composite,
    )
    logging.info("Done. Alpha mattes saved to %s", output_dir / "alpha")


if __name__ == "__main__":
    main()
