"""Composite model-predicted alpha mattes onto a user-provided background.

The inference path (dataset transform, model prediction, reverse transform and
postprocessing) is shared with ``demo.py``.  For every input image two
composites are written:

* ``composite_raw``     -- ``alpha * I + (1 - alpha) * B`` with the observed
                           image colours acting as the foreground.  Whatever
                           part of the original background survived in the
                           semi-transparent band stays visible as a rim halo.
* ``composite_refined`` -- the same equation after the foreground colour has
                           been re-estimated from ``(I, alpha)`` with a
                           multi-scale weighted-Laplacian solve.  This is the
                           foreground refinement (``refine_foreground``) of
                           "Improving Deep Image Matting via Local Smoothness
                           Assumption", adapted to an RGB-only input.

Example:
    python demo_composite.py \
        --config configs/Matting/focal_matting_3gpu.yaml \
        --input-dir /path/to/images \
        --background /path/to/background.jpg \
        --output-dir output/composite_demo \
        --save-foreground \
        --save-visualization
"""

import argparse
import logging
import sys
import time
from pathlib import Path

import cv2
import numpy as np
from PIL import Image
import torch
from torch.utils.data import DataLoader

sys.path.insert(0, str(Path(__file__).resolve().parent))

from demo import (  # noqa: E402  (imported after sys.path patching)
    IMAGE_EXTENSIONS,
    FixedSizeImageFolderDataset,
    ImageFolderDataset,
    load_model,
    postprocess,
    resolve_device,
    reverse_prediction_transform,
    rgb_guided_filter,
)
from maggie.utils import CONFIG  # noqa: E402


# ---------------------------------------------------------------------------
# Foreground refinement (local smoothness assumption)
# ---------------------------------------------------------------------------


def _fill_reflect(buffer, values):
    """Fill ``buffer`` with the reflect padding of ``values`` (edge not repeated)."""
    buffer[1:-1, 1:-1] = values
    buffer[1:-1, 0] = values[:, 1]
    buffer[1:-1, -1] = values[:, -2]
    buffer[0, 1:-1] = values[1]
    buffer[-1, 1:-1] = values[-2]
    buffer[0, 0] = values[1, 1]
    buffer[0, -1] = values[1, -2]
    buffer[-1, 0] = values[-2, 1]
    buffer[-1, -1] = values[-2, -2]


def _resize_to(array, size):
    """Resize a HxW[...] array to ``(height, width)``."""
    height, width = size
    if array.shape[0] == height and array.shape[1] == width:
        return array
    shrinking = height * width < array.shape[0] * array.shape[1]
    interpolation = cv2.INTER_AREA if shrinking else cv2.INTER_LINEAR
    return cv2.resize(array, (width, height), interpolation=interpolation)


def _jacobi_laplacian_solve(source, trust, edge_a, edge_c, init, kappa,
                            iterations):
    """Solve ``(diag(trust) + kappa * L) x = diag(trust) * source``.

    Args:
        source: HxWxC float array, the value each pixel is pulled towards.
        trust: HxW float array, per-pixel weight of the data term.  Where it is
            zero the pixel is purely interpolated from its neighbours.
        edge_a, edge_c: HxW float arrays defining the smoothness weight
            ``kappa * (a_i * c_j + a_j * c_i)`` of the edge between the
            neighbours ``i`` and ``j``.  The generic form covers both the
            reference ``q_i + q_j`` weighting (``c = 1``) and the object
            anchored one (``a = alpha * (1 - alpha)``, ``c = alpha``).
        init: HxWxC float array used to start the iteration.  The fixed point
            is independent of it; a good guess merely saves iterations.
        kappa: strength of the smoothness term.
        iterations: number of Jacobi sweeps.

    Every update is a convex combination of ``source`` and the four neighbour
    values, so the result stays inside the convex hull of the inputs.

    The weights only depend on ``diffusion``, so they are built once and every
    sweep runs in place on preallocated buffers.  Channels are processed as
    separate 2D planes: NumPy's broadcast path for a ``HxWx1`` operand against a
    ``HxWx3`` operand is more than an order of magnitude slower than a
    contiguous 2D multiply, which otherwise dominates the runtime here.
    """
    source = np.asarray(source, dtype=np.float32)
    if source.ndim == 2:
        source = source[..., None]
    init = np.asarray(init, dtype=np.float32)
    if init.ndim == 2:
        init = init[..., None]
    trust = np.asarray(trust, dtype=np.float32)
    edge_a = np.asarray(edge_a, dtype=np.float32)
    edge_c = np.asarray(edge_c, dtype=np.float32)
    height, width, channels = source.shape

    padded_a = np.pad(edge_a, 1, mode="reflect")
    padded_c = np.pad(edge_c, 1, mode="reflect")
    neighbour_slices = (
        (slice(1, -1), slice(0, -2)),    # left
        (slice(1, -1), slice(2, None)),  # right
        (slice(0, -2), slice(1, -1)),    # up
        (slice(2, None), slice(1, -1)),  # down
    )
    edge_weights = [
        kappa * (edge_a * padded_c[rows, cols] + padded_a[rows, cols] * edge_c)
        for rows, cols in neighbour_slices
    ]
    del padded_a, padded_c
    denominator = trust + sum(edge_weights)
    weights = [np.ascontiguousarray(weight / denominator)
               for weight in edge_weights]

    source_terms = [
        np.ascontiguousarray(trust * source[..., index] / denominator)
        for index in range(channels)
    ]
    results = [
        np.array(init[..., index], dtype=np.float32, copy=True)
        for index in range(channels)
    ]

    padded = np.empty((height + 2, width + 2), dtype=np.float32)
    accumulator = np.empty((height, width), dtype=np.float32)
    scratch = np.empty((height, width), dtype=np.float32)

    for _ in range(max(0, int(iterations))):
        for index in range(channels):
            plane = results[index]
            _fill_reflect(padded, plane)
            np.multiply(weights[0], padded[1:-1, 0:-2], out=accumulator)
            np.multiply(weights[1], padded[1:-1, 2:], out=scratch)
            np.add(accumulator, scratch, out=accumulator)
            np.multiply(weights[2], padded[0:-2, 1:-1], out=scratch)
            np.add(accumulator, scratch, out=accumulator)
            np.multiply(weights[3], padded[2:, 1:-1], out=scratch)
            np.add(accumulator, scratch, out=accumulator)
            np.add(accumulator, source_terms[index], out=plane)

    return np.stack(results, axis=2)


def _multiscale_laplacian_solve(source, trust, edge_a, edge_c, init, kappa,
                                scales, iterations):
    """Coarse-to-fine version of :func:`_jacobi_laplacian_solve`.

    Jacobi sweeps propagate information one pixel at a time, so a full
    resolution solve would need a prohibitive number of iterations.  Following
    the multi-level foreground estimation of the paper, the solve starts at
    ``1 / 2**(scales - 1)`` resolution, which propagates long range cheaply, and
    is then refined at every finer level.  The iteration budget is scaled by
    ``2**level`` so that coarse levels converge while fine levels only correct
    their immediate neighbourhood.
    """
    height, width = source.shape[:2]
    usable = max(1, int(np.floor(np.log2(max(2, min(height, width))))))
    scales = max(1, min(int(scales), usable))

    result = np.array(init, dtype=np.float32, copy=True)
    for level in range(scales)[::-1]:
        if level == 0:
            level_size = (height, width)
        else:
            level_size = (
                max(2, height // 2 ** level),
                max(2, width // 2 ** level),
            )
        level_source = _resize_to(source, level_size)
        level_trust = _resize_to(trust, level_size)
        level_edge_a = _resize_to(edge_a, level_size)
        level_edge_c = _resize_to(edge_c, level_size)
        result = _resize_to(result, level_size)
        result = _jacobi_laplacian_solve(
            level_source,
            level_trust,
            level_edge_a,
            level_edge_c,
            result,
            kappa=kappa,
            iterations=iterations * 2 ** level,
        )
    return result


def refine_foreground(image, alpha, kappa=4.0, alpha_power=2.0, scales=5,
                      iterations=40, weighting="object"):
    """Estimate the uncontaminated foreground colour from ``(image, alpha)``.

    Solves a weighted least squares problem in the shape of the local
    smoothness assumption, minimising

        | 1/2 * sum_i alpha_i**p * ||F_i - I_i||^2
        | + kappa * sum_i sum_{j in d{i}} (a_i * c_j + a_j * c_i) * ||F_i - F_j||^2

    so the semi-transparent band trades the observed colour against its
    neighbours, and sub-pixel mixtures that the original background created are
    replaced by a neighbourhood colour.  That is what removes the rim halo when
    compositing onto a new background.

    Which neighbours may contribute is the one place where this adaptation has
    to differ from the paper.  The paper refines a *known* foreground and uses
    the weights ``(1 - alpha)**2`` because it wants the ``alpha == 0`` part of
    that foreground to become locally constant.  Used to *recover* a foreground
    those weights are actively harmful: at ``alpha = 0.5`` with two opaque and
    two fully transparent neighbours the update returns ``0.19 * F + 0.81 * B``,
    which is further from the truth than the observation it started from,
    because quadratic weights make the transparent neighbours dominate.

    ``weighting="object"`` therefore uses the edge weight
    ``kappa * (a_i * c_j + a_j * c_i)`` with ``a = alpha * (1 - alpha)`` and
    ``c = alpha``: a transition pixel is pulled only by neighbours that are
    closer to opaque, while the object interior and the background stay exact.
    ``weighting="paper"`` keeps ``a = (1 - alpha)**2`` and ``c = 1`` for
    comparison.

    Args:
        image: HxWx3 float array in [0, 1], the observed RGB image.
        alpha: HxW float array in [0, 1].
        kappa: smoothness strength; the main decontamination knob.
        alpha_power: exponent of the data weight.  ``1.0`` trusts the observed
            colour linearly in alpha, ``2.0`` matches the paper's weighting and
            smooths the transition band more aggressively.
        scales: number of pyramid levels.  The defaults (shared with the CLI) are
            ``kappa=4``, ``alpha_power=2``, ``scales=5`` and ``iterations=40``,
            picked on synthetic halos of one to three pixels; the paper uses
            ``kappa=1``, ``alpha_power=2``, ``scales=6`` and ``iterations=20``.
        iterations: Jacobi sweeps at the finest level.  The solve is only
            approximately converged, so more sweeps leave less residual halo.
        weighting: ``"object"`` anchors the band on the opaque neighbourhood,
            ``"paper"`` uses the original ``(1 - alpha)**2`` weights.
    """
    alpha = np.clip(np.asarray(alpha, dtype=np.float32), 0.0, 1.0)
    trust = np.power(alpha, float(alpha_power))
    if weighting == "paper":
        edge_a = (1.0 - alpha) ** 2
        edge_c = np.ones_like(alpha)
    elif weighting == "object":
        edge_a = alpha * (1.0 - alpha)
        edge_c = alpha
        # Keep the fully transparent pixels as an identity pass-through instead
        # of leaving them undefined (their foreground colour is never used).
        trust = np.maximum(trust, 1.0 - (alpha > 0.0))
    else:
        raise ValueError("refine weighting must be 'object' or 'paper'")
    return _multiscale_laplacian_solve(
        image, trust, edge_a, edge_c, image,
        kappa=kappa, scales=scales, iterations=iterations,
    )


def solve_foreground_from_background(image, alpha, scales=5, iterations=40,
                                     min_alpha=1.0 / 255.0,
                                     background_threshold=0.02,
                                     clip_out_of_gamut=False):
    """Unmatte with an estimated original background: ``F = (I - (1-a) B) / a``.

    The original background is only needed inside and around the transition
    band, so it is recovered by a harmonic extension of the observed background
    across the object: pixels with ``alpha <= background_threshold`` keep their
    observed colour and every other pixel is interpolated from them with the
    same multi-scale solver (with a hard mask the smoothness strength cancels,
    so no ``kappa`` is needed).  For a locally constant or smoothly varying
    original background this recovers the true colours of the semi-transparent
    pixels rather than merely plausible ones; where ``alpha`` is tiny the
    division is guarded by ``min_alpha``.

    Returns:
        The foreground in [0, 1] unless ``clip_out_of_gamut`` is False (the
        default).  Out-of-gamut values appear where the predicted alpha
        disagrees with the image, and keeping them makes
        ``alpha * F + (1 - alpha) * B`` reproduce the observation exactly when
        the new background equals the original one.  Clipping here would put
        part of the halo back, so saving code should clip instead.

    The inversion is only trusted where it is physically consistent: if it
    implies a foreground outside [0, 1] the predicted alpha cannot explain the
    observed colour (typically ``alpha`` is too large in a wide soft band), so
    the observation is kept for that pixel and the result degrades to the raw
    composite instead of inventing a brighter, wrong rim.
    """
    alpha = np.clip(np.asarray(alpha, dtype=np.float32), 0.0, 1.0)
    complement = 1.0 - alpha
    unknown = (alpha > background_threshold).astype(np.float32)
    background = _multiscale_laplacian_solve(
        image,
        1.0 - unknown,
        unknown,
        np.ones_like(unknown),
        image,
        kappa=1.0,
        scales=scales,
        iterations=iterations,
    )
    reliable = alpha >= min_alpha
    implied = (
        image - complement[..., None] * background
    ) / np.maximum(alpha, min_alpha)[..., None]
    in_gamut = ((implied >= 0.0) & (implied <= 1.0)).all(axis=2)
    reliable = reliable & in_gamut
    foreground = np.where(reliable[..., None], implied, image)
    if clip_out_of_gamut:
        foreground = np.clip(foreground, 0.0, 1.0)
    return foreground


# ---------------------------------------------------------------------------
# Background handling and compositing
# ---------------------------------------------------------------------------


def build_background_index(background):
    """Resolve ``--background`` into a file or a directory of candidates."""
    path = Path(background).expanduser().resolve()
    if path.is_dir():
        files = sorted(
            candidate for candidate in path.rglob("*")
            if candidate.is_file() and candidate.suffix.lower() in IMAGE_EXTENSIONS
        )
        if not files:
            raise RuntimeError("No supported background images in {}".format(path))
        return {
            "path": path,
            "files": files,
            "by_stem": {candidate.stem: candidate for candidate in files},
        }
    if not path.is_file():
        raise FileNotFoundError("Background not found: {}".format(background))
    return {"path": path, "files": [path], "by_stem": {path.stem: path}}


def select_background(index, relative_path, position):
    """Pick a background: name match first, otherwise cycle through the folder."""
    stem = Path(relative_path).stem
    if stem in index["by_stem"]:
        return index["by_stem"][stem]
    files = index["files"]
    return files[(position - 1) % len(files)]


def fit_background(background, size, fit):
    """Resize a uint8 RGB background to ``(height, width)``."""
    target_h, target_w = size
    source_h, source_w = background.shape[:2]
    shrink = target_h * target_w < source_h * source_w
    interpolation = cv2.INTER_AREA if shrink else cv2.INTER_LINEAR

    if fit == "stretch":
        return cv2.resize(
            background, (target_w, target_h), interpolation=interpolation)

    if fit == "cover":
        scale = max(target_h / float(source_h), target_w / float(source_w))
        resized = cv2.resize(
            background,
            (max(target_w, int(round(source_w * scale))),
             max(target_h, int(round(source_h * scale)))),
            interpolation=interpolation,
        )
        top = (resized.shape[0] - target_h) // 2
        left = (resized.shape[1] - target_w) // 2
        return resized[top:top + target_h, left:left + target_w]

    scale = min(target_h / float(source_h), target_w / float(source_w))
    resized = cv2.resize(
        background,
        (max(1, int(round(source_w * scale))),
         max(1, int(round(source_h * scale)))),
        interpolation=interpolation,
    )
    canvas = np.zeros((target_h, target_w, 3), dtype=np.uint8)
    top = (target_h - resized.shape[0]) // 2
    left = (target_w - resized.shape[1]) // 2
    canvas[top:top + resized.shape[0], left:left + resized.shape[1]] = resized
    return canvas


def load_background(source, size, fit, cache):
    """Load, fit and cache a background as a float array in [0, 1]."""
    key = (str(source), size, fit)
    if key not in cache:
        with Image.open(source) as image:
            background = np.array(image.convert("RGB"), dtype=np.uint8)
        cache[key] = fit_background(background, size, fit).astype(np.float32) / 255.0
    return cache[key]


def composite_over(foreground, alpha, background):
    """``alpha * F + (1 - alpha) * B`` for float arrays in [0, 1]."""
    alpha = alpha[..., None]
    return alpha * foreground + (1.0 - alpha) * background


# ---------------------------------------------------------------------------
# Inference driver
# ---------------------------------------------------------------------------


def _to_uint8(array):
    return np.rint(np.clip(array, 0.0, 1.0) * 255.0).astype(np.uint8)


def _save_image(array, path):
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(array).save(path, format="PNG")


def _output_path(output_dir, subdir, relative_path, instance_index=None):
    relative = Path(relative_path).with_suffix(".png")
    if instance_index is not None:
        relative = relative.with_name(
            "{}_{:02d}.png".format(relative.stem, instance_index))
    return output_dir / subdir / relative


def decontaminate_foreground(image, alpha, args):
    """Return the foreground colours used by the refined composite."""
    if args.decontaminate == "none":
        return image
    if args.decontaminate == "bg-solve":
        return solve_foreground_from_background(
            image, alpha,
            scales=args.refine_scales,
            iterations=args.refine_iters,
        )
    return refine_foreground(
        image, alpha,
        kappa=args.refine_kappa,
        alpha_power=args.refine_alpha_power,
        scales=args.refine_scales,
        iterations=args.refine_iters,
        weighting=args.refine_weighting,
    )


@torch.inference_mode()
def run_composite(model, data_loader, device, output_dir, args):
    total = len(data_loader.dataset)
    index = build_background_index(args.background)
    cache = {}

    for position, sample in enumerate(data_loader, start=1):
        image_path = sample.pop("image_path")[0]
        relative_path = sample.pop("relative_path")[0]
        transform_info = sample.pop("transform_info")
        batch = {
            key: value.to(device, non_blocking=True)
            for key, value in sample.items()
        }

        output = model(batch, mem_feat=None)
        alpha = output.get("refined_masks", output.get("alpha_pred"))
        if alpha is None:
            raise KeyError("Model output contains neither 'refined_masks' nor 'alpha_pred'")

        alpha = reverse_prediction_transform(alpha, transform_info).cpu().numpy()
        alpha[alpha <= 1.0 / 255.0] = 0.0
        alpha[alpha >= 254.0 / 255.0] = 1.0
        if args.postprocessing:
            alpha = postprocess(alpha)

        # Trimap-free image models emit exactly one matte per input image; the
        # reshape keeps support for frame/instance dimensions.
        alpha = alpha[0].reshape(-1, *alpha.shape[-2:])

        with Image.open(image_path) as image:
            image_rgb = np.array(image.convert("RGB"), dtype=np.uint8)
        if image_rgb.shape[:2] != alpha.shape[-2:]:
            raise ValueError(
                "Prediction and input image sizes do not match: "
                "alpha={}, image={} for {}".format(
                    alpha.shape[-2:], image_rgb.shape[:2], image_path))

        if args.rgb_guided_filter:
            alpha = np.stack([
                rgb_guided_filter(
                    image_rgb, instance_alpha,
                    radius=args.guided_radius, eps=args.guided_eps,
                    backend=args.guided_backend)
                for instance_alpha in alpha
            ], axis=0)

        background = load_background(
            select_background(index, relative_path, position),
            image_rgb.shape[:2], args.background_fit, cache)
        image_float = image_rgb.astype(np.float32) / 255.0

        multi_instance = len(alpha) > 1
        for instance_index, instance_alpha in enumerate(alpha):
            suffix = instance_index if multi_instance else None

            raw_composite = composite_over(image_float, instance_alpha, background)

            started = time.perf_counter()
            foreground = decontaminate_foreground(image_float, instance_alpha, args)
            refine_ms = (time.perf_counter() - started) * 1000.0
            refined_composite = composite_over(foreground, instance_alpha, background)

            out_of_gamut = (
                (foreground < 0.0) | (foreground > 1.0)).any(axis=2).mean()
            if out_of_gamut > 0.005:
                logging.warning(
                    "%s%s: %.1f%% of the decontaminated foreground is outside "
                    "[0, 1] even after the consistency gate; pass "
                    "--decontaminate laplacian if the result looks wrong",
                    relative_path, "" if suffix is None else "#{}".format(instance_index),
                    100.0 * out_of_gamut)

            _save_image(
                _to_uint8(instance_alpha),
                _output_path(output_dir, "alpha", relative_path, suffix))
            _save_image(
                _to_uint8(raw_composite),
                _output_path(output_dir, "composite_raw", relative_path, suffix))
            _save_image(
                _to_uint8(refined_composite),
                _output_path(output_dir, "composite_refined", relative_path, suffix))

            if args.save_foreground:
                _save_image(
                    _to_uint8(foreground),
                    _output_path(output_dir, "foreground", relative_path, suffix))

            if args.save_rgba:
                _save_image(
                    np.dstack((_to_uint8(foreground), _to_uint8(instance_alpha))),
                    _output_path(output_dir, "rgba", relative_path, suffix))

            if args.save_visualization:
                alpha_panel = np.repeat(
                    _to_uint8(instance_alpha)[..., None], 3, axis=-1)
                panel = np.concatenate([
                    image_rgb,
                    alpha_panel,
                    _to_uint8(raw_composite),
                    _to_uint8(refined_composite),
                ], axis=1)
                _save_image(
                    panel,
                    _output_path(output_dir, "visualization", relative_path, suffix))

            logging.info(
                "[%d/%d] %s%s  refine %.0f ms",
                position, total, relative_path,
                "" if suffix is None else "#{}".format(instance_index), refine_ms)


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Composite model-predicted alpha onto a background image and save "
            "the raw and the foreground-refined result"
        )
    )
    parser.add_argument("--config", required=True, help="Experiment YAML config")
    parser.add_argument("--input-dir", required=True,
                        help="Folder containing the foreground images")
    parser.add_argument(
        "--background", required=True,
        help=(
            "Background image, or a folder of images.  In folder mode a file "
            "sharing the input stem is preferred, otherwise candidates are cycled"
        ),
    )
    parser.add_argument("--output-dir", required=True,
                        help="Folder used to save composites")
    parser.add_argument(
        "--background-fit", choices=("stretch", "cover", "contain"),
        default="stretch",
        help=(
            "How to fit the background to the image size (default: stretch, "
            "which distorts nothing but crops nothing either)"
        ),
    )
    parser.add_argument("--weights", default=None,
                        help="Override model.weights from the config")
    parser.add_argument("--short-size", type=int, default=None,
                        help="Override dataset.test.short_size from the config")
    parser.add_argument(
        "--resize-mode", choices=("short-side", "fixed"), default="short-side",
        help="Inference preprocessing, identical to demo.py",
    )
    parser.add_argument("--device", default="auto",
                        help="Inference device, e.g. cuda:0 or cpu")
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--non-recursive", action="store_true",
                        help="Only read images immediately inside input-dir")

    postprocess_group = parser.add_mutually_exclusive_group()
    postprocess_group.add_argument("--postprocessing", action="store_true")
    postprocess_group.add_argument("--no-postprocessing", action="store_true")

    parser.add_argument(
        "--rgb-guided-filter", action="store_true",
        help="Refine alpha with the RGB guided filter before compositing",
    )
    parser.add_argument("--guided-radius", type=int, default=4)
    parser.add_argument("--guided-eps", type=float, default=1e-4)
    parser.add_argument(
        "--guided-backend", choices=("numpy", "opencv", "auto"),
        default="numpy",
        help="Guided-filter implementation used by --rgb-guided-filter",
    )

    parser.add_argument(
        "--decontaminate", choices=("laplacian", "bg-solve", "none"),
        default="laplacian",
        help=(
            "Foreground used by composite_refined: 'laplacian' (default) anchors "
            "the transition band on the nearby opaque colours and was the more "
            "reliable option on real data; 'bg-solve' inverts the compositing "
            "equation with an estimated original background, which removes more "
            "halo on clean mattes but amplifies an inaccurate alpha; 'none' "
            "reuses the observed colours"
        ),
    )
    parser.add_argument(
        "--refine-kappa", type=float, default=4.0,
        help=(
            "Smoothness strength of the 'laplacian' mode (default: 4, chosen on "
            "synthetic halos of one to three pixels; raise it to pull the "
            "transition band harder towards the opaque neighbourhood)"
        ),
    )
    parser.add_argument(
        "--refine-alpha-power", type=float, default=2.0,
        help=(
            "Exponent of the refinement data weight (default: 2.0, which trusts "
            "the observed colour quadratically in alpha and decontaminates the "
            "transition band harder; 1.0 keeps more of the observed colour)"
        ),
    )
    parser.add_argument(
        "--refine-scales", type=int, default=5,
        help=(
            "Pyramid levels used by the refinement (default: 5); extra levels "
            "beyond five only add cost on typical image sizes"
        ),
    )
    parser.add_argument(
        "--refine-weighting", choices=("object", "paper"), default="object",
        help=(
            "Neighbour weighting of the 'laplacian' mode: 'object' (default) "
            "pulls the transition band towards the opaque neighbourhood, "
            "'paper' keeps the (1 - alpha)^2 weights of the original method"
        ),
    )
    parser.add_argument(
        "--refine-iters", type=int, default=40,
        help=(
            "Jacobi sweeps at the finest pyramid level (default: 40).  More "
            "sweeps leave less residual halo (the solver is only approximately "
            "converged) and widen the reach of 'laplacian' by roughly one pixel "
            "per sweep"
        ),
    )

    parser.add_argument("--save-foreground", action="store_true",
                        help="Also save the refined foreground colours")
    parser.add_argument("--save-rgba", action="store_true",
                        help="Also save a straight RGBA cutout of the refined foreground")
    parser.add_argument(
        "--save-visualization", action="store_true",
        help="Also save [input | alpha | raw composite | refined composite] panels",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")

    if args.refine_kappa <= 0:
        raise ValueError("refine-kappa must be > 0")
    if args.refine_alpha_power <= 0:
        raise ValueError("refine-alpha-power must be > 0")
    if args.refine_scales < 1:
        raise ValueError("refine-scales must be >= 1")
    if args.refine_iters < 1:
        raise ValueError("refine-iters must be >= 1")
    if args.guided_radius < 1:
        raise ValueError("guided-radius must be >= 1")
    if args.guided_eps <= 0:
        raise ValueError("guided-eps must be > 0")

    # Fail on a bad background before building the model.
    background_index = build_background_index(args.background)

    cfg = CONFIG.clone()
    cfg.merge_from_file(args.config)
    if args.weights is not None:
        cfg.model.weights = args.weights

    short_size = (
        args.short_size if args.short_size is not None
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
    args.postprocessing = do_postprocessing

    device = resolve_device(args.device)
    output_dir = Path(args.output_dir).expanduser().resolve()
    dataset_class = (
        FixedSizeImageFolderDataset if args.resize_mode == "fixed"
        else ImageFolderDataset
    )
    dataset = dataset_class(
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

    logging.info(
        "Found %d images; %s preprocessing on %s",
        len(dataset), args.resize_mode, device)
    if background_index["path"].is_dir():
        logging.info(
            "Background folder %s (%d candidates)",
            background_index["path"], len(background_index["files"]))
    else:
        logging.info("Background image %s", background_index["path"])
    logging.info("Background fit: %s", args.background_fit)
    logging.info(
        "Postprocessing: %s; RGB guided filter: %s",
        do_postprocessing, args.rgb_guided_filter)
    if args.decontaminate == "none":
        logging.warning(
            "Decontamination is disabled: composite_refined equals composite_raw")
    elif args.decontaminate == "bg-solve":
        logging.info(
            "Decontamination 'bg-solve': scales=%d, iters=%d; inverting the "
            "compositing equation only helps where alpha is accurate and the "
            "original background is locally smooth, and is gated to in-gamut "
            "solutions otherwise",
            args.refine_scales, args.refine_iters)
    else:
        logging.info(
            "Decontamination '%s': kappa=%g, alpha_power=%g, weighting=%s, "
            "scales=%d, iters=%d",
            args.decontaminate, args.refine_kappa, args.refine_alpha_power,
            args.refine_weighting, args.refine_scales, args.refine_iters)

    model = load_model(cfg, device)
    run_composite(model, data_loader, device, output_dir, args)

    logging.info("Raw composites saved to %s", output_dir / "composite_raw")
    logging.info("Refined composites saved to %s", output_dir / "composite_refined")
    logging.info("Alpha mattes saved to %s", output_dir / "alpha")
    if args.save_foreground:
        logging.info("Refined foregrounds saved to %s", output_dir / "foreground")
    if args.save_rgba:
        logging.info("RGBA cutouts saved to %s", output_dir / "rgba")
    if args.save_visualization:
        logging.info("Visualizations saved to %s", output_dir / "visualization")


if __name__ == "__main__":
    main()
