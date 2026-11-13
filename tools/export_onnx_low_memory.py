#!/usr/bin/env python3
"""Export FocalNetMatting with weight-preserving, low-memory rewrites.

Example (from the repository root)::

    python tools/export_onnx_low_memory.py \
        --config configs/Matting/focal_matting_3gpu_add_mhp.yaml \
        --weights /path/to/checkpoint.pth --output focal_low_memory.onnx \
        --height 512 --width 512 --image-dir /path/to/images

By default, validates original/re-written PyTorch and original/re-written ONNX
Runtime CPU outputs. Uses strict --atol/--rtol for the rewrite, and separate
--onnx-atol/--onnx-rtol for cross-framework comparisons. Exits nonzero on failure.
Only the optimized ONNX is retained, unless --reference-output is supplied.
This is numerical equivalence testing, not a substitute for a dataset benchmark.
Real images use RGB ImageNet normalization and centered letterbox padding.
The x16/x8 transpose-convolution fusions, x4 fusion and final head are always
rewritten. Add --split-neck to also split the x2 neck's first DW/PW block.
Checkpoint tensors stay unchanged; fused deployment kernels may be smaller.
"""

import argparse
import copy
import json
import logging
from pathlib import Path
import sys
import tempfile
from types import MethodType

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from maggie.network import build_model
from maggie.network.decoder.focal_decoder import DepthwiseSeparableConv, FocalDecoder
from maggie.utils import CONFIG


def _fold_projection(conv, norm):
    """Fold eval BN into an ungrouped pointwise convolution without new Parameters."""
    scale = norm.weight * torch.rsqrt(norm.running_var + norm.eps)
    weight = conv.weight * scale[:, None, None, None]
    bias = norm.bias - norm.running_mean * scale
    if conv.bias is not None:
        bias = bias + conv.bias * scale
    return weight, bias


def _transpose_coefficients(up, fusion):
    """Contract ConvT [Cin,Cmid,Kh,Kw] with pointwise [Cout,Cmid]."""
    weight, bias = _fold_projection(fusion[0], fusion[1])
    split = up.out_channels
    main_weight = weight[:, :split, 0, 0]
    matrix = up.weight.permute(1, 0, 2, 3).reshape(split, -1)
    kernel = torch.mm(main_weight, matrix).reshape(
        fusion[0].out_channels, up.in_channels, *up.kernel_size)
    kernel = kernel.permute(1, 0, 2, 3).contiguous()
    if up.bias is not None:
        bias = bias + F.linear(up.bias, main_weight)
    return kernel, weight[:, split:].contiguous(), bias


def _fused_transpose_branch(feature, skip, up, fusion, coefficients=None):
    """One contracted ConvT plus skip PW; activation stays after the addition.

    Export uses coefficients cached before tracing: older PyTorch versions can
    fail to constant-fold the multi-axis reshape/transpose kernel contraction.
    No kernel construction or weight multiplication runs in the exported graph.
    """
    kernel, skip_weight, bias = (
        _transpose_coefficients(up, fusion) if coefficients is None else coefficients)
    main = F.conv_transpose2d(
        feature, kernel, bias=None, stride=up.stride, padding=up.padding,
        output_padding=up.output_padding, groups=1, dilation=up.dilation)
    if main.shape[-2:] != skip.shape[-2:]:
        main = F.interpolate(main, size=skip.shape[-2:], mode='bilinear', align_corners=False)
    projected_skip = F.conv2d(skip, skip_weight, bias=bias)
    return fusion[2](main + projected_skip)


def _split_neck_block(main, skip, block, split):
    """Split concat -> channelwise DW -> PW/BN; keep DW AFTER upsampling."""
    dw = block.depthwise
    weight, bias = _fold_projection(block.pointwise, block.norm)
    main = F.conv2d(
        main, dw.weight[:split], None if dw.bias is None else dw.bias[:split],
        stride=dw.stride, padding=dw.padding, dilation=dw.dilation, groups=split)
    main = F.conv2d(main, weight[:, :split], bias=None)
    skip = F.conv2d(
        skip, dw.weight[split:], None if dw.bias is None else dw.bias[split:],
        stride=dw.stride, padding=dw.padding, dilation=dw.dilation,
        groups=dw.in_channels - split)
    skip = F.conv2d(skip, weight[:, split:], bias=bias)
    return block.activation(main + skip)


def _low_memory_decode(self, features):
    if len(features) != 6:
        raise ValueError('Expected x1/x2/x3/x4/x5/x6')
    _, x2, x3, x4, x5, x6 = features
    if tuple(x.shape[1] for x in features) != self.input_channels:
        raise ValueError('Encoder channels do not match the decoder')

    p6 = self.refine6(self.top_fuse(torch.cat(
        [self.top_reduce(x6), self.top_context(x6)], dim=1)))
    skip5 = self.skip5(x5, x5.shape[-2:])
    coefficients5 = (self._export_up5_kernel, self._export_up5_skip, self._export_up5_bias)
    p5 = self.refine5(_fused_transpose_branch(
        p6, skip5, self.up6_to5, self.fuse5, coefficients5))
    skip4 = self.skip4(x4, x4.shape[-2:])
    coefficients4 = (self._export_up4_kernel, self._export_up4_skip, self._export_up4_bias)
    p4 = self.refine4(_fused_transpose_branch(
        p5, skip4, self.up5_to4, self.fuse4, coefficients4))

    # W * cat(U(p4), skip) == U(W_main * p4) + W_skip * skip.
    # Keep the original weight tensor; no new parameters and no duplicated bias.
    split = self.skip4.align.out_channels
    # Fold eval BN into the original projection before splitting. This also
    # avoids retaining BN's running-stat constants in the final ONNX graph.
    weight, bias = _fold_projection(self.fuse3[0], self.fuse3[1])
    main = F.conv2d(p4, weight[:, :split], bias=None)
    main = F.interpolate(main, size=x3.shape[-2:], mode='bilinear', align_corners=False)
    skip = self.skip3(x3, x3.shape[-2:])
    skip = F.conv2d(skip, weight[:, split:], bias=bias)
    # Only linear eval BN was folded; GELU must stay AFTER the addition.
    p3 = self.fuse3[2](main + skip)
    p3 = self.refine3(p3)

    neck = F.interpolate(p3, size=x2.shape[-2:], mode='bilinear', align_corners=False)
    stem = self.stem_align(x2, x2.shape[-2:])
    if self._export_split_neck:
        neck = _split_neck_block(neck, stem, self.neck[0], self.skip3.align.out_channels)
        for layer in list(self.neck.children())[1:]:
            neck = layer(neck)
    else:
        neck = self.neck(torch.cat([neck, stem], dim=1))
    return p6, p5, p4, p3, neck


def _low_memory_forward(self, features):
    if self.training:
        raise RuntimeError('Low-memory rewrite is export/evaluation only')
    _, _, _, _, neck = self.decode(features)
    split = self.pred_neck.in_channels
    # Project x2 to ONE logit channel before resizing, never resize 32 channels.
    neck_logits = F.conv2d(neck, self.alpha_head.weight[:, :split], bias=None)
    neck_logits = F.interpolate(
        neck_logits, size=features[0].shape[-2:], mode='bilinear', align_corners=False)
    rgb_logits = F.conv2d(
        self.rgb_stem(features[0]), self.alpha_head.weight[:, split:],
        bias=self.alpha_head.bias)
    # Architecture applies sigmoid only AFTER both branches have been added.
    return [neck_logits + rgb_logits]


def _validate_pointwise(conv):
    if not isinstance(conv, nn.Conv2d) or (
            conv.kernel_size, conv.stride, conv.padding, conv.dilation, conv.groups,
            conv.padding_mode
    ) != ((1, 1), (1, 1), (0, 0), (1, 1), 1, 'zeros'):
        raise ValueError('Rewritten projections must be ordinary ungrouped 1x1 convolutions')


def _validate_norm(norm):
    if not isinstance(norm, nn.BatchNorm2d) or not norm.affine or not norm.track_running_stats:
        raise ValueError('Fusion BN must have affine parameters and eval running statistics')


def rewrite_for_export(model, split_neck=False):
    """Rewrite an already-loaded model; do not load new weights into the copy.

    Derived kernels are non-persistent buffers, not extra learned parameters.
    Re-run this function after loading another checkpoint to refresh them.
    """
    decoder = model.decoder
    # Subclasses (notably the NIG decoder) have a different forward contract.
    if type(decoder) is not FocalDecoder:
        raise TypeError('This exporter supports exactly FocalDecoder, not {}'.format(
            type(decoder).__name__))
    if decoder.eval_output != 'alpha':
        raise ValueError('Set decoder_args.eval_output=alpha to validate the final head')
    for fusion in (decoder.fuse3, decoder.fuse4, decoder.fuse5):
        if len(fusion) != 3 or not isinstance(fusion[2], nn.GELU):
            raise ValueError('Expected fusion = 1x1 Conv + BatchNorm + GELU')
        _validate_pointwise(fusion[0])
        _validate_norm(fusion[1])
    _validate_pointwise(decoder.alpha_head)
    for up, fusion, skip in ((decoder.up6_to5, decoder.fuse5, decoder.skip5),
                             (decoder.up5_to4, decoder.fuse4, decoder.skip4)):
        if not isinstance(up, nn.ConvTranspose2d) or up.groups != 1 or up.padding_mode != 'zeros':
            raise ValueError('Expected an ungrouped, zero-padded transpose convolution')
        if fusion[0].in_channels != up.out_channels + skip.align.out_channels:
            raise ValueError('Unsupported transpose fusion channel layout')
    if decoder.fuse3[0].in_channels != (
            decoder.skip4.align.out_channels + decoder.skip3.align.out_channels):
        raise ValueError('Unsupported fuse3 input channel layout')
    if decoder.alpha_head.in_channels != (
            decoder.pred_neck.in_channels + decoder.rgb_stem[0].out_channels):
        raise ValueError('Unsupported final head channel layout')
    if split_neck:
        if not isinstance(decoder.neck, nn.Sequential) or not len(decoder.neck):
            raise ValueError('Expected a sequential neck')
        block = decoder.neck[0]
        if type(block) is not DepthwiseSeparableConv:
            raise ValueError('Neck splitting requires DepthwiseSeparableConv')
        dw = block.depthwise
        if not isinstance(dw, nn.Conv2d) or not (
                dw.groups == dw.in_channels == dw.out_channels) or dw.padding_mode != 'zeros':
            raise ValueError('Neck splitting requires multiplier-one depthwise convolution')
        _validate_pointwise(block.pointwise)
        _validate_norm(block.norm)
        if dw.in_channels != decoder.skip3.align.out_channels + decoder.stem_align.align.out_channels:
            raise ValueError('Unsupported neck channel layout')
        if block.pointwise.in_channels != dw.out_channels:
            raise ValueError('Unsupported neck pointwise input channels')

    rewritten = copy.deepcopy(model).eval()
    rewritten.decoder._export_split_neck = bool(split_neck)
    with torch.no_grad():
        for name, up, fusion in (
                ('up5', rewritten.decoder.up6_to5, rewritten.decoder.fuse5),
                ('up4', rewritten.decoder.up5_to4, rewritten.decoder.fuse4)):
            for suffix, tensor in zip(('kernel', 'skip', 'bias'), _transpose_coefficients(up, fusion)):
                rewritten.decoder.register_buffer(
                    '_export_{}_{}'.format(name, suffix), tensor, persistent=False)
    rewritten.decoder.decode = MethodType(_low_memory_decode, rewritten.decoder)
    rewritten.decoder.forward = MethodType(_low_memory_forward, rewritten.decoder)
    return rewritten


class AlphaExport(nn.Module):
    def __init__(self, model):
        super().__init__()
        self.model = model

    def forward(self, image):
        return self.model({'image': image}, mem_feat=None)['alpha_pred']


def compare_arrays(reference, candidate, atol, rtol):
    if reference.shape != candidate.shape:
        raise ValueError('Output shapes differ: {} vs {}'.format(reference.shape, candidate.shape))
    delta = np.abs(reference.astype(np.float64) - candidate.astype(np.float64))
    finite = bool(np.isfinite(reference).all() and np.isfinite(candidate).all())
    passed = np.isfinite(delta) & (delta <= atol + rtol * np.abs(reference))
    return {
        'passed': finite and bool(passed.all()),
        'finite': finite,
        'max_abs_error': float(delta.max()),
        'mean_abs_error': float(delta.mean()),
        'rmse': float(np.sqrt(np.mean(delta ** 2))),
        'fraction_outside_tolerance': float(1.0 - passed.mean()),
        'fraction_binary_changed_at_0.5': float(((reference >= 0.5) != (candidate >= 0.5)).mean()),
    }


def validation_inputs(args):
    """Yield normalized NCHW FP32 inputs; real images are not random stand-ins."""
    generator = torch.Generator().manual_seed(args.seed)
    mean = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
    std = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)
    for index in range(args.random_tests):
        rgb = torch.rand(args.batch_size, 3, args.height, args.width, generator=generator)
        yield 'random_{:03d}'.format(index), (rgb - mean) / std
    if args.image_dir is None:
        return
    from PIL import Image
    import cv2

    root = Path(args.image_dir).expanduser().resolve()
    if not root.is_dir():
        raise NotADirectoryError(str(root))
    extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.webp', '.tif', '.tiff'}
    paths = sorted(p for p in root.rglob('*') if p.suffix.lower() in extensions and p.is_file())
    if not paths:
        raise ValueError('No images found in {}'.format(root))
    for path in paths[:args.max_images]:
        with Image.open(path) as image:
            rgb = np.asarray(image.convert('RGB'))
        height, width = rgb.shape[:2]
        scale = min(args.width / width, args.height / height)
        new_w, new_h = max(1, int(width * scale)), max(1, int(height * scale))
        canvas = np.zeros((args.height, args.width, 3), np.uint8)
        top, left = (args.height - new_h) // 2, (args.width - new_w) // 2
        canvas[top:top + new_h, left:left + new_w] = cv2.resize(rgb, (new_w, new_h))
        tensor = torch.from_numpy(canvas.copy()).permute(2, 0, 1).unsqueeze(0).float() / 255.0
        tensor = ((tensor - mean) / std).repeat(args.batch_size, 1, 1, 1)
        yield str(path.relative_to(root)), tensor


def export_graph(model, sample, path, args):
    torch.onnx.export(
        AlphaExport(model).eval(), sample, str(path),
        opset_version=args.opset_version, input_names=['input'],
        output_names=['alpha_pred'], do_constant_folding=True,
    )


def graph_summary(path):
    import onnx
    graph = onnx.load(str(path))
    onnx.checker.check_model(graph)
    counts = {}
    tensor_constant_elements = 0
    for node in graph.graph.node:
        counts[node.op_type] = counts.get(node.op_type, 0) + 1
        if node.op_type == 'Constant':
            tensor_constant_elements += sum(
                int(np.prod(attr.t.dims)) for attr in node.attribute
                if attr.name == 'value' and attr.HasField('t'))
    initializer_elements = sum(int(np.prod(item.dims)) for item in graph.graph.initializer)
    return {
        'path': str(path), 'file_bytes': path.stat().st_size,
        'initializer_elements': initializer_elements,
        'tensor_constant_elements': tensor_constant_elements,
        # Includes helper constants too, not just learned weights. Some PyTorch
        # versions export non-persistent buffers as Constant rather than initializer.
        'stored_tensor_elements': initializer_elements + tensor_constant_elements,
        'op_counts': counts,
    }


def load_models(args):
    cfg = CONFIG.clone()
    cfg.merge_from_file(args.config)
    # Avoid build_model treating a missing local checkpoint as a HuggingFace ID.
    checkpoint_path = Path(args.weights).expanduser().resolve()
    if not checkpoint_path.is_file():
        raise FileNotFoundError(str(checkpoint_path))
    cfg.model.weights = ''
    original, _ = build_model(cfg.model)
    checkpoint = torch.load(str(checkpoint_path), map_location='cpu')
    if isinstance(checkpoint, dict) and 'state_dict' in checkpoint:
        checkpoint = checkpoint['state_dict']
    if all(key.startswith('module.') for key in checkpoint):
        checkpoint = {key[len('module.'):]: value for key, value in checkpoint.items()}
    original.load_state_dict(checkpoint, strict=True)
    original.cpu().eval().requires_grad_(False)
    optimized = rewrite_for_export(original, split_neck=args.split_neck)
    old_state, new_state = original.state_dict(), optimized.state_dict()
    if old_state.keys() != new_state.keys() or any(
            not torch.equal(old_state[key], new_state[key]) for key in old_state):
        raise AssertionError('Rewrite changed checkpoint tensors')
    return original, optimized


def run(args):
    import onnxruntime as ort

    torch.set_num_threads(args.threads)
    output = Path(args.output).expanduser().resolve()
    report_path = Path(args.report).expanduser().resolve() if args.report else output.with_suffix('.validation.json')
    reference_path = Path(args.reference_output).expanduser().resolve() if args.reference_output else None
    if output in {report_path, reference_path} or (reference_path is not None and reference_path == report_path):
        raise ValueError('Output, reference-output and report paths must differ')
    if any(path == Path(args.weights).expanduser().resolve() for path in (output, report_path, reference_path) if path):
        raise ValueError('Do not overwrite the checkpoint')
    original, optimized = load_models(args)
    n_original = sum(p.numel() for p in original.parameters())
    n_optimized = sum(p.numel() for p in optimized.parameters())
    if n_original != n_optimized:
        raise AssertionError('Parameter count changed')
    logging.info('Parameters: original=%d optimized=%d; state_dict tensors unchanged', n_original, n_optimized)
    output.parent.mkdir(parents=True, exist_ok=True)
    report_path.parent.mkdir(parents=True, exist_ok=True)

    with tempfile.TemporaryDirectory(prefix='focal_onnx_validation_') as temporary:
        baseline = reference_path or Path(temporary) / 'original.onnx'
        baseline.parent.mkdir(parents=True, exist_ok=True)
        sample = torch.zeros(args.batch_size, 3, args.height, args.width)
        with torch.no_grad():
            export_graph(original, sample, baseline, args)
            export_graph(optimized, sample, output, args)
        graphs = {'original': graph_summary(baseline), 'optimized': graph_summary(output)}
        options = ort.SessionOptions()
        options.intra_op_num_threads = args.threads
        options.inter_op_num_threads = 1
        original_session = ort.InferenceSession(str(baseline), options, providers=['CPUExecutionProvider'])
        optimized_session = ort.InferenceSession(str(output), options, providers=['CPUExecutionProvider'])
        report = {
            'config': str(Path(args.config).resolve()), 'weights': str(Path(args.weights).resolve()),
            'input_shape': list(sample.shape), 'dtype': 'float32',
            'preprocessing': 'RGB ImageNet normalization; real images use centered letterbox',
            'rewrite_tolerance': {'atol': args.atol, 'rtol': args.rtol},
            'cross_framework_tolerance': {'atol': args.onnx_atol, 'rtol': args.onnx_rtol},
            'parameters_original': n_original, 'parameters_optimized': n_optimized,
            'state_dict_unchanged': True, 'graphs': graphs,
            'rewrites': {'final_head': True, 'x4_fusion': True,
                         'x8_fusion': True, 'x16_fusion': True,
                         'x2_neck_split': args.split_neck},
            'original_onnx_retained': reference_path is not None,
            'scope': 'Numerical output equivalence, not dataset MAD/Grad/IoU evaluation or MNN validation',
            'cases': [],
        }
        for name, tensor in validation_inputs(args):
            with torch.no_grad():
                pt_original = AlphaExport(original)(tensor).cpu().numpy()
                pt_optimized = AlphaExport(optimized)(tensor).cpu().numpy()
            feed = {'input': tensor.numpy()}
            ort_original = original_session.run(['alpha_pred'], feed)[0]
            ort_optimized = optimized_session.run(['alpha_pred'], feed)[0]
            pairs = {
                'pytorch_original_vs_optimized': (pt_original, pt_optimized),
                'pytorch_original_vs_onnx_original': (pt_original, ort_original),
                'pytorch_optimized_vs_onnx_optimized': (pt_optimized, ort_optimized),
                'onnx_original_vs_optimized': (ort_original, ort_optimized),
                'pytorch_original_vs_onnx_optimized': (pt_original, ort_optimized),
            }
            checks = {}
            for key, (a, b) in pairs.items():
                same_framework = key in ('pytorch_original_vs_optimized', 'onnx_original_vs_optimized')
                atol = args.atol if same_framework else args.onnx_atol
                rtol = args.rtol if same_framework else args.onnx_rtol
                checks[key] = dict(compare_arrays(a, b, atol, rtol), atol=atol, rtol=rtol)
            report['cases'].append({'input': name, 'comparisons': checks})
            for key, stats in checks.items():
                logging.info('%s | %s | %s max=%.3g mean=%.3g', name, key,
                             'PASS' if stats['passed'] else 'FAIL', stats['max_abs_error'], stats['mean_abs_error'])
        report['rewrite_passed'] = bool(report['cases']) and all(
            case['comparisons'][key]['passed'] for case in report['cases']
            for key in ('pytorch_original_vs_optimized', 'onnx_original_vs_optimized'))
        report['cross_framework_passed'] = bool(report['cases']) and all(
            stats['passed'] for case in report['cases'] for key, stats in case['comparisons'].items()
            if key not in ('pytorch_original_vs_optimized', 'onnx_original_vs_optimized'))
        report['passed'] = bool(report['cases']) and all(
            stats['passed'] for case in report['cases'] for stats in case['comparisons'].values())
        with report_path.open('w', encoding='utf-8') as stream:
            json.dump(report, stream, indent=2, ensure_ascii=False)
        if not report['passed']:
            raise RuntimeError('Equivalence validation FAILED; exported file is NOT validated. See {}'.format(report_path))
    logging.info('PASS: %d cases; ONNX=%s; report=%s', len(report['cases']), output, report_path)
    return report


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--config', required=True)
    parser.add_argument('--weights', required=True)
    parser.add_argument('--output', default='focal_low_memory.onnx')
    parser.add_argument('--height', type=int, default=512)
    parser.add_argument('--width', type=int, default=512)
    parser.add_argument('--batch-size', '--batch_size', dest='batch_size', type=int, default=1)
    parser.add_argument('--opset-version', '--opset_version', dest='opset_version', type=int, default=13)
    parser.add_argument('--random-tests', type=int, default=3)
    parser.add_argument('--seed', type=int, default=1234)
    parser.add_argument('--image-dir', help='Optional real-image directory for numerical validation')
    parser.add_argument('--max-images', type=int, default=10)
    parser.add_argument('--atol', type=float, default=1e-5)
    parser.add_argument('--rtol', type=float, default=1e-4)
    parser.add_argument('--onnx-atol', type=float, default=5e-4,
                        help='Cross-framework absolute tolerance; does not relax rewrite checks')
    parser.add_argument('--onnx-rtol', type=float, default=1e-4)
    parser.add_argument('--threads', type=int, default=4)
    parser.add_argument('--split-neck', action='store_true',
                        help='Optionally split the x2 neck DW/PW block; default keeps the original neck')
    parser.add_argument('--reference-output', help='Optionally retain the original ONNX for MNN A/B tests')
    parser.add_argument('--report', help='Default: output filename with .validation.json suffix')
    args = parser.parse_args(argv)
    if min(args.height, args.width, args.batch_size, args.threads, args.max_images) <= 0:
        parser.error('Input sizes, batch-size, threads and max-images must be positive')
    if args.random_tests < 0 or (args.random_tests == 0 and not args.image_dir):
        parser.error('Provide at least one random test or an image directory')
    tolerances = [args.atol, args.rtol, args.onnx_atol, args.onnx_rtol]
    if not np.isfinite(tolerances).all() or min(tolerances) < 0:
        parser.error('Tolerances must be finite and non-negative')
    return args


if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)s %(message)s')
    run(parse_args())
