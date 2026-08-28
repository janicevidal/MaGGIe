"""Filter extracted S3OD samples with a fully local instruction model."""

import argparse
import hashlib
import json
import logging
import math
import os
import shutil
import threading
import time
from pathlib import Path

from tqdm import tqdm


LOGGER = logging.getLogger(__name__)
SYSTEM_PROMPT = """You are a precise dataset caption classifier.
Decide whether each caption describes at least one real person or real animal
present in the depicted scene.

Person includes explicit people and strongly implied human actors, such as a
chef, rider, crowd, child, hand holding an object, or someone wearing clothes.
Animal includes mammals, birds, fish, reptiles, amphibians, and insects.

Do not count statues, mannequins, dolls, toys, logos, printed pictures,
paintings, animal-shaped objects, or prepared food when no living person or
animal is described. Do not infer an entity from an object alone.

Classify one caption and return exactly one digit:
0 = neither person nor animal
1 = person only
2 = animal only
3 = both person and animal
Return only the digit, without explanations or punctuation."""
PROMPT_SHA256 = hashlib.sha256(SYSTEM_PROMPT.encode('utf-8')).hexdigest()
IMAGE_EXTENSIONS = ('.jpg', '.jpeg', '.png', '.webp', '.bmp', '.tif', '.tiff')
MASK_EXTENSIONS = ('.png', '.jpg', '.jpeg', '.webp', '.bmp', '.tif', '.tiff')
LABELS_BY_CODE = {
    '0': frozenset(),
    '1': frozenset({'person'}),
    '2': frozenset({'animal'}),
    '3': frozenset({'person', 'animal'}),
}


def format_bytes(num_bytes):
    value = float(num_bytes)
    for unit in ('B', 'KiB', 'MiB', 'GiB', 'TiB'):
        if value < 1024 or unit == 'TiB':
            return f'{value:.2f} {unit}'
        value /= 1024


def device_map_summary(device_map):
    summary = {}
    for device in device_map.values():
        device = str(device)
        summary[device] = summary.get(device, 0) + 1
    return ', '.join(
        f'{device}={num_modules} modules'
        for device, num_modules in sorted(summary.items()))


def caption_digest(caption):
    return hashlib.sha256(caption.encode('utf-8')).hexdigest()


class LocalTransformersClassifier:
    """Run a local causal language model without any network access."""

    def __init__(self, model_path, device='auto', dtype='auto',
                 max_new_tokens=2, heartbeat_seconds=30):
        model_path = Path(model_path).expanduser().resolve()
        if not model_path.is_dir():
            raise FileNotFoundError(
                f'Local model directory does not exist: {model_path}')
        if max_new_tokens < 2:
            raise ValueError('max_new_tokens must be at least 2')
        if heartbeat_seconds < 0:
            raise ValueError('heartbeat_seconds cannot be negative')

        # Enforce offline behavior even if the host has network access.
        os.environ['HF_HUB_OFFLINE'] = '1'
        os.environ['TRANSFORMERS_OFFLINE'] = '1'
        try:
            import torch
            from transformers import AutoModelForCausalLM, AutoTokenizer
        except ImportError as error:
            raise RuntimeError(
                'Local inference requires torch and transformers. Install '
                'them in the Python environment running this script.') from error

        LOGGER.info(
            'Runtime: torch=%s, CUDA available=%s, CUDA devices=%d, '
            'requested device=%s, requested dtype=%s',
            torch.__version__, torch.cuda.is_available(),
            torch.cuda.device_count(), device, dtype)
        if torch.cuda.is_available():
            for device_index in range(torch.cuda.device_count()):
                properties = torch.cuda.get_device_properties(device_index)
                LOGGER.info(
                    'CUDA %d: %s, total memory=%s, capability=%d.%d',
                    device_index, properties.name,
                    format_bytes(properties.total_memory),
                    properties.major, properties.minor)
        elif device != 'cpu':
            LOGGER.warning(
                'CUDA is unavailable. device=%s may place the 7B model on '
                'CPU and generation can be extremely slow.', device)

        if dtype == 'auto':
            torch_dtype = 'auto'
        else:
            torch_dtype = getattr(torch, dtype)
        load_args = {
            'local_files_only': True,
            'torch_dtype': torch_dtype,
            'low_cpu_mem_usage': True,
        }
        if device == 'auto':
            load_args['device_map'] = 'auto'

        LOGGER.info('Loading tokenizer from %s', model_path)
        load_start = time.perf_counter()
        self.tokenizer = AutoTokenizer.from_pretrained(
            model_path, local_files_only=True)
        self.tokenizer.padding_side = 'left'
        if self.tokenizer.pad_token_id is None:
            self.tokenizer.pad_token_id = self.tokenizer.eos_token_id
        LOGGER.info(
            'Tokenizer loaded in %.2fs; loading model weights now',
            time.perf_counter() - load_start)
        model_load_start = time.perf_counter()
        self.model = AutoModelForCausalLM.from_pretrained(
            model_path, **load_args)
        LOGGER.info(
            'Model weights loaded in %.2fs',
            time.perf_counter() - model_load_start)
        if device != 'auto':
            LOGGER.info('Moving model to %s', device)
            move_start = time.perf_counter()
            self.model = self.model.to(device)
            LOGGER.info(
                'Model moved to %s in %.2fs',
                device, time.perf_counter() - move_start)
        self.model.eval()
        # Qwen's saved generation config may contain sampling parameters even
        # when greedy decoding is requested. Clear them to avoid misleading
        # warnings and keep classification deterministic.
        self.model.generation_config.do_sample = False
        self.model.generation_config.temperature = None
        self.model.generation_config.top_p = None
        self.model.generation_config.top_k = None
        self.torch = torch
        self.input_device = next(self.model.parameters()).device
        # One constrained class-code token followed by EOS. Retain the CLI
        # argument for compatibility, but never permit long free-form output.
        self.max_new_tokens = 2
        if max_new_tokens != self.max_new_tokens:
            LOGGER.info(
                'Overriding max_new_tokens=%d with %d for constrained '
                'single-code classification',
                max_new_tokens, self.max_new_tokens)
        self.heartbeat_seconds = heartbeat_seconds

        self.code_id_to_labels = {}
        self.allowed_code_ids = []
        for code, labels in LABELS_BY_CODE.items():
            token_ids = self.tokenizer.encode(
                code, add_special_tokens=False)
            if len(token_ids) != 1:
                raise ValueError(
                    f'Classification code {code!r} is not one token: '
                    f'{token_ids}')
            token_id = token_ids[0]
            self.allowed_code_ids.append(token_id)
            self.code_id_to_labels[token_id] = set(labels)
        if len(self.code_id_to_labels) != len(LABELS_BY_CODE):
            raise ValueError('Classification codes do not have unique token IDs')

        model_dtype = next(self.model.parameters()).dtype
        footprint = self.model.get_memory_footprint()
        LOGGER.info(
            'Model ready: input_device=%s, dtype=%s, memory footprint=%s, '
            'max_new_tokens=%d',
            self.input_device, model_dtype, format_bytes(footprint),
            self.max_new_tokens)
        device_map = getattr(self.model, 'hf_device_map', None)
        if device_map:
            summary = device_map_summary(device_map)
            LOGGER.info('Model device map: %s', summary)
            mapped_devices = {str(value) for value in device_map.values()}
            if 'cpu' in mapped_devices or 'disk' in mapped_devices:
                LOGGER.warning(
                    'Model contains CPU/disk-offloaded modules (%s). '
                    'Autoregressive generation may be very slow.', summary)
        self._log_cuda_memory('after model loading')

    def _log_cuda_memory(self, stage):
        if not self.torch.cuda.is_available():
            return
        for device_index in range(self.torch.cuda.device_count()):
            LOGGER.info(
                'CUDA %d memory %s: allocated=%s, reserved=%s, free=%s',
                device_index, stage,
                format_bytes(self.torch.cuda.memory_allocated(device_index)),
                format_bytes(self.torch.cuda.memory_reserved(device_index)),
                format_bytes(self.torch.cuda.mem_get_info(device_index)[0]))

    def _generation_heartbeat(self, stop_event, start_time, input_tokens):
        while not stop_event.wait(self.heartbeat_seconds):
            LOGGER.info(
                'Generation still running: elapsed=%.1fs, input_tokens=%d, '
                'max_new_tokens=%d',
                time.perf_counter() - start_time, input_tokens,
                self.max_new_tokens)

    def classify(self, captions):
        LOGGER.info(
            'Preparing model input: captions=%d, total_characters=%d',
            len(captions), sum(len(caption) for caption in captions))
        prepare_start = time.perf_counter()
        prompts = []
        for caption in captions:
            messages = [
                {'role': 'system', 'content': SYSTEM_PROMPT},
                {
                    'role': 'user',
                    'content': f'Caption: {caption}',
                },
            ]
            prompts.append(self.tokenizer.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True))
        inputs = self.tokenizer(
            prompts, padding=True, return_tensors='pt')
        inputs = {
            name: tensor.to(self.input_device)
            for name, tensor in inputs.items()
        }
        padded_input_length = inputs['input_ids'].shape[1]
        token_counts = inputs['attention_mask'].sum(dim=1)
        total_input_tokens = int(token_counts.sum().item())
        LOGGER.info(
            'Input ready in %.2fs: sequences=%d, total_tokens=%d, '
            'tokens_per_sequence=%d..%d, padded_length=%d, device=%s; '
            'starting constrained generation',
            time.perf_counter() - prepare_start, len(captions),
            total_input_tokens, int(token_counts.min().item()),
            int(token_counts.max().item()), padded_input_length,
            self.input_device)
        self._log_cuda_memory('before generation')

        def allowed_tokens(batch_id, input_ids):
            generated_length = input_ids.shape[-1] - padded_input_length
            if generated_length == 0:
                return self.allowed_code_ids
            return [self.tokenizer.eos_token_id]

        generation_start = time.perf_counter()
        stop_event = threading.Event()
        heartbeat = None
        if self.heartbeat_seconds > 0:
            heartbeat = threading.Thread(
                target=self._generation_heartbeat,
                args=(stop_event, generation_start, total_input_tokens),
                daemon=True)
            heartbeat.start()
        try:
            with self.torch.inference_mode():
                output_ids = self.model.generate(
                    **inputs,
                    max_new_tokens=self.max_new_tokens,
                    do_sample=False,
                    pad_token_id=self.tokenizer.pad_token_id,
                    eos_token_id=self.tokenizer.eos_token_id,
                    prefix_allowed_tokens_fn=allowed_tokens)
        finally:
            stop_event.set()
            if heartbeat is not None:
                heartbeat.join()

        generation_time = time.perf_counter() - generation_start
        output_length = output_ids.shape[1] - padded_input_length
        total_output_tokens = output_length * len(captions)
        LOGGER.info(
            'Generation completed in %.2fs: sequences=%d, '
            'output_tokens=%d, throughput=%.2f tokens/s',
            generation_time, len(captions), total_output_tokens,
            total_output_tokens / max(generation_time, 1e-8))
        code_ids = output_ids[:, padded_input_length].tolist()
        results = []
        for code_id in code_ids:
            if code_id not in self.code_id_to_labels:
                raise RuntimeError(
                    f'Constrained generation returned invalid token ID '
                    f'{code_id}')
            results.append(set(self.code_id_to_labels[code_id]))
        LOGGER.info(
            'Classification codes parsed: matched=%d/%d, persons=%d, '
            'animals=%d, codes=%s',
            sum(bool(labels) for labels in results), len(results),
            sum('person' in labels for labels in results),
            sum('animal' in labels for labels in results),
            self.tokenizer.convert_ids_to_tokens(code_ids))
        return results


def load_cached_results(cache_path):
    results = {}
    if not cache_path.exists():
        return results
    lines = cache_path.read_text(encoding='utf-8').splitlines()
    for line_number, line in enumerate(lines, 1):
        if not line.strip():
            continue
        try:
            result = json.loads(line)
            results[result['image_id']] = result
        except (KeyError, json.JSONDecodeError) as error:
            if line_number == len(lines):
                LOGGER.warning(
                    'Ignoring an incomplete final cache line in %s', cache_path)
                break
            raise ValueError(
                f'Invalid cache line {line_number} in {cache_path}: '
                f'{error}') from error
    return results


def find_sample_file(directory, image_id, extensions=IMAGE_EXTENSIONS,
                     required=True):
    # Avoid Path.glob here: repeatedly scanning a directory containing tens of
    # thousands of samples makes cached resume effectively O(N^2).
    for extension in extensions:
        for candidate_extension in (extension, extension.upper()):
            path = directory / f'{image_id}{candidate_extension}'
            if path.is_file():
                return path
    if required:
        raise FileNotFoundError(f'No file for {image_id} under {directory}')
    return None


def copy_file(source, destination_dir, overwrite=False):
    destination = destination_dir / source.name
    if destination.exists() and not overwrite:
        return False
    temporary = destination.with_suffix(destination.suffix + '.tmp')
    shutil.copy2(source, temporary)
    temporary.replace(destination)
    return True


def copy_matching_sample(source_root, output_root, image_id,
                         captions_only=False, overwrite=False):
    source_caption = source_root / 'captions' / f'{image_id}.txt'
    caption_output_dir = output_root / 'captions'
    caption_output_dir.mkdir(parents=True, exist_ok=True)
    if not source_caption.is_file():
        raise FileNotFoundError(source_caption)
    copy_file(source_caption, caption_output_dir, overwrite=overwrite)

    if captions_only:
        return
    for directory_name, extensions in (
            ('images', IMAGE_EXTENSIONS), ('masks', MASK_EXTENSIONS)):
        destination_dir = output_root / directory_name
        destination_dir.mkdir(parents=True, exist_ok=True)
        if (not overwrite and find_sample_file(
                destination_dir, image_id, extensions, required=False)):
            continue
        source = find_sample_file(
            source_root / directory_name, image_id, extensions)
        copy_file(source, destination_dir, overwrite=overwrite)


def build_result(image_id, caption, labels, model, caption_path=None,
                 caption_stat=None):
    labels = sorted(labels)
    result = {
        'image_id': image_id,
        'caption_sha256': caption_digest(caption),
        'prompt_sha256': PROMPT_SHA256,
        'match': bool(labels),
        'contains_person': 'person' in labels,
        'contains_animal': 'animal' in labels,
        'labels': labels,
        'model': model,
    }
    if caption_path is not None:
        stat = caption_stat if caption_stat is not None else caption_path.stat()
        result.update({
            'caption_size': stat.st_size,
            'caption_mtime_ns': stat.st_mtime_ns,
        })
    return result


def cached_result_is_current(result, caption_path, model):
    if result is None:
        return False, False
    if (result.get('prompt_sha256') != PROMPT_SHA256 or
            result.get('model') != model):
        return False, False

    cached_size = result.get('caption_size')
    cached_mtime_ns = result.get('caption_mtime_ns')
    if cached_size is None or cached_mtime_ns is None:
        # Older cache records contain a content digest but no file metadata.
        # Trust them once: re-reading every small TXT would defeat fast resume.
        return True, True
    stat = caption_path.stat()
    return (
        cached_size == stat.st_size and
        cached_mtime_ns == stat.st_mtime_ns
    ), False


def filter_captions(source_root, output_root, classifier, model,
                    batch_size=64, captions_only=False, overwrite=False,
                    max_samples=0):
    source_root = Path(source_root)
    output_root = Path(output_root)
    LOGGER.info('Scanning caption directory: %s', source_root / 'captions')
    scan_start = time.perf_counter()
    caption_files = sorted((source_root / 'captions').glob('*.txt'))
    total_caption_files = len(caption_files)
    if max_samples:
        caption_files = caption_files[:max_samples]
    if not caption_files:
        raise FileNotFoundError(
            f'No caption TXT files found under {source_root / "captions"}')
    if batch_size < 1:
        raise ValueError('batch_size must be positive')
    LOGGER.info(
        'Caption scan completed in %.2fs: found=%d, selected=%d',
        time.perf_counter() - scan_start, total_caption_files,
        len(caption_files))

    output_root.mkdir(parents=True, exist_ok=True)
    cache_path = output_root / 'classification.jsonl'
    LOGGER.info('Loading classification cache: %s', cache_path)
    cache_start = time.perf_counter()
    cached = load_cached_results(cache_path)
    LOGGER.info(
        'Classification cache loaded in %.2fs: records=%d',
        time.perf_counter() - cache_start, len(cached))
    pending = []
    matched = 0
    cached_count = 0
    copy_errors = 0

    # Validate cache entries against caption content so edited captions are
    # automatically classified again.
    LOGGER.info(
        'Validating cache metadata; caption contents are read lazily per batch')
    validation_start = time.perf_counter()
    legacy_cache_count = 0
    for caption_path in tqdm(
            caption_files, unit='caption', desc='Validating cache'):
        image_id = caption_path.stem
        result = cached.get(image_id)
        is_current, is_legacy = cached_result_is_current(
            result, caption_path, model)
        if is_current:
            legacy_cache_count += int(is_legacy)
            cached_count += 1
            if result.get('match'):
                matched += 1
                try:
                    copy_matching_sample(
                        source_root, output_root, image_id,
                        captions_only=captions_only, overwrite=overwrite)
                except Exception as error:
                    copy_errors += 1
                    LOGGER.error('Could not copy cached sample %s: %s',
                                 image_id, error)
        else:
            pending.append(caption_path)

    LOGGER.info(
        'Caption validation completed in %.2fs: captions=%d, cached=%d, '
        'legacy_cached=%d, pending=%d, cached_matches=%d, copy_errors=%d',
        time.perf_counter() - validation_start, len(caption_files),
        cached_count, legacy_cache_count, len(pending), matched, copy_errors)
    if legacy_cache_count:
        LOGGER.warning(
            'Trusted %d legacy cache records without re-reading caption '
            'contents. Newly written records include size/mtime validation.',
            legacy_cache_count)
    num_batches = math.ceil(len(pending) / batch_size) if pending else 0
    LOGGER.info(
        'Starting classification: batches=%d, captions_per_request=%d',
        num_batches, batch_size)
    progress = tqdm(total=len(pending), unit='caption', desc='Classifying')
    try:
        with cache_path.open('a', encoding='utf-8') as cache_file:
            for start in range(0, len(pending), batch_size):
                batch_paths = pending[start:start + batch_size]
                batch_index = start // batch_size + 1
                LOGGER.info(
                    'Batch %d/%d started: captions=%d, image_ids=%s..%s',
                    batch_index, num_batches, len(batch_paths),
                    batch_paths[0].stem, batch_paths[-1].stem)
                batch_start = time.perf_counter()
                read_start = time.perf_counter()
                batch = [
                    (caption_path.stem,
                     caption_path.read_text(encoding='utf-8').strip(),
                     caption_path,
                     caption_path.stat())
                    for caption_path in batch_paths
                ]
                LOGGER.info(
                    'Batch %d/%d captions read in %.2fs: bytes=%d',
                    batch_index, num_batches,
                    time.perf_counter() - read_start,
                    sum(stat.st_size for _, _, _, stat in batch))
                labels_per_caption = classifier.classify(
                    [caption for _, caption, _, _ in batch])
                if len(labels_per_caption) != len(batch):
                    raise RuntimeError(
                        'Classifier returned a different number of results')

                for batch_item, labels in zip(batch, labels_per_caption):
                    image_id, caption, caption_path, caption_stat = batch_item
                    result = build_result(
                        image_id, caption, labels, model,
                        caption_path=caption_path,
                        caption_stat=caption_stat)
                    cache_file.write(
                        json.dumps(result, ensure_ascii=False) + '\n')
                    if result['match']:
                        matched += 1
                        try:
                            copy_matching_sample(
                                source_root, output_root, image_id,
                                captions_only=captions_only,
                                overwrite=overwrite)
                        except Exception as error:
                            copy_errors += 1
                            LOGGER.error('Could not copy sample %s: %s',
                                         image_id, error)
                    progress.update(1)
                cache_file.flush()
                LOGGER.info(
                    'Batch %d/%d finished in %.2fs: matched=%d, '
                    'total_matched=%d, copy_errors=%d',
                    batch_index, num_batches,
                    time.perf_counter() - batch_start,
                    sum(bool(labels) for labels in labels_per_caption),
                    matched, copy_errors)
    finally:
        progress.close()

    LOGGER.info(
        'Done: matched=%d/%d, classified=%d, cached=%d, copy_errors=%d',
        matched, len(caption_files), len(pending), cached_count, copy_errors)
    return {
        'total': len(caption_files),
        'matched': matched,
        'classified': len(pending),
        'cached': cached_count,
        'copy_errors': copy_errors,
    }


def parse_args():
    parser = argparse.ArgumentParser(
        description='Filter S3OD captions for real people or animals')
    parser.add_argument('--source-root', type=Path, required=True,
                        help='Extracted root containing images/masks/captions')
    parser.add_argument('--output-root', type=Path, required=True,
                        help='Destination for matching samples and cache')
    parser.add_argument('--model-path', type=Path, required=True,
                        help='Local Qwen2.5 model directory')
    parser.add_argument('--device', default='auto',
                        help='auto for multi-GPU dispatch, or cuda:0/cpu')
    parser.add_argument(
        '--dtype', default='auto',
        choices=['auto', 'float16', 'bfloat16', 'float32'],
        help='Model weight dtype')
    parser.add_argument('--batch-size', type=int, default=64,
                        help='Captions classified in one model request')
    parser.add_argument(
        '--max-new-tokens', type=int, default=2,
        help='Compatibility option; constrained classification always uses 2')
    parser.add_argument(
        '--heartbeat-seconds', type=int, default=30,
        help='Log while generation is running; 0 disables heartbeat')
    parser.add_argument('--captions-only', action='store_true',
                        help='Copy only matching TXT files')
    parser.add_argument('--overwrite', action='store_true',
                        help='Overwrite existing copied files')
    parser.add_argument('--max-samples', type=int, default=0,
                        help='Only consider the first N captions; 0 means all')
    parser.add_argument(
        '--log-level', default='INFO',
        choices=['DEBUG', 'INFO', 'WARNING', 'ERROR'])
    return parser.parse_args()


def main():
    args = parse_args()
    logging.basicConfig(
        level=getattr(logging, args.log_level),
        format='%(asctime)s - %(levelname)s - %(message)s')
    model_path = args.model_path.expanduser().resolve()
    classifier = LocalTransformersClassifier(
        model_path=model_path,
        device=args.device,
        dtype=args.dtype,
        max_new_tokens=args.max_new_tokens,
        heartbeat_seconds=args.heartbeat_seconds)
    filter_captions(
        source_root=args.source_root,
        output_root=args.output_root,
        classifier=classifier,
        model=str(model_path),
        batch_size=args.batch_size,
        captions_only=args.captions_only,
        overwrite=args.overwrite,
        max_samples=args.max_samples)


if __name__ == '__main__':
    main()
