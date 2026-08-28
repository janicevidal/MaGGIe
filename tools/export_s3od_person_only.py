"""Export person-only samples from completed S3OD classifications."""

import argparse
import json
import logging
from pathlib import Path

from tqdm import tqdm

from filter_s3od_captions import copy_matching_sample, load_cached_results


LOGGER = logging.getLogger(__name__)


def classification_flags(result):
    """Read flags while remaining compatible with older cache records."""
    labels = set(result.get('labels') or [])
    contains_person = bool(
        result.get('contains_person', 'person' in labels))
    contains_animal = bool(
        result.get('contains_animal', 'animal' in labels))
    return contains_person, contains_animal


def select_person_only(results):
    """Select records containing a person but no animal."""
    selected = []
    counts = {
        'person_only': 0,
        'animal_only': 0,
        'both': 0,
        'neither': 0,
    }
    for image_id, result in results.items():
        contains_person, contains_animal = classification_flags(result)
        if contains_person and not contains_animal:
            counts['person_only'] += 1
            selected.append((image_id, result))
        elif contains_animal and not contains_person:
            counts['animal_only'] += 1
        elif contains_person and contains_animal:
            counts['both'] += 1
        else:
            counts['neither'] += 1
    selected.sort(key=lambda item: item[0])
    return selected, counts


def write_selection_manifest(output_root, selected):
    manifest_path = output_root / 'selection.jsonl'
    temporary_path = manifest_path.with_suffix('.jsonl.tmp')
    with temporary_path.open('w', encoding='utf-8') as manifest:
        for _, result in selected:
            manifest.write(json.dumps(result, ensure_ascii=False) + '\n')
    temporary_path.replace(manifest_path)
    return manifest_path


def export_person_only(classification_root, output_root, overwrite=False,
                       captions_only=False, max_samples=0):
    classification_root = Path(classification_root)
    output_root = Path(output_root)
    cache_path = classification_root / 'classification.jsonl'
    if not cache_path.is_file():
        raise FileNotFoundError(cache_path)
    if max_samples < 0:
        raise ValueError('max_samples cannot be negative')

    LOGGER.info('Loading final classifications from %s', cache_path)
    results = load_cached_results(cache_path)
    selected, counts = select_person_only(results)
    if max_samples:
        selected = selected[:max_samples]
    LOGGER.info(
        'Classification summary: total=%d, person_only=%d, '
        'animal_only=%d, both=%d, neither=%d',
        len(results), counts['person_only'], counts['animal_only'],
        counts['both'], counts['neither'])
    LOGGER.info('Exporting %d person-only samples to %s',
                len(selected), output_root)

    output_root.mkdir(parents=True, exist_ok=True)
    manifest_path = write_selection_manifest(output_root, selected)
    copied = 0
    errors = 0
    for image_id, _ in tqdm(
            selected, unit='sample', desc='Exporting person-only'):
        try:
            copy_matching_sample(
                classification_root, output_root, image_id,
                captions_only=captions_only, overwrite=overwrite)
            copied += 1
        except Exception as error:
            errors += 1
            LOGGER.error('Could not export %s: %s', image_id, error)

    LOGGER.info(
        'Export finished: selected=%d, completed=%d, errors=%d, manifest=%s',
        len(selected), copied, errors, manifest_path)
    return {
        'selected': len(selected),
        'completed': copied,
        'errors': errors,
        'counts': counts,
        'manifest': manifest_path,
    }


def parse_args():
    parser = argparse.ArgumentParser(
        description='Export S3OD samples classified as person without animal')
    parser.add_argument(
        '--classification-root', type=Path, required=True,
        help='Root containing classification.jsonl and matched samples')
    parser.add_argument('--output-root', type=Path, required=True)
    parser.add_argument('--captions-only', action='store_true')
    parser.add_argument('--overwrite', action='store_true')
    parser.add_argument('--max-samples', type=int, default=0,
                        help='Export at most N samples; 0 means all')
    parser.add_argument(
        '--log-level', default='INFO',
        choices=['DEBUG', 'INFO', 'WARNING', 'ERROR'])
    return parser.parse_args()


def main():
    args = parse_args()
    logging.basicConfig(
        level=getattr(logging, args.log_level),
        format='%(asctime)s - %(levelname)s - %(message)s')
    result = export_person_only(
        classification_root=args.classification_root,
        output_root=args.output_root,
        overwrite=args.overwrite,
        captions_only=args.captions_only,
        max_samples=args.max_samples)
    if result['errors']:
        raise SystemExit(1)


if __name__ == '__main__':
    main()
