import math

import numpy as np
import torch
from torch.utils.data import Dataset, Sampler


class MixedSupervisionDataset(Dataset):
    """Mix matting and binary samples behind one collatable interface.

    ``alpha`` contains the alpha matte for matting samples and the person mask
    for binary samples. ``is_matting`` tells the model whether alpha-specific
    losses are allowed for that sample.
    """

    def __init__(self, matting_dataset, binary_dataset, binary_ratio=0.3,
                 epoch_size=0, random_seed=2023):
        if len(matting_dataset) == 0:
            raise ValueError("The matting dataset is empty")
        if len(binary_dataset) == 0:
            raise ValueError("The binary person dataset is empty")
        if not 0 < binary_ratio < 1:
            raise ValueError("binary_ratio must be between 0 and 1")

        self.matting_dataset = matting_dataset
        self.binary_dataset = binary_dataset
        self.binary_ratio = binary_ratio

        if epoch_size <= 0:
            epoch_size = max(
                math.ceil(len(matting_dataset) / (1 - binary_ratio)),
                math.ceil(len(binary_dataset) / binary_ratio))
        self.epoch_size = int(epoch_size)

        random = np.random.RandomState(random_seed)
        num_binary = min(
            max(int(round(self.epoch_size * binary_ratio)), 1),
            self.epoch_size - 1)
        num_matting = self.epoch_size - num_binary

        sample_types = np.concatenate([
            np.ones(num_matting, dtype=np.bool_),
            np.zeros(num_binary, dtype=np.bool_)
        ])
        random.shuffle(sample_types)

        matting_indices = self._cycled_indices(
            len(matting_dataset), num_matting, random)
        binary_indices = self._cycled_indices(
            len(binary_dataset), num_binary, random)

        self.samples = []
        matting_offset = 0
        binary_offset = 0
        for is_matting in sample_types:
            if is_matting:
                self.samples.append((True, matting_indices[matting_offset]))
                matting_offset += 1
            else:
                self.samples.append((False, binary_indices[binary_offset]))
                binary_offset += 1

    @staticmethod
    def _cycled_indices(dataset_size, num_samples, random):
        indices = []
        while len(indices) < num_samples:
            permutation = random.permutation(dataset_size).tolist()
            indices.extend(permutation)
        return indices[:num_samples]

    def __len__(self):
        return self.epoch_size

    def __getitem__(self, index):
        is_matting, sample_index = self.samples[index]
        if is_matting:
            sample = self.matting_dataset[sample_index]
            target = sample['alpha']
        else:
            sample = self.binary_dataset[sample_index]
            target = sample['mask']

        return {
            'image': sample['image'],
            'alpha': target,
            'is_matting': torch.tensor(is_matting, dtype=torch.bool)
        }


class MixedSupervisionBatchSampler(Sampler):
    """Build equally composed local batches for single- or multi-GPU train.

    Keeping at least one matting sample on every rank is important when the
    decoder contains SyncBatchNorm: all ranks must enter the decoder in the
    same training step. Pools are reshuffled and cycled independently each
    epoch, so smaller datasets can still participate at the requested ratio.
    """

    def __init__(self, dataset, batch_size, num_replicas=1, rank=0,
                 seed=0):
        if not isinstance(dataset, MixedSupervisionDataset):
            raise TypeError(
                "MixedSupervisionBatchSampler requires a mixed dataset")
        if batch_size < 2:
            raise ValueError(
                "Mixed supervision requires train.batch_size >= 2 so each "
                "rank receives both matting and binary samples")
        if num_replicas < 1:
            raise ValueError("num_replicas must be positive")
        if not 0 <= rank < num_replicas:
            raise ValueError("rank must be in [0, num_replicas)")

        self.dataset = dataset
        self.batch_size = int(batch_size)
        self.num_replicas = int(num_replicas)
        self.rank = int(rank)
        self.seed = int(seed)
        self.epoch = 0

        self.matting_indices = [
            index for index, (is_matting, _) in enumerate(dataset.samples)
            if is_matting
        ]
        self.binary_indices = [
            index for index, (is_matting, _) in enumerate(dataset.samples)
            if not is_matting
        ]
        self.num_binary = min(
            max(int(round(batch_size * dataset.binary_ratio)), 1),
            batch_size - 1)
        self.num_matting = batch_size - self.num_binary
        global_batch_size = batch_size * num_replicas
        self.num_batches = max(
            1, math.ceil(len(dataset) / global_batch_size))

    def set_epoch(self, epoch):
        self.epoch = int(epoch)

    @staticmethod
    def _infinite_shuffled(indices, random):
        while True:
            for index in random.permutation(indices).tolist():
                yield index

    def __iter__(self):
        random = np.random.RandomState(self.seed + self.epoch)
        matting_pool = self._infinite_shuffled(
            self.matting_indices, random)
        binary_pool = self._infinite_shuffled(
            self.binary_indices, random)

        for _ in range(self.num_batches):
            global_matting = [
                next(matting_pool)
                for _ in range(self.num_matting * self.num_replicas)
            ]
            global_binary = [
                next(binary_pool)
                for _ in range(self.num_binary * self.num_replicas)
            ]
            matting_start = self.rank * self.num_matting
            binary_start = self.rank * self.num_binary
            local_batch = (
                global_matting[
                    matting_start:matting_start + self.num_matting] +
                global_binary[
                    binary_start:binary_start + self.num_binary]
            )
            random.shuffle(local_batch)
            yield local_batch

    def __len__(self):
        return self.num_batches
