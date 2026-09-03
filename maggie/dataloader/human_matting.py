import math
import os

import numpy as np
from torch.utils.data import Dataset, Sampler

from . import transforms as T
from .binary_segmentation import (
    build_sampling_schedule,
    infinite_shuffled,
)


class MattingDataset(Dataset):
    """Single-image matting dataset supporting one or more dataset roots."""

    def __init__(self, root_dir, short_size=768, is_train=False,
                 random_seed=2023, crop=(512, 512), padding_crop_p=0.1,
                 flip_p=0.5, gamma_p=0.3, add_noise_p=0.3, jpeg_p=0.1,
                 affine_p=0.8, keep_whole_p=0.7, alpha_dir_name='alphas',
                 root_sampling_rates=None, **kwargs):
        self.root_dir = root_dir
        if isinstance(root_dir, (str, os.PathLike)):
            self.root_dirs = [os.fspath(root_dir)]
        else:
            self.root_dirs = [os.fspath(path) for path in root_dir]
        self.alpha_dir_name = alpha_dir_name
        self.is_train = is_train
        self.short_size = short_size
        self.random = np.random.RandomState(random_seed)
        self.uses_root_sampling_rates = bool(root_sampling_rates)
        self.root_sampling_rates = self._validate_sampling_rates(
            root_sampling_rates)
        self.valid_image_extensions = [
            '.jpg', '.jpeg', '.png', '.JPG', '.JPEG', '.PNG'
        ]

        self.prepare_image()

        self.transforms = [
            T.Load(),
        ]
        
        if self.is_train:
            self.transforms += [
                T.Stack(),
                # T.RandomAffineCrop(crop, self.random, p=affine_p, angle_range=(-15, 15), scale_range=(0.9, 1.1), shift_ratio=0.2),
                T.RandomAffineCrop(
                    crop, self.random, p=affine_p,
                    angle_range=(-15, 15), scale_range=(0.9, 1.1),
                    shift_ratio=0.1, keep_whole_p=keep_whole_p),
                T.RandomHorizontalFlip(self.random, flip_p),
                T.GammaContrast(self.random, p=gamma_p),
                T.AdditiveGaussionNoise(self.random, p=add_noise_p),
                T.JpegCompression(self.random, p=jpeg_p)
            ]
        else:
            self.transforms += [
                T.ResizeShort(short_size),
                T.PaddingMultiplyBy(32),
                T.Stack()
            ]

        self.transforms += [
            T.ToTensor(),
            T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ]
        
        self.transforms = T.Compose(self.transforms)

    def _validate_sampling_rates(self, sampling_rates):
        if sampling_rates is None or len(sampling_rates) == 0:
            return None
        if len(sampling_rates) != len(self.root_dirs):
            raise ValueError(
                "root_sampling_rates must have one value for every root_dir "
                f"({len(sampling_rates)} rates for {len(self.root_dirs)} "
                "directories)")

        rates = np.asarray(sampling_rates, dtype=np.float64)
        if not np.all(np.isfinite(rates)) or np.any(rates < 0):
            raise ValueError(
                "root_sampling_rates must contain finite non-negative values")
        if rates.sum() <= 0:
            raise ValueError(
                "At least one root_sampling_rates value must be positive")
        return rates

    def prepare_image(self):
        self.data = []
        self.root_indices = []

        # Pair samples within each root first, then concatenate all roots while
        # retaining their configured order.
        for root_dir in self.root_dirs:
            indices = []
            image_dir = os.path.join(root_dir, "images")
            if not os.path.isdir(image_dir):
                self.root_indices.append(indices)
                continue

            images = [
                os.path.join(image_dir, file_name)
                for file_name in os.listdir(image_dir)
                if any(file_name.endswith(ext) for ext in self.valid_image_extensions)
            ]
            images.sort()

            alpha_dir = os.path.join(root_dir, self.alpha_dir_name)
            for image in images:
                image_name = os.path.splitext(os.path.basename(image))[0]
                alpha = os.path.join(alpha_dir, image_name + ".png")
                if os.path.exists(alpha):
                    indices.append(len(self.data))
                    self.data.append((image, alpha))
            self.root_indices.append(indices)

        if self.root_sampling_rates is not None:
            empty_roots = [
                self.root_dirs[index]
                for index, (rate, indices) in enumerate(zip(
                    self.root_sampling_rates, self.root_indices))
                if rate > 0 and not indices
            ]
            if empty_roots:
                raise ValueError(
                    "Positive sampling rates were assigned to roots without "
                    f"valid image/alpha pairs: {empty_roots}")

    def sampling_groups(self):
        """Return root pools weighted by size times configured sample rate."""
        groups = [indices for indices in self.root_indices if indices]
        if not groups:
            raise ValueError("The matting dataset is empty")

        if self.root_sampling_rates is None:
            rates = np.ones(len(groups), dtype=np.float64)
        else:
            rates = np.asarray([
                rate for rate, indices in zip(
                    self.root_sampling_rates, self.root_indices)
                if indices
            ], dtype=np.float64)
            positive = rates > 0
            groups = [
                group for group, keep in zip(groups, positive) if keep
            ]
            rates = rates[positive]
        weights = np.asarray([
            len(group) * rate for group, rate in zip(groups, rates)
        ], dtype=np.float64)
        return groups, weights / weights.sum()

    def effective_epoch_size(self):
        """Number of samples in one epoch after applying per-root rates."""
        if self.root_sampling_rates is None:
            return len(self.data)
        size = sum(
            len(indices) * rate
            for indices, rate in zip(
                self.root_indices, self.root_sampling_rates)
        )
        return max(1, int(round(size)))

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        image_path, alpha_path = self.data[index]
        
        # Load image
        input_dict = {
            "frames": [image_path],
            "alphas": [alpha_path],
            "masks": None,
            "weights": None
        }
        
        output_dict = self.transforms(input_dict)
        
        image, alpha, transform_info = output_dict["frames"][0], output_dict["alphas"][0], output_dict["transform_info"]
        
        if not self.is_train:
            alpha = output_dict["ori_alphas"][0]
            
        alpha = alpha * 1.0 / 255
        
        out = {
            'image': image, 
            'alpha': alpha.float(),
        }
        
        if not self.is_train:
            out.update({'image_names': [image_path], 
                        'alpha_names': [alpha_path], 
                        'transform_info': transform_info, 
                        "skip": 0})    
        
        return out


class MattingBatchSampler(Sampler):
    """Sample a configured fraction of each matting root per epoch."""

    def __init__(self, dataset, batch_size, num_replicas=1, rank=0, seed=0):
        if not isinstance(dataset, MattingDataset):
            raise TypeError("MattingBatchSampler requires a matting dataset")
        if batch_size < 1:
            raise ValueError("batch_size must be positive")
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
        self.groups, self.rates = dataset.sampling_groups()
        global_batch_size = self.batch_size * self.num_replicas
        self.effective_num_samples = dataset.effective_epoch_size()
        self.num_batches = max(
            1, math.ceil(self.effective_num_samples / global_batch_size))

    def set_epoch(self, epoch):
        self.epoch = int(epoch)

    def __iter__(self):
        random = np.random.RandomState(self.seed + self.epoch)
        total_samples = (
            self.num_batches * self.batch_size * self.num_replicas)
        schedule = build_sampling_schedule(
            self.rates, total_samples, random,
            batch_size=self.batch_size * self.num_replicas)
        pools = [infinite_shuffled(group, random) for group in self.groups]

        global_batch_size = self.batch_size * self.num_replicas
        for batch_index in range(self.num_batches):
            start = batch_index * global_batch_size
            group_ids = schedule[start:start + global_batch_size]
            global_batch = [next(pools[group_id]) for group_id in group_ids]
            local_start = self.rank * self.batch_size
            yield global_batch[local_start:local_start + self.batch_size]

    def __len__(self):
        return self.num_batches
