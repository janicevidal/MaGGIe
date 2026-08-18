import os

import numpy as np
from torch.utils.data import Dataset

from . import transforms as T


class BinarySegmentationDataset(Dataset):
    """Single-image binary segmentation dataset.

    Each dataset root contains an ``images`` directory and a mask directory.
    Images and PNG masks are paired by filename stem.
    """

    def __init__(self, root_dir, short_size=768, is_train=False,
                 random_seed=2023, crop=(512, 512), padding_crop_p=0.1,
                 flip_p=0.5, gamma_p=0.3, add_noise_p=0.3, jpeg_p=0.1,
                 affine_p=0.8, mask_dir_name='masks', **kwargs):
        self.root_dir = root_dir
        if isinstance(root_dir, (str, os.PathLike)):
            self.root_dirs = [os.fspath(root_dir)]
        else:
            self.root_dirs = [os.fspath(path) for path in root_dir]
        self.mask_dir_name = mask_dir_name
        self.is_train = is_train
        self.random = np.random.RandomState(random_seed)
        self.valid_image_extensions = [
            '.jpg', '.jpeg', '.png', '.JPG', '.JPEG', '.PNG'
        ]

        self.prepare_image()

        transforms = [T.Load()]
        if self.is_train:
            transforms += [
                T.Stack(),
                T.RandomAffineCrop(
                    crop, self.random, p=affine_p,
                    angle_range=(-15, 15), scale_range=(0.9, 1.1),
                    shift_ratio=0.1),
                T.RandomHorizontalFlip(self.random, flip_p),
                T.GammaContrast(self.random, p=gamma_p),
                T.AdditiveGaussionNoise(self.random, p=add_noise_p),
                T.JpegCompression(self.random, p=jpeg_p)
            ]
        else:
            transforms += [
                T.ResizeShort(short_size),
                T.PaddingMultiplyBy(32),
                T.Stack()
            ]
        transforms += [
            T.ToTensor(),
            T.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225])
        ]
        self.transforms = T.Compose(transforms)

    def prepare_image(self):
        self.data = []
        for root_dir in self.root_dirs:
            image_dir = os.path.join(root_dir, 'images')
            if not os.path.isdir(image_dir):
                continue

            images = [
                os.path.join(image_dir, file_name)
                for file_name in os.listdir(image_dir)
                if any(file_name.endswith(ext)
                       for ext in self.valid_image_extensions)
            ]
            images.sort()

            mask_dir = os.path.join(root_dir, self.mask_dir_name)
            for image_path in images:
                image_name = os.path.splitext(os.path.basename(image_path))[0]
                mask_path = os.path.join(mask_dir, image_name + '.png')
                if os.path.exists(mask_path):
                    self.data.append((image_path, mask_path))

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        image_path, mask_path = self.data[index]
        input_dict = {
            'frames': [image_path],
            'alphas': None,
            'masks': [mask_path],
            'weights': None
        }
        output_dict = self.transforms(input_dict)

        image = output_dict['frames'][0]
        mask = output_dict['masks'][0]
        if not self.is_train:
            mask = output_dict['ori_masks'][0]
        # Binary datasets commonly store foreground as either 1 or 255.
        mask = (mask > 0).float()

        out = {
            'image': image,
            'mask': mask
        }
        if not self.is_train:
            out.update({
                'image_names': [image_path],
                'mask_names': [mask_path],
                'transform_info': output_dict['transform_info'],
                'skip': 0
            })
        return out
