# maggie/dataloader/birefnet_dataset.py
import os
import glob
import numpy as np
from PIL import Image
from torch.utils.data import Dataset

from . import transforms as T

class MattingDataset(Dataset):
    def __init__(self, root_dir, short_size=768, is_train=False, random_seed=2023, \
                crop=(512, 512), padding_crop_p=0.1, flip_p=0.5, gamma_p=0.3, add_noise_p=0.3, jpeg_p=0.1, affine_p=0.8, \
                alpha_dir_name='alphas', **kwargs):
        self.is_train = is_train
        self.root_dir = root_dir
        self.alpha_dir_name = alpha_dir_name
        self.is_train = is_train

        self.short_size = short_size
        self.random = np.random.RandomState(random_seed)
        
        self.valid_image_extensions = ['.jpg', '.jpeg', '.png', '.JPG', '.JPEG', '.PNG']
        
        self.prepare_image()

        self.transforms = [
            T.Load(),
        ]
        
        if self.is_train:
            self.transforms += [
                T.Stack(),
                T.RandomAffineCrop(crop, self.random, p=affine_p, angle_range=(-15, 15), scale_range=(0.9, 1.1), shift_ratio=0.2),
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

    def prepare_image(self):
        image_dir = os.path.join(self.root_dir, "images")
        if not os.path.isdir(image_dir):
            self.data = []
            return
        all_files = os.listdir(image_dir)
        images = [os.path.join(image_dir, f) for f in all_files 
                  if any(f.endswith(ext) for ext in self.valid_image_extensions)]
        images.sort()
        
        all_alphas = []
        valid_images = []
        for image in images:
            image_name = os.path.basename(image).split('.')[0]
            alpha_dir = os.path.join(self.root_dir, self.alpha_dir_name)

            alpha = os.path.join(alpha_dir, image_name + ".png")
            
            if not os.path.exists(alpha):
                continue
            
            valid_images.append(image)
            all_alphas.append(alpha)
        
        self.data = list(zip(valid_images, all_alphas))

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