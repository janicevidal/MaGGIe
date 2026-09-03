from .him import HIMDataset
from .vim import VIMDataset
from .human_matting import MattingBatchSampler, MattingDataset
from .binary_segmentation import (
    BinarySegmentationBatchSampler,
    BinarySegmentationDataset,
)
from .mixed_supervision import (
    MixedSupervisionBatchSampler,
    MixedSupervisionDataset,
)


def build_dataset(cfg, is_train=True, random_seed=0):
    if cfg.name in ["HIM"]:
        if is_train:
            dataset = HIMDataset(root_dir=cfg.root_dir, split=cfg.split, max_inst=cfg.max_inst, short_size=cfg.short_size, 
                                 crop=cfg.crop, is_train=is_train, random_seed=random_seed, alpha_dir_name=cfg.alpha_dir_name, mask_dir_name=cfg.mask_dir_name,
                                 padding_crop_p=cfg.padding_crop_p, flip_p=cfg.flip_p, gamma_p=cfg.gamma_p, add_noise_p=cfg.add_noise_p, jpeg_p=cfg.jpeg_p, affine_p=cfg.affine_p, 
                                 binarized_kernel=cfg.binarized_kernel, downscale_mask_p=cfg.downscale_mask_p)
        else:
            dataset = HIMDataset(root_dir=cfg.root_dir, split=cfg.split, short_size=cfg.short_size, is_train=is_train, 
                                 downscale_mask_p=0 if cfg.downscale_mask else 1, alpha_dir_name=cfg.alpha_dir_name, mask_dir_name=cfg.mask_dir_name)
    elif cfg.name in ["VIM"]:
        if is_train:
            dataset = VIMDataset(root_dir=cfg.root_dir, split=cfg.split, is_train=is_train, alpha_dir_name=cfg.alpha_dir_name, mask_dir_name=cfg.mask_dir_name,
                                 clip_length=cfg.clip_length, max_step_size=cfg.max_step_size, max_inst=cfg.max_inst, short_size=cfg.short_size, crop=cfg.crop, 
                                 padding_crop_p=cfg.padding_crop_p, flip_p=cfg.flip_p, gamma_p=cfg.gamma_p, motion_p=cfg.motion_p, add_noise_p=cfg.add_noise_p, 
                                 jpeg_p=cfg.jpeg_p, affine_p=cfg.affine_p, binarized_kernel=cfg.binarized_kernel, downscale_mask_p=cfg.downscale_mask_p, random_seed=random_seed)
        else:
            dataset = VIMDataset(root_dir=cfg.root_dir, split=cfg.split, clip_length=cfg.clip_length, overlap=cfg.clip_overlap, is_train=is_train, 
                                 short_size=cfg.short_size, mask_dir_name=cfg.mask_dir_name, alpha_dir_name=cfg.alpha_dir_name)
    elif cfg.name in ["Matting"]:
        root_dir = cfg.root_dirs if cfg.root_dirs else cfg.root_dir
        if is_train:
            dataset = MattingDataset(root_dir=root_dir, split=cfg.split, short_size=cfg.short_size, crop=cfg.crop, is_train=is_train, random_seed=random_seed,
                                     alpha_dir_name=cfg.alpha_dir_name, padding_crop_p=cfg.padding_crop_p, flip_p=cfg.flip_p, gamma_p=cfg.gamma_p, add_noise_p=cfg.add_noise_p, 
                                     jpeg_p=cfg.jpeg_p, affine_p=cfg.affine_p,
                                     root_sampling_rates=getattr(cfg, 'root_sampling_rates', []))
        else:
            dataset = MattingDataset(root_dir=root_dir, split=cfg.split, short_size=cfg.short_size, is_train=is_train, alpha_dir_name=cfg.alpha_dir_name)
    elif cfg.name in ["BinarySegmentation", "Segmentation"]:
        root_dir = cfg.root_dirs if cfg.root_dirs else cfg.root_dir
        if is_train:
            dataset = BinarySegmentationDataset(
                root_dir=root_dir, split=cfg.split,
                short_size=cfg.short_size, crop=cfg.crop,
                is_train=is_train, random_seed=random_seed,
                mask_dir_name=cfg.mask_dir_name,
                mask_threshold=cfg.mask_threshold,
                root_sampling_rates=getattr(cfg, 'root_sampling_rates', []),
                padding_crop_p=cfg.padding_crop_p, flip_p=cfg.flip_p,
                gamma_p=cfg.gamma_p, add_noise_p=cfg.add_noise_p,
                jpeg_p=cfg.jpeg_p, affine_p=cfg.affine_p)
        else:
            dataset = BinarySegmentationDataset(
                root_dir=root_dir, split=cfg.split,
                short_size=cfg.short_size, is_train=is_train,
                mask_dir_name=cfg.mask_dir_name,
                mask_threshold=cfg.mask_threshold)
    elif cfg.name == "MixedMattingSegmentation":
        if not is_train:
            raise ValueError(
                "MixedMattingSegmentation is only supported for training")
        matting_root = cfg.root_dirs if cfg.root_dirs else cfg.root_dir
        binary_root = (
            cfg.binary_root_dirs if cfg.binary_root_dirs
            else cfg.binary_root_dir)
        matting_dataset = MattingDataset(
            root_dir=matting_root, split=cfg.split,
            short_size=cfg.short_size, crop=cfg.crop, is_train=True,
            random_seed=random_seed, alpha_dir_name=cfg.alpha_dir_name,
            padding_crop_p=cfg.padding_crop_p, flip_p=cfg.flip_p,
            gamma_p=cfg.gamma_p, add_noise_p=cfg.add_noise_p,
            jpeg_p=cfg.jpeg_p, affine_p=cfg.affine_p)
        binary_dataset = BinarySegmentationDataset(
            root_dir=binary_root, split=cfg.split,
            short_size=cfg.short_size, crop=cfg.crop, is_train=True,
            random_seed=random_seed + 1,
            mask_dir_name=cfg.binary_mask_dir_name,
            mask_threshold=cfg.binary_mask_threshold,
            root_sampling_rates=getattr(
                cfg, 'binary_root_sampling_rates', []),
            padding_crop_p=cfg.padding_crop_p, flip_p=cfg.flip_p,
            gamma_p=cfg.gamma_p, add_noise_p=cfg.add_noise_p,
            jpeg_p=cfg.jpeg_p, affine_p=cfg.affine_p)
        dataset = MixedSupervisionDataset(
            matting_dataset, binary_dataset,
            binary_ratio=cfg.binary_ratio,
            epoch_size=cfg.mixed_epoch_size,
            random_seed=random_seed)
    else:
        raise NotImplementedError
    return dataset
