import os
import itertools
import glob
import time
import torch
import logging
import numpy as np
import wandb
import torchvision.utils as vutils
from torch.utils import data as torch_data
from torch.cuda.amp import autocast, GradScaler

from maggie.dataloader import build_dataset
from maggie.network import build_model
from maggie.utils.metric import build_metric

from .optim import build_optim_lr_scheduler, plot_lr_scheduler
from .test import eval_image, eval_video

from PIL import Image
from mmengine.logging import MMLogger, HistoryBuffer
from torch.utils.tensorboard import SummaryWriter


def log_alpha(tensor, tag, index=0, inst_idx=0):
    if tensor.dim() == 5:   # (B, T, N, H, W)
        alpha = tensor[0, index, inst_idx]
    elif tensor.dim() == 4: # (B, N, H, W) 或 (B, T, H, W) 或 (B*T, 1, H, W)
        if tensor.shape[1] == 1:
            alpha = tensor[0, 0]
        else:
            alpha = tensor[0, inst_idx] if tensor.shape[1] > inst_idx else tensor[0, 0]
    elif tensor.dim() == 3:  # (B, H, W) 或 (T, H, W)
        alpha = tensor[0]
    else:
        raise ValueError(f"Unsupported tensor shape: {tensor.shape}")
    
    alpha = alpha.detach().cpu().numpy()
    if alpha.ndim == 2:
        alpha = (alpha * 255).astype('uint8')
    elif alpha.ndim == 3 and alpha.shape[0] == 1:
        alpha = alpha[0]
        alpha = (alpha * 255).astype('uint8')
    else:
        raise ValueError(f"Un-supported shape for image conversion {alpha.shape}")
    return wandb.Image(alpha, caption=tag)

def wandb_log_image(batch, output, iter):
    # Log transition_preds
    log_images = []
    index = batch['image'].shape[1] - 1
    inst_index = 0
    image = batch['image'][0].cpu()
    image = image * torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1) + torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
    image = (image * 255).permute(1, 2, 0).numpy().astype(np.uint8)
    log_images.append(wandb.Image(image, caption="image"))
    
    log_images.append(log_alpha(batch['alpha'], 'alpha_gt', index, inst_index))
    
    if 'alpha_pred' in output:
        log_images.append(log_alpha(output['alpha_pred'], 'alpha_pred', index, inst_index))

    wandb.log({"examples/all": log_images}, commit=True)


def tensorboard_log_image(batch, output, iter, writer, n_samples=5):
    if not hasattr(Image, 'ANTIALIAS'):
        Image.ANTIALIAS = Image.Resampling.LANCZOS
    
    images = batch['image']
    alphas = batch['alpha']
    
    n = min(n_samples, images.shape[0])
    
    img_list = []
    alpha_gt_list = []
    alpha_pred_list = []

    mean = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
    std = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)
    
    alpha_pred_out = output.get('alpha_pred', None)

    for i in range(n):
        img = images[i].cpu()
        img = img * std + mean
        img = torch.clamp(img, 0, 1).squeeze(0)
        img_list.append(img)

        alpha_gt = alphas[i, 0].cpu()
        alpha_gt = alpha_gt.unsqueeze(0).repeat(3, 1, 1)  # (3, H, W)
        alpha_gt_list.append(alpha_gt)

        if alpha_pred_out is not None:
            alpha_pred = alpha_pred_out[i, 0].detach().cpu()
            alpha_pred = torch.clamp(alpha_pred, 0, 1)
            alpha_pred = alpha_pred.unsqueeze(0).repeat(3, 1, 1)
            alpha_pred_list.append(alpha_pred)
    
    if alpha_pred_list:
        grid_img = vutils.make_grid(img_list, nrow=n, pad_value=1)
        grid_gt = vutils.make_grid(alpha_gt_list, nrow=n, pad_value=1)
        grid_pred = vutils.make_grid(alpha_pred_list, nrow=n, pad_value=1)
        final_grid = torch.cat([grid_img, grid_gt, grid_pred], dim=1)
    else:
        grid_img = vutils.make_grid(img_list, nrow=n, pad_value=1)
        grid_gt = vutils.make_grid(alpha_gt_list, nrow=n, pad_value=1)
        final_grid = torch.cat([grid_img, grid_gt], dim=1)

    writer.add_image('vis/comparison', final_grid, global_step=iter)


def adapt_training_batch(batch):
    """Map segmentation supervision to the alpha key used by current models."""
    if 'alpha' in batch:
        return batch
    if 'mask' in batch:
        batch['alpha'] = batch.pop('mask')
        return batch
    if 'masks' in batch:
        batch['alpha'] = batch.pop('masks')
        return batch
    raise KeyError("Training batch must contain 'alpha', 'mask', or 'masks'")


def load_state_dict(model, state_dict):
    current_state_dict = model.state_dict()
    missing_keys = []
    unexpected_keys = []
    mismatch_keys = []
    for name, param in state_dict.items():
        if name not in current_state_dict:
            unexpected_keys.append(name)
        elif param.shape != current_state_dict[name].shape:
            mismatch_keys.append(name)
        else:
            current_state_dict[name].copy_(param)
    for name in current_state_dict.keys():
        if name not in state_dict:
            missing_keys.append(name)
    
    return missing_keys, unexpected_keys, mismatch_keys

def load_resume_model(model, optimizer, lr_scheduler, resume_path, device):
    logging.info("Resuming model from {}".format(resume_path))
    if not os.path.exists(os.path.join(resume_path, 'last_model.pth')) or not os.path.exists(os.path.join(resume_path, 'last_opt.pth')):
        raise ValueError("Cannot resume model from {}".format(resume_path))
    state_dict = torch.load(os.path.join(resume_path, 'last_model.pth'), map_location=device)
    opt_dict = torch.load(os.path.join(resume_path, 'last_opt.pth'), map_location=device)
    model.load_state_dict(state_dict, strict=True)
    
    # Load optimizer and lr_scheduler
    optimizer.load_state_dict(opt_dict['optimizer'])
    lr_scheduler.load_state_dict(opt_dict['lr_scheduler'])

    # Load epoch, iteration, best score
    iter = opt_dict['iter']
    best_score = opt_dict['best_score']
    return iter, best_score

def train(cfg, rank, is_dist=False, precision=32, global_rank=None):
    if global_rank is None:
        global_rank = rank
    
    device = f'cuda:{rank}'
    
    log_file = os.path.join(cfg.output_dir, 'train.log')
    logger = MMLogger.get_instance('matting', log_file=log_file, log_level='INFO', file_mode='a')
    
    scalar_writer = None
    image_writer = None
    if global_rank == 0:
        tb_root = os.path.join(cfg.output_dir, 'tensorboard')

        scalar_dir = os.path.join(tb_root, 'scalars')
        os.makedirs(scalar_dir, exist_ok=True)
        scalar_writer = SummaryWriter(log_dir=scalar_dir)
        
        image_dir = os.path.join(tb_root, 'images')
        os.makedirs(image_dir, exist_ok=True)
        image_writer = SummaryWriter(log_dir=image_dir)
        
        logger.info(f"Scalar logs will be saved to {scalar_dir}")
        logger.info(f"Image logs will be saved to {image_dir}")

    # Create dataset
    logging.info("Creating train dataset...")
    train_dataset = build_dataset(cfg.dataset.train, is_train=True, random_seed=cfg.train.seed)

    # Create dataloader
    if is_dist:
        train_sampler = torch_data.DistributedSampler(train_dataset)
    else:
        train_sampler = None
    
    g = torch.Generator()
    g.manual_seed(cfg.train.seed)

    train_loader = torch_data.DataLoader(
        train_dataset, batch_size=cfg.train.batch_size, shuffle=(train_sampler is None),
        num_workers=cfg.train.num_workers,
        pin_memory=True, sampler=train_sampler,
        generator=g)
    
    logging.info("Creating val dataset...")
    val_dataset = build_dataset(cfg.dataset.test, is_train=False)
    val_sampler = torch_data.DistributedSampler(val_dataset, shuffle=False) if (is_dist and cfg.train.val_dist) else None
    val_loader = torch_data.DataLoader(
        val_dataset, batch_size=cfg.test.batch_size, shuffle=False, pin_memory=True,
        sampler=val_sampler,
        num_workers=cfg.test.num_workers)
    
    # Build model
    logging.info("Building model...")
    model, is_from_hf = build_model(cfg.model)
    model = model.to(device)
    training_params = sum([p.numel() for p in model.parameters() if p.requires_grad])
    logging.info("Number of trainable parameters: {}".format(training_params))

    # Define optimizer and lr scheduler
    logging.info("Building optimizer and lr scheduler...")
    optimizer, lr_scheduler = build_optim_lr_scheduler(cfg, model)
    
    plot_lr_scheduler(optimizer, lr_scheduler, total_iters=cfg.train.max_iter, save_dir=cfg.output_dir)

    if is_dist:
        if cfg.model.sync_bn:
            model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(model)
        having_unused_params = cfg.model.having_unused_params
        model = torch.nn.parallel.DistributedDataParallel(
                model, device_ids=[rank], find_unused_parameters=having_unused_params)

    epoch = 0
    iter = 0
    best_score = None

    # Load pretrained model
    if os.path.isfile(cfg.model.weights):
        logging.info("Loading pretrained model from {}".format(cfg.model.weights))
        state_dict = torch.load(cfg.model.weights, map_location=device)
        missing_keys, unexpected_keys, mismatch_keys = load_state_dict(model if not is_dist else model.module, state_dict)
        if len(missing_keys) > 0:
            logging.warn("Missing keys: {}".format(missing_keys))
        if len(unexpected_keys) > 0:
            logging.warn("Unexpected keys: {}".format(unexpected_keys))
        if len(mismatch_keys) > 0:
            logging.warn("Mismatch keys: {}".format(mismatch_keys))

    # Resume model from a checkpoint
    if cfg.train.resume != '' or cfg.train.resume_last:
        model_path = cfg.train.resume if cfg.train.resume != '' else cfg.output_dir
        if os.path.exists(model_path) and os.path.isdir(model_path):
            iter, best_score = load_resume_model(model if not is_dist else model.module, optimizer, lr_scheduler, cfg.train.resume, device)
            epoch =  iter // len(train_loader)
            logging.info("Resuming from epoch {}, iter {}, best score {}".format(epoch, iter, best_score))
        else:
            raise ValueError("Cannot resume model from {}".format(model_path))

    loss_buffers = {}
    batch_time_buffer = HistoryBuffer(max_length=50)
    data_time_buffer = HistoryBuffer(max_length=50)
    grad_norm_buffer = HistoryBuffer(max_length=50)
    grad_clip_enabled = cfg.train.gradient_clipping.enabled
    max_grad_norm = cfg.train.gradient_clipping.max_norm
    if grad_clip_enabled and max_grad_norm <= 0:
        raise ValueError(
            'train.gradient_clipping.max_norm must be greater than 0')
    if grad_clip_enabled:
        logger.info(
            f'Gradient clipping enabled: max_norm={max_grad_norm}, norm_type=2')
    else:
        logger.info('Gradient clipping disabled; monitoring grad_norm only')

    # Build validation metrics
    val_error_dict = build_metric(cfg.train.val_metrics)
    assert len(val_error_dict) > 0, "No validation metrics found!"
    assert cfg.train.val_best_metric in val_error_dict, "Best validation metric not found!"
    best_metric = val_error_dict[cfg.train.val_best_metric]
    if best_score is None:
        best_score = (-float('inf') if best_metric.higher_is_better
                      else float('inf'))

    # Start training
    logging.info("Start training...")
    model.train()
    logging.info("Iter: {}, len dataloader: {}".format(iter, len(train_loader)))
    epoch =  iter // len(train_loader)
    scaler = GradScaler() if precision == 16 else None

    eval_fn = eval_video if cfg.dataset.test.name == 'VIM' else eval_image
    
    end_time = time.time()
    while iter < cfg.train.max_iter:
        
        for _, batch in enumerate(train_loader):
            
            if is_dist:
                train_sampler.set_epoch(epoch)

            data_time_val = time.time() - end_time
            data_time_buffer.update(data_time_val)

            iter += 1
            if iter > cfg.train.max_iter:
                break

            batch = adapt_training_batch(batch)
            batch = {k: v.to(device) for k, v in batch.items()}
            batch['iter'] = iter
            optimizer.zero_grad()
            if precision == 16:
                with autocast():
                    output, loss = model(batch, mem_feat=None)
            else:
                output, loss = model(batch, mem_feat=None)
            if loss is None:
                logging.error("Loss is None!")
                continue

            if precision == 16:
                scaler.scale(loss['total']).backward()
                # Clip the real gradients, not the AMP-scaled gradients.
                scaler.unscale_(optimizer)
            else:
                loss['total'].backward()

            all_params = list(itertools.chain(
                *[group["params"] for group in optimizer.param_groups]))
            if grad_clip_enabled:
                total_grad_norm = torch.nn.utils.clip_grad_norm_(
                    all_params, max_grad_norm)
            else:
                grad_norms = [
                    torch.linalg.vector_norm(param.grad.detach(), ord=2)
                    for param in all_params if param.grad is not None
                ]
                if grad_norms:
                    total_grad_norm = torch.linalg.vector_norm(
                        torch.stack(grad_norms), ord=2)
                else:
                    total_grad_norm = torch.zeros((), device=device)
            grad_norm = total_grad_norm.item()
            grad_norm_buffer.update(grad_norm)

            # Store to log_metrics
            loss_reduced = loss
            for k, v in loss_reduced.items():
                if k not in loss_buffers:
                    loss_buffers[k] = HistoryBuffer(max_length=50)
                loss_buffers[k].update(v.item())
            
            batch_time_val = time.time() - end_time
            batch_time_buffer.update(batch_time_val)
            
            # Logging
            if iter % cfg.train.log_iter == 0:
                log_str = "Epoch: {}, Iter: {}/{}".format(epoch, iter, cfg.train.max_iter)
                for k, v in loss_buffers.items():
                    log_str += ", {}: {:.4f}".format(k, v.mean())
                log_str += ", grad_norm: {:.4f}".format(grad_norm_buffer.mean())
                log_str += ", lr: {:.6f}".format(lr_scheduler.get_last_lr()[0])
                log_str += ", batch_time: {:.4f}s".format(batch_time_buffer.mean())
                log_str += ", data_time: {:.4f}s".format(data_time_buffer.mean())

                # logging.info(log_str)
                logger.info(log_str)
                
                if scalar_writer is not None:
                    for k, buf in loss_buffers.items():
                        scalar_writer.add_scalar(f'train/{k}', buf.mean(), iter)
                    scalar_writer.add_scalar('train/grad_norm', grad_norm_buffer.mean(), iter)
                    scalar_writer.add_scalar('train/lr', lr_scheduler.get_last_lr()[0], iter)
                    scalar_writer.add_scalar('train/batch_time', batch_time_buffer.mean(), iter)
                    scalar_writer.add_scalar('train/epoch', epoch, iter)

            if global_rank == 0 and cfg.wandb.use and iter % cfg.train.log_iter == 0:
                for k, buf in loss_buffers.items():
                    wandb.log({"train/" + k: buf.mean()}, commit=False)
                wandb.log({"train/grad_norm": grad_norm_buffer.mean()},commit=False)
                wandb.log({"train/lr": lr_scheduler.get_last_lr()[0]}, commit=False)
                wandb.log({"train/batch_time": batch_time_buffer.mean()}, commit=False)
                wandb.log({"train/data_time": data_time_buffer.mean()}, commit=False)
                wandb.log({"train/epoch": epoch}, commit=False)
                wandb.log({"train/iter": iter}, commit=True)

            # Update
            if precision == 16:
                scaler.step(optimizer)
                scaler.update()
            else:
                optimizer.step()

            lr_scheduler.step()
            
            # Visualization
            if global_rank == 0 and iter % cfg.train.vis_iter == 0:
                if cfg.wandb.use:
                    try:
                        wandb_log_image(batch, output, iter)
                    except Exception as e:
                        logger.warning(f"wandb log image failed: {e}")

                if image_writer is not None:
                    try:
                        tensorboard_log_image(batch, output, iter, image_writer, n_samples=5)
                    except Exception as e:
                        logger.warning(f"TensorBoard log image failed: {e}")
                
            # Evaluation
            if iter % cfg.train.val_iter == 0 and (cfg.train.val_dist or (not cfg.train.val_dist and global_rank == 0)):
                logging.info("Start validation...")
                model.eval()
                val_model = model.module if is_dist else model
                _ = [v.reset() for v in val_error_dict.values()]
                _ = eval_fn(val_model, val_loader, device, cfg.test.log_iter, val_error_dict, do_postprocessing=False, callback=None)

                if is_dist and cfg.train.val_dist:
                    logging.info("Gathering metrics...")
                    # Gather all metrics
                    for k, v in val_error_dict.items():
                        v.gather_metric(0)

                if global_rank == 0:
                    log_str = "Validation:"
                    for k, v in val_error_dict.items():
                        log_str += "{}: {:.4f}, ".format(k, v.average())
                    logging.info(log_str)
                    
                    # Save best model
                    total_error = val_error_dict[cfg.train.val_best_metric].average()
                    is_better = (
                        total_error > best_score
                        if best_metric.higher_is_better
                        else total_error < best_score)
                    if is_better:
                        logging.info("Best score changed from {:.4f} to {:.4f}".format(best_score, total_error))
                        best_score = total_error
                        logging.info("Saving best model...")
                        save_path = os.path.join(cfg.output_dir, 'best_model.pth')
                        with open(os.path.join(cfg.output_dir, "best_metrics.txt"), 'w') as f:
                            f.write("iter: {}\n".format(iter))
                            for k, v in val_error_dict.items():
                                f.write("{}: {:.4f}\n".format(k, v.average()))
                        torch.save(val_model.state_dict(), save_path)
                    
                    if cfg.wandb.use:
                        for k, v in val_error_dict.items():
                            wandb.log({"val/" + k: v.average()}, commit=False)
                        wandb.log({"val/epoch": epoch}, commit=False)
                        wandb.log({"val/best_error": best_score}, commit=False)
                        wandb.log({"val/iter": iter}, commit=True)
                    
                    if scalar_writer is not None:
                        for k, v in val_error_dict.items():
                            scalar_writer.add_scalar(f'val/{k}', v.average(), iter)
                        scalar_writer.add_scalar('val/best_error', best_score, iter)
                    
                    logging.info("Saving the last model...")
                    save_dict = {
                        'optimizer': optimizer.state_dict(),
                        'lr_scheduler': lr_scheduler.state_dict(),
                        'iter': iter,
                        'best_score': best_score
                    }
                    save_path = os.path.join(cfg.output_dir, 'last_opt.pth')
                    torch.save(save_dict, save_path)
                    save_path = os.path.join(cfg.output_dir, 'last_model.pth')
                    torch.save(val_model.state_dict(), save_path)

                    model_path = os.path.join(cfg.output_dir, f'model_iter{iter}.pth')
                    opt_path = os.path.join(cfg.output_dir, f'model_opt_iter{iter}.pth')
                    torch.save(val_model.state_dict(), model_path)
                    torch.save(save_dict, opt_path)

                    model_files = glob.glob(os.path.join(cfg.output_dir, 'model_iter*.pth'))
                    iter_nums = []
                    for f in model_files:
                        try:
                            num = int(f.split('_iter')[-1].split('.pth')[0])
                            iter_nums.append((num, f))
                        except:
                            continue
                    iter_nums.sort(key=lambda x: x[0])

                    if len(iter_nums) > 3:
                        for _, old_model_path in iter_nums[:-3]:
                            old_opt_path = old_model_path.replace('model_iter', 'model_opt_iter')
                            if os.path.exists(old_model_path):
                                os.remove(old_model_path)
                            if os.path.exists(old_opt_path):
                                os.remove(old_opt_path)

                model.train()
            end_time = time.time()
        epoch += 1
        
    if scalar_writer is not None:
        scalar_writer.close()
    if image_writer is not None:
        image_writer.close()
        
    logger.info("Training finished.")
