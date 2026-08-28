from yacs.config import CfgNode as CN

CONFIG = CN()

# ------------------------ General ------------------------
CONFIG.output_dir = 'logs'
CONFIG.name = 'default'

# ------------------------ Training ------------------------
CONFIG.train = CN({})
CONFIG.train.seed = -1
CONFIG.train.batch_size = 2
CONFIG.train.num_workers = 16
CONFIG.train.resume = '' # Resume from a checkpoint
CONFIG.train.resume_last = False # Resume last model or not
CONFIG.train.max_iter = 100000
CONFIG.train.log_iter = 50
CONFIG.train.vis_iter = 500
CONFIG.train.val_iter = 2000
CONFIG.train.val_metrics = ['MAD', 'MSE', 'dtSSD']
CONFIG.train.val_best_metric = 'MAD' # Metric to save the best model
CONFIG.train.val_dist = True # Evaluate distributed

gradient_clipping = CN({})
gradient_clipping.enabled = True
gradient_clipping.max_norm = 1.0 # Global L2 gradient norm
CONFIG.train.gradient_clipping = gradient_clipping

optimizer = CN({})
optimizer.name = 'sgd' # sgd
optimizer.lr = 1.0e-4
optimizer.momentum = 0.9
optimizer.weight_decay = 1.0e-2
optimizer.betas = (0.9, 0.999)
CONFIG.train.optimizer = optimizer

scheduler = CN({})
scheduler.name = 'poly' # step, cosine
scheduler.power = 0.9 # for poly
scheduler.step_size = 10000 # for step
scheduler.gamma = 0.1 # for step or warmup
scheduler.warmup_iters = 1000
scheduler.warmup_factor = 0.001
scheduler.delay_iters = 10000
scheduler.decay_steps = [20000, 30000, 40000]
scheduler.peak_scale = 1.0
scheduler.eta_min = 0.00001
CONFIG.train.scheduler = scheduler

CONFIG.wandb = CN({})
CONFIG.wandb.project = 'maggie'
CONFIG.wandb.entity = 'vidal'
CONFIG.wandb.use = False
CONFIG.wandb.id = ''

CONFIG.tensorboard = CN({})
CONFIG.tensorboard.use = True
CONFIG.tensorboard.log_dir = 'tensorboard'

# ------------------------ Testing ------------------------
CONFIG.test = CN({})
CONFIG.test.batch_size = 1 # Only support 1 for now
CONFIG.test.num_workers = 4
CONFIG.test.save_results = True
CONFIG.test.save_dir = 'logs'
CONFIG.test.postprocessing = True
CONFIG.test.metrics = ['MAD', 'MSE', 'SAD', 'Conn', 'Grad', 'dtSSD', 'MESSDdt']
CONFIG.test.log_iter = 50

# ------------------------ Model ------------------------
CONFIG.model = CN({})
CONFIG.model.weights = ''
# Load only model.encoder from model.weights during training. Decoder and auxiliary heads retain their initialization. This does not affect resume.
CONFIG.model.load_encoder_only = False
CONFIG.model.arch = 'MaGGIe'
CONFIG.model.sync_bn = True
CONFIG.model.having_unused_params = False
CONFIG.model.warmup_iters = 5000

# Encoder
CONFIG.model.encoder = 'res_encoder_29' # resnet34
CONFIG.model.encoder_args = CN({}, new_allowed=True)
# CONFIG.model.encoder_args.pretrained = True
# CONFIG.model.encoder_args.num_mask = 1

# ASPP
CONFIG.model.aspp = CN({})
CONFIG.model.aspp.in_channels = 512
CONFIG.model.aspp.out_channels = 512

# Decoder
CONFIG.model.decoder = ''
CONFIG.model.decoder_args = CN({}, new_allowed=True)

# For loss
CONFIG.model.loss_alpha_w = 1.0
CONFIG.model.loss_alpha_type = 'l1'
CONFIG.model.loss_alpha_grad_w = 1.0
CONFIG.model.loss_alpha_lap_w = 1.0
CONFIG.model.loss_alpha_gradun_w = 1.0
CONFIG.model.loss_alpha_lapun_w = 1.0
CONFIG.model.loss_alpha_ddc_w = 1.0
CONFIG.model.loss_atten_w = 1.0
CONFIG.model.loss_reweight_os8 = True
CONFIG.model.loss_dtSSD_w = 1.0

CONFIG.model.lambdas_pix_last = CN()
CONFIG.model.lambdas_pix_last.bce = 30
CONFIG.model.lambdas_pix_last.mae = 100
CONFIG.model.lambdas_pix_last.ssim = 10
CONFIG.model.pix_loss_weight = 1.0
CONFIG.model.coarse_semantic_loss_weight = 10.0
CONFIG.model.gdt_loss_weight = 1.0
CONFIG.model.use_ddc_loss = False

# Training-only person semantic auxiliary head
CONFIG.model.semantic_aux = CN({})
CONFIG.model.semantic_aux.enabled = False
CONFIG.model.semantic_aux.hidden_channels = 64
CONFIG.model.semantic_aux.target_threshold = 0.05
CONFIG.model.semantic_aux.loss_weight = 0.5
CONFIG.model.semantic_aux.bce_weight = 1.0
CONFIG.model.semantic_aux.dice_weight = 1.0
CONFIG.model.semantic_aux.hybrid_e_weight = 0.0
CONFIG.model.semantic_aux.hybrid_e_kernel_size = 7
CONFIG.model.semantic_aux.hybrid_e_boundary_factor = 5.0
CONFIG.model.semantic_aux.hard_negative_weight = 1.0
CONFIG.model.semantic_aux.hard_negative_ratio = 0.2

# For SHM
CONFIG.model.shm = CN({})
CONFIG.model.shm.lr_scale = 0.5
CONFIG.model.shm.dilation_kernel = 15
CONFIG.model.shm.max_n_pixel = 4000000
CONFIG.model.shm.mgm_weights = ''

# ------------------------ Dataset ------------------------
dataset = CN({})

dataset.train = CN({})
dataset.train.name = 'VIM'
dataset.train.root_dir = ''
dataset.train.root_dirs = []
# Per-root epoch sampling fractions aligned with root_dirs. For example, 1.0
# uses the full root and 0.3 uses about 30 percent. Empty uses every sample.
dataset.train.root_sampling_rates = []
dataset.train.split = 'train'

dataset.train.short_size = 768

# For augmentation
dataset.train.random_state = 2023
dataset.train.crop = [512, 512] # (h, w)
dataset.train.max_inst = 10
dataset.train.padding_crop_p = 0.1
dataset.train.flip_p = 0.5
dataset.train.gamma_p = 0.3

dataset.train.add_noise_p = 0.3
dataset.train.jpeg_p = 0.1
dataset.train.affine_p = 0.1
dataset.train.binarized_kernel = 30
dataset.train.downscale_mask_p = 0.5
dataset.train.mask_dir_name = "masks_matched"
dataset.train.mask_threshold = 0
dataset.train.alpha_dir_name = 'pha'

# For mixed matting and binary-person supervision
dataset.train.binary_root_dir = ''
dataset.train.binary_root_dirs = []
dataset.train.binary_root_sampling_rates = []
dataset.train.binary_mask_dir_name = 'masks'
dataset.train.binary_mask_threshold = 0
dataset.train.binary_ratio = 0.3
dataset.train.mixed_epoch_size = 0

# For video augmentation
dataset.train.clip_length = 8
dataset.train.max_step_size = 2
dataset.train.motion_p = 0.3

dataset.test = CN({})
dataset.test.name = 'VIM'
dataset.test.root_dir = ''
dataset.test.root_dirs = []
dataset.test.split = 'valid'
dataset.test.short_size = 768
dataset.test.downscale_mask = True
dataset.test.alpha_dir_name = "alphas"
dataset.test.mask_dir_name = "masks_matched"
dataset.test.mask_threshold = 0

# For video size
dataset.test.clip_length = 8
dataset.test.clip_overlap = 2

CONFIG.dataset = dataset
