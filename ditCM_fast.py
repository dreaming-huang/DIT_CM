# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

"""
A minimal training script for DiT.
"""
import random
import torch
# the first flag below was False when we tested this script but True makes A100 training a lot faster:
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import Dataset, DataLoader
from torch.utils.data.distributed import DistributedSampler
from torchvision.datasets import ImageFolder
from torchvision import transforms
import numpy as np
from collections import OrderedDict
from PIL import Image
from copy import deepcopy
from glob import glob
from time import time
import argparse
import logging
import os
from torchvision.utils import save_image

os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'
from accelerate import Accelerator
from accelerate.utils import set_seed

from models import DiT_models
from diffusion import create_diffusion
from diffusers.models import AutoencoderKL
from download import find_model


#################################################################################
#                             Training Helper Functions                         #
#################################################################################

@torch.no_grad()
def update_ema(ema_model, model, decay=0.99):
    """
    Step the EMA model towards the current model.
    """
    ema_params = OrderedDict(ema_model.named_parameters())
    model_params = OrderedDict(model.named_parameters())

    for name, param in model_params.items():
        name = name.replace("module.", "")
        # TODO: Consider applying only to params that require_grad to avoid small numerical changes of pos_embed
        ema_params[name].mul_(decay).add_(param.data, alpha=1 - decay)


def requires_grad(model, flag=True):
    """
    Set requires_grad flag for all parameters in a model.
    """
    for p in model.parameters():
        p.requires_grad = flag


def create_logger(logging_dir):
    """
    Create a logger that writes to a log file and stdout.
    """
    logging.basicConfig(
        level=logging.INFO,
        format='[\033[34m%(asctime)s\033[0m] %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S',
        handlers=[logging.StreamHandler(), logging.FileHandler(f"{logging_dir}/log.txt")]
    )
    logger = logging.getLogger(__name__)
    return logger


def center_crop_arr(pil_image, image_size):
    """
    Center cropping implementation from ADM.
    https://github.com/openai/guided-diffusion/blob/8fb3ad9197f16bbc40620447b2742e13458d2831/guided_diffusion/image_datasets.py#L126
    """
    while min(*pil_image.size) >= 2 * image_size:
        pil_image = pil_image.resize(
            tuple(x // 2 for x in pil_image.size), resample=Image.BOX
        )

    scale = image_size / min(*pil_image.size)
    pil_image = pil_image.resize(
        tuple(round(x * scale) for x in pil_image.size), resample=Image.BICUBIC
    )

    arr = np.array(pil_image)
    crop_y = (arr.shape[0] - image_size) // 2
    crop_x = (arr.shape[1] - image_size) // 2
    return Image.fromarray(arr[crop_y: crop_y + image_size, crop_x: crop_x + image_size])


class CustomDataset(Dataset):
    def __init__(self, features_dir, labels_dir):
        self.features_dir = features_dir
        self.labels_dir = labels_dir

        self.features_files = sorted(os.listdir(features_dir))
        self.labels_files = sorted(os.listdir(labels_dir))

    def __len__(self):
        assert len(self.features_files) == len(self.labels_files), \
            "Number of feature files and label files should be same"
        return len(self.features_files)

    def __getitem__(self, idx):
        feature_file = self.features_files[idx]
        label_file = self.labels_files[idx]

        features = np.load(os.path.join(self.features_dir, feature_file))
        labels = np.load(os.path.join(self.labels_dir, label_file))
        return torch.from_numpy(features), torch.from_numpy(labels)

# def set_seed(seed: int, device_specific: bool = False):
#     """
#     Helper function for reproducible behavior to set the seed in `random`, `numpy`, `torch`.

#     Args:
#         seed (`int`):
#             The seed to set.
#         device_specific (`bool`, *optional*, defaults to `False`):
#             Whether to differ the seed on each device slightly with `self.process_index`.
#     """
#     if device_specific:
#         seed += AcceleratorState().process_index
#     random.seed(seed)
#     np.random.seed(seed)
#     torch.manual_seed(seed)
#     if is_xpu_available():
#         torch.xpu.manual_seed_all(seed)
#     elif is_npu_available():
#         torch.npu.manual_seed_all(seed)
#     else:
#         torch.cuda.manual_seed_all(seed)
#     # ^^ safe to call this function even if cuda is not available
#     if is_tpu_available():
#         xm.set_rng_state(seed)

#################################################################################
#                                  Training Loop                                #
#################################################################################

def main(args):
    """
    Trains a new DiT model.
    """
    assert torch.cuda.is_available(), "Training currently requires at least one GPU."

    # Setup accelerator:
    set_seed(args.global_seed)
    torch.cuda.manual_seed_all(args.global_seed)
    
    accelerator = Accelerator()
    device = accelerator.device

    # Setup an experiment folder:
    if accelerator.is_main_process:
        os.makedirs(args.results_dir, exist_ok=True)  # Make results folder (holds all experiment subfolders)
        experiment_index = len(glob(f"{args.results_dir}/*"))
        model_string_name = args.model.replace("/", "-")  # e.g., DiT-XL/2 --> DiT-XL-2 (for naming folders)
        experiment_dir = f"{args.results_dir}/{experiment_index:03d}-{model_string_name}"  # Create an experiment folder
        checkpoint_dir = f"{experiment_dir}/checkpoints"  # Stores saved model checkpoints
        os.makedirs(checkpoint_dir, exist_ok=True)
        logger = create_logger(experiment_dir)
        logger.info(f"Experiment directory created at {experiment_dir}")

    # Create model:
    assert args.image_size % 8 == 0, "Image size must be divisible by 8 (for the VAE encoder)."
    latent_size = args.image_size // 8
    # 创建老师: 
    model_teacher = DiT_models["DiT-XL/2"](
        input_size=latent_size,
        num_classes=args.num_classes
    ).to(device)

    ckpt_path = args.ckpt or f"DiT-XL-2-{args.image_size}x{args.image_size}.pt"
    state_dict = find_model(ckpt_path)
    model_teacher.load_state_dict(state_dict)
    model_teacher.eval()  # important!
    requires_grad(model_teacher, False)

    model = DiT_models[args.model](
        input_size=latent_size,
        num_classes=args.num_classes
    )
    if args.pretrain: 
        ckpt_path = args.pretrain_ckpt or f"DiT-XL-2-{args.image_size}x{args.image_size}.pt"
        state_dict = find_model(ckpt_path)
        model.load_state_dict(state_dict)
    # Note that parameter initialization is done within the DiT constructor
    model = model.to(device)
    ema = deepcopy(model).to(device)  # Create an EMA of the model for use after training
    requires_grad(ema, False)
    diffusion = create_diffusion(timestep_respacing=str(args.num_sampling_steps))  # default: 1000 steps, linear noise schedule
    vae = AutoencoderKL.from_pretrained(f"stabilityai/sd-vae-ft-{args.vae}").to(device)
    if accelerator.is_main_process:
        logger.info(f"DiT Parameters: {sum(p.numel() for p in model.parameters()):,}")

    # Setup optimizer (we used default Adam betas=(0.9, 0.999) and a constant learning rate of 1e-4 in our paper):
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0)

    # Setup data:
    features_dir = f"{args.feature_path}/imagenet256_features"
    labels_dir = f"{args.feature_path}/imagenet256_labels"
    dataset = CustomDataset(features_dir, labels_dir)
    loader = DataLoader(
        dataset,
        batch_size=int(args.global_batch_size // accelerator.num_processes),
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=True,
        drop_last=True
    )
    if accelerator.is_main_process:
        logger.info(f"Dataset contains {len(dataset):,} images ({args.feature_path})")

    # Prepare models for training:
    update_ema(ema, model, decay=0)  # Ensure EMA is initialized with synced weights
    model.train()  # important! This enables embedding dropout for classifier-free guidance
    ema.train()  # EMA model should always be in eval mode
    model, opt, loader = accelerator.prepare(model, opt, loader)

    # Variables for monitoring/logging purposes:
    train_steps = 0
    log_steps = 0
    running_loss = 0
    start_time = time()


    valid_z=None
    
    if accelerator.is_main_process:
        logger.info(f"Training for {args.epochs} epochs...")
    for epoch in range(args.epochs):
        if accelerator.is_main_process:
            logger.info(f"Beginning epoch {epoch}...")
        for x, y in loader:
            if train_steps % args.log_every == 0:
                if accelerator.is_main_process:
                    with torch.no_grad():
                        #展示图片验证
                        class_labels = [207, 360, 387, 974, 88, 979, 417, 279]
                        latent_size = args.image_size // 8
                        # Create sampling noise:
                        n = len(class_labels)
                        valid_z = torch.randn(n, 4, latent_size, latent_size, device=device) if valid_z is None else valid_z
                        z=valid_z
                        y_valid = torch.tensor(class_labels, device=device)

                        # Setup classifier-free guidance:
                        z = torch.cat([z, z], 0)
                        y_null = torch.tensor([1000] * n, device=device)
                        y_valid = torch.cat([y_valid, y_null], 0)

                        # model_kwargs = dict(y=y)
                        model_kwargs = dict(y=y_valid, cfg_scale=4.0)

                        # t_list=[0,12,24,36,49][::-1]
                        # t_list=[0,49][::-1]
                        # t_list=list(range(1,50))[::-1]
                        # print(t_list)

                        t_list=np.linspace(0, args.num_sampling_steps-1, 5).round().tolist()[1:][::-1]

                        sample_valid=diffusion.cm_sample(ema.forward_with_cfg,torch.tensor(t_list,device=device,dtype=int),z.shape,z,device=device,model_kwargs=model_kwargs)
                        sample_valid, sample_valid_nocfg_last = sample_valid.chunk(2, dim=0)  # Remove null class samples
                        # print(sample_valid)
                        sample_valid = vae.decode(sample_valid / 0.18215).sample
                        sample_valid_nocfg_last = vae.decode(sample_valid_nocfg_last / 0.18215).sample
                        sample_valid_nocfg_all,_=diffusion.cm_sample(ema,torch.tensor(t_list,device=device,dtype=int),z.shape,z,device=device,model_kwargs=model_kwargs).chunk(2, dim=0)
                        sample_valid_nocfg_all = vae.decode(sample_valid_nocfg_all / 0.18215).sample
                        if train_steps==0:
                            sample_teacher=diffusion.ddim_sample_loop(
                                model_teacher.forward_with_cfg, z.shape, z, clip_denoised=False, model_kwargs=model_kwargs, progress=True, device=device
                            )
                            sample_teacher, _ = sample_teacher.chunk(2, dim=0)  # Remove null class samples
                            sample_teacher = vae.decode(sample_teacher / 0.18215).sample
                            save_image(sample_teacher, f"{experiment_dir}/{train_steps:07d}sample_teacher.png", nrow=4, normalize=True, value_range=(-1, 1))

                        save_image(sample_valid, f"{experiment_dir}/{train_steps:07d}sample_valid.png", nrow=4, normalize=True, value_range=(-1, 1))
                        save_image(sample_valid_nocfg_last, f"{experiment_dir}/{train_steps:07d}sample_valid_nocfg_last.png", nrow=4, normalize=True, value_range=(-1, 1))
                        save_image(sample_valid_nocfg_all, f"{experiment_dir}/{train_steps:07d}sample_valid_nocfg_all.png", nrow=4, normalize=True, value_range=(-1, 1))



            x = x.to(device)
            y = y.to(device)
            x = x.squeeze(dim=1)
            y = y.squeeze(dim=1)
            t = torch.randint(0, diffusion.num_timesteps, (x.shape[0],), device=device)
            model_kwargs = dict(y=y)
            loss_dict=diffusion.cm_training_losses_mutilStep(model,ema,model_teacher.forward_with_cfg,x,t,model_kwargs)
            # print('a')
            # loss_dict=diffusion.cm_training_losses(model,ema,model_teacher.forward_with_cfg,x,t,model_kwargs)

            # print("loss1: ",diffusion.cm_training_losses_mutilStep(model,ema,model_teacher.forward_with_cfg,x,t,model_kwargs)["loss"].mean())
            # print("loss2: ",diffusion.cm_training_losses(model,ema,model_teacher.forward_with_cfg,x,t,model_kwargs)["loss"].mean())
            
            loss = loss_dict["loss"].mean()
            opt.zero_grad()
            accelerator.backward(loss)
            opt.step()

            if train_steps % args.log_every == 0:
                # Measure training speed:
                torch.cuda.synchronize()
                end_time = time()
                steps_per_sec = log_steps / (end_time - start_time) if train_steps else 0
                # Reduce loss history over all processes:
                avg_loss = torch.tensor(running_loss / log_steps, device=device) if train_steps else 0
                avg_loss = avg_loss.item() / accelerator.num_processes if train_steps else 0
                if accelerator.is_main_process:
                    logger.info(f"(step={train_steps:07d}) Train Loss: {avg_loss:.4f}, Train Steps/Sec: {steps_per_sec:.2f}")
                    # #展示图片验证
                    # class_labels = [207, 360, 387, 974, 88, 979, 417, 279]
                    # latent_size = args.image_size // 8
                    # # Create sampling noise:
                    # n = len(class_labels)
                    # valid_z = torch.randn(n, 4, latent_size, latent_size, device=device) if valid_z is None else valid_z
                    # z=valid_z
                    # y = torch.tensor(class_labels, device=device)

                    # # Setup classifier-free guidance:
                    # z = torch.cat([z, z], 0)
                    # y_null = torch.tensor([1000] * n, device=device)
                    # y = torch.cat([y, y_null], 0)

                    # # model_kwargs = dict(y=y)
                    # model_kwargs = dict(y=y, cfg_scale=4.0)

                    # # t_list=[0,12,24,36,49][::-1]
                    # # t_list=[0,49][::-1]
                    # # t_list=list(range(1,50))[::-1]
                    # # print(t_list)

                    # t_list=np.linspace(0, args.num_sampling_steps-1, 5).round().tolist()[1:][::-1]
                    # print(t_list)



                        

                        # ddim_samples = diffusion.p_sample_loop(
                        #     ema.forward_with_cfg, z.shape, z, clip_denoised=False, model_kwargs=model_kwargs, progress=True, device=device
                        # )
                        # ddim_samples, _ = ddim_samples.chunk(2, dim=0)  # Remove null class ddim_samples
                        # print(ddim_samples)
                        # ddim_samples = vae.decode(ddim_samples / 0.18215).sample

                        # save_image(ddim_samples, f"{experiment_dir}/ddim_sample{train_steps:07d}.png", nrow=4, normalize=True, value_range=(-1, 1))

                        # import pdb
                        # pdb.set_trace()
                # Reset monitoring variables:
                running_loss = 0
                log_steps = 0
                start_time = time()


            update_ema(ema, model)

            # Log loss values:
            running_loss += loss.item()
            log_steps += 1
            train_steps += 1

            # Save DiT checkpoint:
            if train_steps % args.ckpt_every == 0 and train_steps > 0:
                if accelerator.is_main_process:
                    checkpoint = {
                        "model": model.module.state_dict(),
                        "ema": ema.state_dict(),
                        "opt": opt.state_dict(),
                        "args": args
                    }
                    checkpoint_path = f"{checkpoint_dir}/{train_steps:07d}.pt"
                    torch.save(checkpoint, checkpoint_path)
                    logger.info(f"Saved checkpoint to {checkpoint_path}")

    model.eval()  # important! This disables randomized embedding dropout
    # do any sampling/FID calculation/etc. with ema (or model) in eval mode ...
    
    if accelerator.is_main_process:
        logger.info("Done!")


if __name__ == "__main__":
    # Default args here will train DiT-XL/2 with the hyperparameters we used in our paper (except training iters).
    parser = argparse.ArgumentParser()
    parser.add_argument("--feature-path", type=str, default="features")
    parser.add_argument("--results-dir", type=str, default="results")
    parser.add_argument("--model", type=str, choices=list(DiT_models.keys()), default="DiT-XL/2")
    parser.add_argument("--image-size", type=int, choices=[256, 512], default=256)
    parser.add_argument("--num-classes", type=int, default=1000)
    parser.add_argument("--epochs", type=int, default=1400)
    parser.add_argument("--global-batch-size", type=int, default=32)
    parser.add_argument("--global-seed", type=int, default=42)
    parser.add_argument("--vae", type=str, choices=["ema", "mse"], default="ema")  # Choice doesn't affect training
    parser.add_argument("--num-workers", type=int, default=16)
    parser.add_argument("--log-every", type=int, default=100)
    parser.add_argument("--ckpt-every", type=int, default=10_000)
    parser.add_argument("--num-sampling-steps", type=int, default=1000)
    parser.add_argument("--ckpt", type=str, default=None,
        help="Optional path to a DiT checkpoint (default: auto-download a pre-trained DiT-XL/2 model).")
    parser.add_argument("--pretrain-ckpt", type=str, default=None,
        help="学生加载的预训练")
    parser.add_argument("--pretrain",action='store_true')
    parser.add_argument("--lr", type=float, default=1e-4)
    
    args = parser.parse_args()
    main(args)
