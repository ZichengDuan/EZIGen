import sys
import os
import time
import math
import random
import shutil
import subprocess
import logging
import warnings
from pathlib import Path
from copy import deepcopy
import copy
from argparse import ArgumentParser
from glob import glob
from diffusers.pipelines.stable_diffusion.pipeline_stable_diffusion import retrieve_timesteps
import torch
from torch.fx.interpreter import config
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.checkpoint
from torch.utils.data import ConcatDataset, DataLoader
import numpy as np
import cv2
from PIL import Image
import datasets
import torchvision.transforms as T
import torchvision.transforms as transforms
from packaging import version
from tqdm.auto import tqdm
from matplotlib import pyplot as plt
from omegaconf import OmegaConf
import argparse
import yaml
import shutil
import random
from PIL import Image
import os
import gc
import clip

from transformers import CLIPTextModel, CLIPTokenizer, CLIPProcessor, CLIPModel, AutoImageProcessor, AutoModel, AutoTokenizer, CLIPTextModelWithProjection, T5EncoderModel, T5TokenizerFast
from transformers.utils import logging as transformers_logging
from transformers.utils import ContextManagers
from transformers.utils import logging as hf_logging

import diffusers
from diffusers import (
    AutoencoderKL, DDPMScheduler, DDIMScheduler, StableDiffusionPipeline, UNet2DConditionModel, PNDMScheduler,
    DPMSolverSDEScheduler, DPMSolverMultistepInverseScheduler, DPMSolverMultistepScheduler, EulerDiscreteScheduler, HeunDiscreteScheduler,DDIMInverseScheduler
)
from models import FluxTransformer2DModel, FlowMatchEulerDiscreteScheduler

from diffusers.utils import is_xformers_available, check_min_version, deprecate, is_wandb_available, make_image_grid, convert_state_dict_to_diffusers, check_min_version
from diffusers.training_utils import EMAModel, compute_snr
from diffusers.optimization import get_scheduler
from diffusers.utils.testing_utils import enable_full_determinism

from accelerate.utils import ProjectConfiguration, set_seed
from accelerate.state import AcceleratorState
from accelerate.logging import get_logger
from accelerate import Accelerator

from utils import add_noise_to_image_flux, calculate_dino_similarity, compute_clip_similarity, resize_image_to_fit_short, create_soft_mask, get_sigmas, encode_prompt, prepare_latents, pack_latents, unpack_latents, compute_text_embeddings, calculate_shift
from models.pipelines import FluxPipeline_main

transformers_logging.set_verbosity_error()
warnings.filterwarnings("ignore", category=FutureWarning, module="diffusers")
warnings.filterwarnings("ignore", category=FutureWarning, module="transformers")

# Will error if the minimal version of diffusers is not installed. Remove at your own risks.
check_min_version("0.27.0.dev0")

logger = get_logger(__name__, log_level="INFO")

device = "cpu" if not torch.cuda.is_available() else "cuda"


def copy_matched_parameters(model_src, model_dst):
    src_state_dict = model_src.state_dict()
    dst_state_dict = model_dst.state_dict()

    for name, param in src_state_dict.items():
        if name in dst_state_dict and dst_state_dict[name].shape == param.shape:
            dst_state_dict[name].copy_(param)


def load_clip_model(device):
    model, preprocess = clip.load("ViT-B/32", device=device)
    return model, preprocess


def loop_infer(args, subject_img_paths, subject_features, vae, infer_noise_scheduler, noise_scheduler_copy, weight_dtype, target_prompt, subject_prompts, train_transforms, generator, init_image, sim_threshold=0.98, pipeline=None, output_root=None, post_fix=None, clip_model=None, clip_processor=None, foreground_mask=None, initial_image_size=None, source_image_path=None, threshold_timestep=None):
    """
    Flux inference loop with iterative refinement
    """
    sim = 0
    cur_loop_num = 1
    split_ratio = args.split_ratio

    loop_image = init_image
    prev_image = init_image
    
    max_num_loop = 10
    min_num_loop = 4

    if args.num_interations != -1:
        max_num_loop = args.num_interations + 1
        min_num_loop = args.num_interations + 1
        sim_threshold = 1

    while ((cur_loop_num < max_num_loop and sim < sim_threshold) or cur_loop_num < min_num_loop):
        
        # Add noise to the loop image
        noisy_latents = add_noise_to_image_flux(img=loop_image, vae=vae, noise_step=threshold_timestep, noise_scheduler=noise_scheduler_copy, train_transforms=train_transforms)
        noisy_latents = pack_latents(noisy_latents, 1, 16, int(args.resolution/8), int(args.resolution/8))
        
        args.skip_adapter_ratio = 0
        with torch.no_grad():
            res = pipeline(
                target_prompt, 
                num_inference_steps=args.infer_steps, 
                generator=generator,
                subject_f eatures=subject_features,
                weight_dtype=weight_dtype,
                args=args,
                latents=noisy_latents,
                height=args.resolution,
                width=args.resolution,
                guidance_scale=args.guidance_scale,
                threshold_timestep=threshold_timestep,
                foreground_mask=foreground_mask
            )
        loop_image = res["images"][0]
        origin_loop_image = loop_image.resize(initial_image_size)
        origin_loop_image.save(f"{output_root}/{target_prompt}{post_fix}/loop_{cur_loop_num}.png")

        sim = compute_clip_similarity(clip_model, clip_processor, image1=prev_image, image2=loop_image, device=device)

        prev_image = loop_image
        print(f"[Validation {target_prompt}{post_fix}][Loop {cur_loop_num}] Overall Similarity: {sim}.")
        cur_loop_num += 1

    torch.cuda.empty_cache()
    return loop_image


def iteration_wrapper(args, accelerator, subject_img_paths, flux_transformer, text_encoder_one, text_encoder_two, tokenizer_one, tokenizer_two, vae, infer_noise_scheduler, noise_scheduler_copy, weight_dtype, target_prompt, subject_prompts, train_transforms, generator=None, output_root=None, post_fix="", pipeline=None, clip_model=None, clip_processor=None, source_image_path=None, foreground_mask_path=None):
    """
    Main iteration wrapper for Flux inference
    """
    # Extract subject features
    subject_image = Image.open(subject_img_paths[0]).convert('RGB')
    subject_image = train_transforms(subject_image)

    noise_step = args.subject_timestep
    subject_noise = torch.randn((1, 16, args.resolution // 8, args.resolution // 8), dtype=weight_dtype, device=accelerator.device)
    noisy_subject_latents = add_noise_to_image_flux(subject_image.unsqueeze(0), vae, noise_step, noise_scheduler_copy, noise=subject_noise)

    # Get subject text embeddings
    subject_prompt_embeds, subject_pooled_prompt_embeds, subject_text_ids = compute_text_embeddings(subject_prompts, [text_encoder_one, text_encoder_two], [tokenizer_one, tokenizer_two])

    latent_image_ids = prepare_latents(8, 1, args.resolution, args.resolution, weight_dtype, accelerator.device)
    noisy_subject_latents = pack_latents(noisy_subject_latents, 1, 16, int(args.resolution/8), int(args.resolution/8))

    guidance = torch.full([1], 3.5, device=accelerator.device, dtype=weight_dtype)
    guidance = guidance.expand(noisy_subject_latents.shape[0])

    # Extract subject image features from flux
    with torch.no_grad():
        _, subject_features = flux_transformer(
                hidden_states=noisy_subject_latents,
                timestep=torch.tensor([noise_step]).to(device=accelerator.device, dtype=weight_dtype) / 1000,
                guidance=guidance,
                pooled_projections=subject_pooled_prompt_embeds,
                encoder_hidden_states=subject_prompt_embeds,
                txt_ids=torch.zeros(subject_prompt_embeds.shape[1], 3).to(device=accelerator.device, dtype=weight_dtype),
                img_ids=latent_image_ids,
                return_dict=False,
                subject_features=[],
                retrieve_subject_features=True,
                noise_step=noise_step
        )

    # Generate simple image without iteration
    args.skip_adapter_ratio = 0
    generator.manual_seed(int(args.seed))
    args.initial_loop = False
    res = pipeline(
        target_prompt,
        num_inference_steps=args.infer_steps,
        generator=generator,
        subject_features=subject_features, 
        weight_dtype=weight_dtype,
        args=args,
        height=args.resolution,
        width=args.resolution,
        guidance_scale=args.guidance_scale,
        is_simple=True
    )
    simple_img = res["images"][0]
    simple_img.save(f"{output_root}/{target_prompt}{post_fix}/simple_img.png")

    args.initial_loop = False
    # Generate pure text image (without subject features)
    res_origin = pipeline(
        target_prompt, 
        num_inference_steps=args.infer_steps, 
        generator=generator.manual_seed(int(args.seed)), 
        subject_features=None, 
        weight_dtype=weight_dtype,
        args=args,
        guidance_scale=args.guidance_scale,
        height=args.resolution,
        width=args.resolution,
        is_pure_text=True
    ) 
    pure_text_image = res_origin["images"][0]
    pure_text_image.save(f"{output_root}/{target_prompt}{post_fix}/pure_text_image.png")
    
    # Calculate discrete timesteps for noise injection
    sigmas = np.linspace(1.0, 1 / args.infer_steps, args.infer_steps)
    image_seq_len = noisy_subject_latents[0].shape[1]
    mu = calculate_shift(
        image_seq_len,
        infer_noise_scheduler.config.get("base_image_seq_len", 256),
        infer_noise_scheduler.config.get("max_image_seq_len", 4096),
        infer_noise_scheduler.config.get("base_shift", 0.5),
        infer_noise_scheduler.config.get("max_shift", 1.15),
    )

    infer_discrete_timesteps, num_inf_steps = retrieve_timesteps(infer_noise_scheduler, args.infer_steps, accelerator.device, sigmas=sigmas, mu=mu)
    infer_discrete_timesteps = infer_discrete_timesteps.cpu().numpy().tolist()

    noise_step = args.split_ratio * infer_noise_scheduler.config.num_train_timesteps
    threshold_timestep = min(filter(lambda x: x <= noise_step, infer_discrete_timesteps), key=lambda x: abs(x - noise_step))
    
    # Generate initial image for iteration
    if args.init_img_path is None:
        args.initial_loop = True
        res = pipeline(
            target_prompt, 
            num_inference_steps=args.infer_steps, 
            generator=generator.manual_seed(int(args.seed)), 
            subject_features=subject_features, 
            weight_dtype=weight_dtype,
            args=args,
            guidance_scale=args.guidance_scale,
            threshold_timestep=threshold_timestep,
            height=args.resolution,
            width=args.resolution,
        )
        initial_image = res["images"][0]
    else:
        initial_image = Image.open(args.init_img_path).resize((args.resolution, args.resolution))
        
    initial_image_resized = initial_image
    W, H = initial_image.size
    initial_image.save(f"{output_root}/{target_prompt}{post_fix}/initial_loop.png")
    
    # Handle foreground mask if provided
    if args.do_editing and foreground_mask_path is not None:
        foreground_mask = Image.open(foreground_mask_path).convert('RGB')
        foreground_mask = resize_image_to_fit_short(foreground_mask, short_size=args.resolution)
        foreground_mask = create_soft_mask(np.array(foreground_mask), sigma=5)
        foreground_mask = foreground_mask[:, :, 0]
        plt.imsave(f"{output_root}/{target_prompt}{post_fix}/mask_foreground.png", foreground_mask * 255)
        foreground_mask = cv2.resize(foreground_mask, (args.resolution, args.resolution))
        foreground_mask = torch.tensor(foreground_mask).to(accelerator.device)
    else:
        foreground_mask = None
    
    # Start iteration loop
    args.initial_loop = False
    args.skip_adapter_ratio = 0
    final_image = loop_infer(args, subject_img_paths, subject_features, vae, infer_noise_scheduler, noise_scheduler_copy, weight_dtype, target_prompt, subject_prompts, train_transforms, generator, initial_image_resized, sim_threshold=args.sim_threshold, pipeline=pipeline, output_root=output_root, post_fix=post_fix, clip_model=clip_model, clip_processor=clip_processor, foreground_mask=foreground_mask, initial_image_size=(W, H), source_image_path=source_image_path, threshold_timestep=threshold_timestep)
    
    return final_image


def load_config_and_args():
    parser = argparse.ArgumentParser(description="Command line argument parser")

    parser.add_argument('--config', type=str, required=True, help="Path to the YAML configuration file")
    parser.add_argument('--target_prompt', type=str, required=True, help="Target prompt string")
    parser.add_argument('--subject_prompts', type=str, required=True, help="Subject prompt string")
    parser.add_argument('--subject_img_paths', type=str, required=True, help="Subject image path")
    parser.add_argument('--init_img_path', type=str, default=None, help="Initial image path")
    
    # Subject driven editing
    parser.add_argument('--do_editing', action='store_true')
    parser.add_argument('--foreground_mask_path', type=str, default=None)
    parser.add_argument('--source_image_path', type=str, default=None)
    
    # misc
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--split_ratio', type=float, default=0.5)
    parser.add_argument('--infer_steps', type=int, default=28)
    parser.add_argument('--guidance_scale', type=float, default=3.5)
    parser.add_argument('--sim_threshold', type=float, default=0.99)

    # output
    parser.add_argument('--output_root', type=str, default="outputs/")
    parser.add_argument('--num_interations', type=int, default=-1)

    args = parser.parse_args()

    with open(args.config, 'r') as f:
        config = yaml.safe_load(f)

    for key, value in config.items():
        setattr(args, key, value)

    return args


def init_acclerator(args):
    accelerator_project_config = ProjectConfiguration(project_dir=args.output_dir)

    accelerator = Accelerator(
        mixed_precision=args.mixed_precision,
        project_config=accelerator_project_config,
    )
    return accelerator


def load_models(args, weight_type):
    # Load Flux models
    noise_scheduler = FlowMatchEulerDiscreteScheduler.from_pretrained(args.pretrained_model_name_or_path, subfolder="scheduler", local_files_only=True, torch_dtype=weight_type)
    noise_scheduler_copy = copy.deepcopy(noise_scheduler)
    
    # Load the tokenizers
    tokenizer_one = CLIPTokenizer.from_pretrained(args.pretrained_model_name_or_path, subfolder="tokenizer", revision=args.revision, use_fast=False, local_files_only=True, torch_dtype=weight_type)
    tokenizer_two = T5TokenizerFast.from_pretrained(args.pretrained_model_name_or_path, subfolder="tokenizer_2", revision=args.revision, use_fast=False, local_files_only=True, torch_dtype=weight_type)
    
    text_encoder_one = CLIPTextModel.from_pretrained(args.pretrained_model_name_or_path, subfolder="text_encoder", revision=args.revision, variant=args.variant, local_files_only=True, torch_dtype=weight_type)
    text_encoder_two = T5EncoderModel.from_pretrained(args.pretrained_model_name_or_path, subfolder="text_encoder_2", revision=args.revision, variant=args.variant, local_files_only=True, torch_dtype=weight_type)
    
    vae = AutoencoderKL.from_pretrained(args.pretrained_model_name_or_path, subfolder="vae", revision=args.revision, variant=args.variant, local_files_only=True, torch_dtype=weight_type)
    
    flux_transformer = FluxTransformer2DModel.from_pretrained(args.pretrained_model_name_or_path, subfolder="transformer", torch_dtype=weight_type, local_files_only=True)
    
    clip_model, clip_processor = clip.load(args.clip_path, device=device)
    
    # freeze everything first
    vae.requires_grad_(False)
    text_encoder_one.requires_grad_(False)
    text_encoder_two.requires_grad_(False)
    flux_transformer.requires_grad_(False)

    return flux_transformer, noise_scheduler, noise_scheduler_copy, tokenizer_one, tokenizer_two, text_encoder_one, text_encoder_two, vae, clip_model, clip_processor


def register_flux_adapters(flux_transformer, args):
    """
    Register adapters to Flux transformer blocks based on training setup
    """
    # Register subject-related modules as in training
    for i, block in enumerate(flux_transformer.transformer_blocks):
        block.sub_norm = copy.deepcopy(block.norm1)
        block.add_module("sub_norm", block.sub_norm)
        
        attn = block.attn
        attn.sub_to_k = copy.deepcopy(attn.to_k)
        attn.add_module("sub_to_k", attn.sub_to_k)

        attn.sub_to_v = copy.deepcopy(attn.to_v)
        attn.add_module("sub_to_v", attn.sub_to_v)
        
    for i, block in enumerate(flux_transformer.single_transformer_blocks):
        block.sub_norm = copy.deepcopy(block.norm)
        block.add_module("sub_norm", block.sub_norm)
        
        attn = block.attn
        attn.sub_to_k = copy.deepcopy(attn.to_k)
        attn.add_module("sub_to_k", attn.sub_to_k)

        attn.sub_to_v = copy.deepcopy(attn.to_v)
        attn.add_module("sub_to_v", attn.sub_to_v)


def load_checkpoint(accelerator, args):
    # load pretrained model
    accelerator.print(f"Resuming from checkpoint {args.checkpoint_path}")
    accelerator.load_state(args.checkpoint_path)
    

def load_pipelines(vae, flux_transformer, infer_noise_scheduler, weight_dtype, args):
    pipeline_modules = {
        "vae": vae,
        "text_encoder": args.text_encoder_one if hasattr(args, 'text_encoder_one') else None,
        "text_encoder_2": args.text_encoder_two if hasattr(args, 'text_encoder_two') else None,
        "tokenizer": args.tokenizer_one if hasattr(args, 'tokenizer_one') else None,
        "tokenizer_2": args.tokenizer_two if hasattr(args, 'tokenizer_two') else None,
        "scheduler": infer_noise_scheduler,
        "transformer": flux_transformer,
    }
    
    pipeline = FluxPipeline_main.from_pretrained(
        args.pretrained_model_name_or_path,
        safety_checker=None,
        torch_dtype=weight_dtype,
        local_files_only=True,
        **{k: v for k, v in pipeline_modules.items() if v is not None}
    )
    
    return pipeline


def main():
    args = load_config_and_args()

    if args.seed is not None:
        generator = torch.Generator(device=device).manual_seed(args.seed)
        set_seed(args.seed)
    else:
        generator = None

    set_seed(args.seed)
    
    # load accelerator
    accelerator = init_acclerator(args)
    
    # Handle the repository creation
    if accelerator.is_main_process:
        os.makedirs(args.output_dir, exist_ok=True)

    weight_dtype = torch.float32
    if accelerator.mixed_precision == "fp16":
        weight_dtype = torch.float16
    elif accelerator.mixed_precision == "bf16":
        weight_dtype = torch.bfloat16

    # load models
    flux_transformer, infer_noise_scheduler, noise_scheduler_copy, tokenizer_one, tokenizer_two, text_encoder_one, text_encoder_two, vae, clip_model, clip_processor = load_models(args, weight_dtype)
    
    # register adapters to transformer blocks
    register_flux_adapters(flux_transformer, args)
    
    # assign device
    flux_transformer.to(device, dtype=weight_dtype)
    text_encoder_one.to(device, dtype=weight_dtype)
    text_encoder_two.to(device, dtype=weight_dtype)
    vae.to(device, dtype=weight_dtype)
    
    # Load checkpoint
    if args.checkpoint_path.endswith(".bin"):
        state_dict = torch.load(args.checkpoint_path, map_location="cpu")
        flux_transformer.load_state_dict(state_dict, strict=False)
    else:
        # loading checkpoints
        flux_transformer = accelerator.prepare(flux_transformer)
        load_checkpoint(accelerator, args)
        flux_transformer = accelerator.unwrap_model(flux_transformer)

    # Set additional attributes for pipeline creation
    args.text_encoder_one = text_encoder_one
    args.text_encoder_two = text_encoder_two
    args.tokenizer_one = tokenizer_one
    args.tokenizer_two = tokenizer_two

    # loading pipelines
    pipeline = load_pipelines(vae, flux_transformer, infer_noise_scheduler, weight_dtype, args)
    pipeline.to(device, dtype=weight_dtype)

    # subject driven generation
    if "|" in args.subject_img_paths:
        subject_img_paths = args.subject_img_paths.split("|")
    else:
        subject_img_paths = [args.subject_img_paths]
    
    if "|" in args.subject_prompts:
        subject_prompts = args.subject_prompts.split("|")
    else:
        subject_prompts = [args.subject_prompts]
        
    target_prompt = args.target_prompt
    # subject-driven editing
    foreground_mask_path = args.foreground_mask_path
    source_image_path = args.source_image_path
    
    # misc preparation
    train_transforms = transforms.Compose(
        [
            transforms.Resize((args.resolution, args.resolution), interpolation=transforms.InterpolationMode.BILINEAR),
            transforms.ToTensor(),
            transforms.Normalize([0.5], [0.5]),
        ]
    )

    # I/O operations
    post_fix = ""
    if args.do_editing:
        post_fix += "_editing"
    post_fix += f"_seed_{args.seed}"

    # create folder
    if os.path.exists(f"{args.output_root}/{target_prompt}{post_fix}/"):
        shutil.rmtree(f"{args.output_root}/{target_prompt}{post_fix}/")
    os.makedirs(f"{args.output_root}/{target_prompt}{post_fix}", exist_ok=True)

    # generation starts here
    kwargs = {
        # user inputs
        "subject_img_paths": subject_img_paths,
        "target_prompt": target_prompt,
        "subject_prompts": subject_prompts,
        "train_transforms": train_transforms,
        "source_image_path": source_image_path,
        "foreground_mask_path": foreground_mask_path,
        "output_root": args.output_root,

        # system
        "post_fix": post_fix,
        "args": args,
        "accelerator": accelerator,
        "flux_transformer": flux_transformer,
        "text_encoder_one": text_encoder_one,
        "text_encoder_two": text_encoder_two,
        "tokenizer_one": tokenizer_one,
        "tokenizer_two": tokenizer_two,
        "vae": vae,
        "infer_noise_scheduler": infer_noise_scheduler,
        "noise_scheduler_copy": noise_scheduler_copy,
        "weight_dtype": weight_dtype,
        "generator": generator,
        "pipeline": pipeline,
        "clip_model": clip_model,
        "clip_processor": clip_processor,
    }
    iteration_wrapper(**kwargs)
    os._exit(0)


if __name__ == "__main__":
    main()