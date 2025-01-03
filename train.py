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
import copy
import safetensors
import pickle
import glob
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.checkpoint
from torch.utils.data import ConcatDataset, DataLoader
import numpy as np
import cv2
import datasets
import torchvision.transforms as transforms
from packaging import version
from tqdm.auto import tqdm
from matplotlib import pyplot as plt
from omegaconf import OmegaConf
from numba import cuda
import argparse
import yaml
from PIL import Image
import gc
import clip
import transformers

from transformers import CLIPTextModel, CLIPTokenizer, CLIPProcessor, CLIPModel, AutoImageProcessor, AutoModel
from transformers.utils import logging as transformers_logging
from transformers.utils import ContextManagers
from transformers.utils import logging as hf_logging

import diffusers
from diffusers import (
    AutoencoderKL, DDPMScheduler, DDIMScheduler, StableDiffusionPipeline, UNet2DConditionModel, 
    DPMSolverSDEScheduler, DPMSolverMultistepInverseScheduler
)
from diffusers.utils import is_xformers_available, check_min_version, deprecate, is_wandb_available, make_image_grid, convert_state_dict_to_diffusers, check_min_version
from diffusers.training_utils import EMAModel, compute_snr
from diffusers.optimization import get_scheduler
from diffusers.utils.testing_utils import enable_full_determinism
from diffusers.utils.torch_utils import is_compiled_module

from accelerate.utils import ProjectConfiguration, set_seed
from accelerate.state import AcceleratorState
from accelerate.logging import get_logger
from accelerate import Accelerator

from mydatasets import DatasetCOCOCap_unet
from mydatasets.datasets_anydoor import YoutubeVISDataset_unet

from models.inversion_models import InversePipelinePartial, ExceptionCLIPTextModel, partial_inverse
from models.main_unet.unet_main import UNet2DConditionModel_main
from models.reference_unet.unet_ref import UNet2DConditionModel_ref
from models.main_unet.adapter import Attention_Adapter  # my model
from models.pipelines.pipline_sd_main import StableDiffusionPipeline_main

from utils import extract_subject_features, add_noise_to_image, calculate_dino_similarity, compute_clip_similarity, resize_image_to_fit_short, extract_subject_features, random_based_on_time, find_subsequence, prepare_mean_masks_each_word, generate_attn_masks_for_each_block, fill_bounding_rect, expand_foreground_hard, expand_foreground_soft

transformers_logging.set_verbosity_error()
warnings.filterwarnings("ignore", category=FutureWarning, module="diffusers")
warnings.filterwarnings("ignore", category=FutureWarning, module="transformers")


# Will error if the minimal version of diffusers is not installed. Remove at your own risks.
check_min_version("0.27.0.dev0")

logger = get_logger(__name__, log_level="INFO")

device = "cpu" if not torch.cuda.is_available() else "cuda"

# Function for unwrapping if model was compiled with `torch.compile`.
def unwrap_model(model, accelerator):
    model = accelerator.unwrap_model(model)
    model = model._orig_mod if is_compiled_module(model) else model
    return model

def get_trainable_params(model):
    return [param for param in model.parameters() if param.requires_grad]

def get_trainable_params_name(model):
    return [name for (name, param) in model.named_parameters() if param.requires_grad]

def copy_matched_parameters(model_src, model_dst):
    src_state_dict = model_src.state_dict()
    dst_state_dict = model_dst.state_dict()
    
    for name, param in src_state_dict.items():
        if name in dst_state_dict and dst_state_dict[name].shape == param.shape:
            dst_state_dict[name].copy_(param)

def save_random_state(filepath):
    # 使用 torch.save 保存基础状态
    torch.save({
        "python_random_state": random.getstate(),
        "numpy_random_state": np.random.get_state(),
        "torch_random_state": torch.get_rng_state(),
        "torch_cuda_random_state": torch.cuda.get_rng_state_all()
    }, filepath)
    print(f"Random state saved to {filepath}")



def load_random_state(filepath):
    """
    29 Dec 2024, comment:
    This function is adjusted to load random states despite the number of GPUs is different from the previous training.
    In the case of inconsistent GPU numbers, no errors will be prompted, however the random state of each GPU may be differnet.
    Should not affect the result that much.
    """
    random_state = torch.load(filepath)

    random.setstate(random_state["python_random_state"])
    np.random.set_state(random_state["numpy_random_state"])
    torch.set_rng_state(random_state["torch_random_state"])

    
    num_current_gpus = torch.cuda.device_count()

    saved_gpu_states = random_state["torch_cuda_random_state"]
    num_saved_gpus = len(saved_gpu_states)

    if num_current_gpus < num_saved_gpus:
        print(f"Warning: Loaded state has {num_saved_gpus} GPUs, but only {num_current_gpus} GPUs are available. Trimming states.")
        adjusted_gpu_states = saved_gpu_states[:num_current_gpus]
    elif num_current_gpus > num_saved_gpus:
        print(f"Warning: Loaded state has {num_saved_gpus} GPUs, but {num_current_gpus} GPUs are available. Padding states.")
        extra_states = [torch.cuda.get_rng_state()] * (num_current_gpus - num_saved_gpus)
        adjusted_gpu_states = saved_gpu_states + extra_states
    else:
        adjusted_gpu_states = saved_gpu_states

    torch.cuda.set_rng_state_all(adjusted_gpu_states)

    print(f"Random state loaded and adjusted for {num_current_gpus} GPUs from {filepath}")


def iteration_wrapper(args, accelerator, batch_img_path, main_unet, reference_unet, text_encoder, tokenizer, vae,  noise_scheduler, subject_noise, weight_dtype, epoch, batch_text_prompt, batch_origin_text_prompt, batch_subject_prompt, train_transforms, generator, vis_image_dict, val_prompt_idx=None, val_image_out_dir=None, post_fix="", pipeline=None, clip_model=None, clip_processor=None):
    """
    1. load load prompts and ref image features
    2. initial loop, gte the initial image and hard masks (all)
    3. merge masks
    4. Start loop, in each loop:
        i: add target_noise to the given image
        ii: init pipeline
        iii: provide given target_noise and give mask
        iv: 
    """
    # extract subject features
    subject_features = extract_subject_features(args, batch_img_path, reference_unet, text_encoder, tokenizer, vae,  noise_scheduler, None,  weight_dtype, train_transforms, text=batch_subject_prompt, subject_denoise_timestep=args.subject_timestep, device=reference_unet.device, generator=generator.manual_seed(int(args.seed)))

    # reshape subject features
    if subject_features[0].ndim == 2:
        for i in range(len(subject_features)):
            subject_features = [torch.cat((reference[None, :, :], reference[None, :, :]), dim=0) for reference in subject_features]
    elif subject_features[0].ndim == 3 and subject_features[0].shape[0] == 1:
        subject_features = [torch.vstack((reference, reference)) for reference in subject_features]
    
    # inject subject features in in all timesteps, namely the full injection in the ablation study
    args.initial_loop = False
    args.skip_adapter_ratio = 0
    with torch.no_grad():
        res = pipeline(
            batch_text_prompt, 
            num_inference_steps=args.infer_steps, 
            generator=generator.manual_seed(int(args.seed)), 
            subject_features = subject_features, 
            image_paths = batch_img_path,
            subject_noise = subject_noise,
            weight_dtype = weight_dtype,
            train_transforms=train_transforms,
            subject_prompt = batch_subject_prompt,
            args=args,
            latents=None,
            latents_steps=None,
        )
    simple_img = res["images"][0]
    simple_img.save(f"{val_image_out_dir}/{batch_text_prompt}_{post_fix}/simple_img.png")
    vis_image_dict[val_prompt_idx].append(simple_img)
    # ==========================================================================
    
    # ==========================================================================
    args.initial_loop = True
    args.skip_adapter_ratio = 1

    with torch.no_grad():
        # generate using vanilla Stable Diffusion 2.1 base,
        # i.e. totally drop the subject guidance and the adapter, and so-called "pure_text_image"
        res_origin = pipeline(
            batch_text_prompt, 
            num_inference_steps=args.infer_steps, 
            generator=generator.manual_seed(int(args.seed)), 
            subject_features = None, 
            image_paths = batch_img_path,
            reference_unet=reference_unet,
            subject_noise = subject_noise,
            weight_dtype = weight_dtype,
            train_transforms=train_transforms,
            subject_prompt = batch_subject_prompt,
            args=args,
            latents=None,
            latents_steps=0 if args.initial_loop else None,
            guidance_scale=10
        ) 
        pure_text_image = res_origin["images"][0]
        pure_text_image.save(f"{val_image_out_dir}/{batch_text_prompt}_{post_fix}/pure_text_image.png")
        vis_image_dict[val_prompt_idx].append(pure_text_image)
        
        # generate the sketch image and do the first iteration
        # Note that we combine the sketch image generation and the first iteration together for
        #   simple implemetation.
        args.initial_loop = True
        args.skip_adapter_ratio = 1 - args.split_ratio
        res = pipeline(
            batch_text_prompt, 
            num_inference_steps=args.infer_steps, 
            generator=generator.manual_seed(int(args.seed)), 
            subject_features = subject_features, 
            image_paths = batch_img_path,
            reference_unet=reference_unet,
            subject_noise = subject_noise,
            weight_dtype = weight_dtype,
            train_transforms=train_transforms,
            subject_prompt = batch_subject_prompt,
            args=args,
            latents=None,
            negative_prompt="dark, blur, dizzy, black"
            latents_steps=0 if args.initial_loop else None,
            guidance_scale=10
        )

        initial_image = res["images"][0]
        initial_image_resized = initial_image 
        W, H = initial_image.size
        initial_image.save(f"{val_image_out_dir}/{batch_text_prompt}_{post_fix}/initial_loop.png")
        
    # Loop starts
    args.initial_loop = False
    args.skip_adapter_ratio = 0
    final_image = loop_infer(args, batch_img_path, subject_features, vae, noise_scheduler, subject_noise, weight_dtype, batch_text_prompt, batch_subject_prompt, train_transforms, initial_image_resized, generator, sim_threshold=0.99, pipeline=pipeline, val_image_out_dir=val_image_out_dir, post_fix=post_fix, clip_model=clip_model, clip_processor=clip_processor, initial_image_size=(W, H))
    vis_image_dict[val_prompt_idx].append(final_image)
    
    return final_image, vis_image_dict
    

def log_validation_batch(vae, text_encoder, tokenizer, main_unet, reference_unet, noise_scheduler, subject_noise, train_transforms, args, accelerator, weight_dtype, epoch, batch_text_prompts, batch_origin_text_prompts, 
batch_subject_prompt, batch_img_paths, batch_variation_num, clip_model, clip_processor):
    logger.info("Running validation... ")
    num_val_example = len(batch_text_prompts)
    
    if len(batch_origin_text_prompts) == 0:
        batch_origin_text_prompts = batch_text_prompts
    
    pipeline = load_pipelines(vae, text_encoder, tokenizer, main_unet, noise_scheduler, weight_dtype, args)
    pipeline = pipeline.to(accelerator.device)
    
    vis_image_dict = {}
            
    with torch.autocast("cuda"):
        for cur_val_example_id in range(num_val_example):
            print(f"[{cur_val_example_id + 1}/{num_val_example}]")
            val_image_out_dir = args.output_dir
            try:
                variation_num = batch_variation_num[cur_val_example_id]
            except:
                variation_num = 0
                
            generator = torch.Generator(device=accelerator.device)
            
            sub_img_names = ""
            for sub_img_idx, sub_image_path in enumerate(batch_img_paths[cur_val_example_id]):
                sub_img_name_without_png = sub_image_path.split(".png")[0].split("/")[-1]
                sub_img_names += f"_ref{sub_img_idx}_{sub_img_name_without_png}"
                
            post_fix = batch_img_paths[cur_val_example_id][0].split(".png")[0].split("/")[-2] + f"_{variation_num}" + sub_img_names + f"_seed_{args.seed}_{variation_num}"
            
            os.makedirs(f"{val_image_out_dir}/{batch_text_prompts[cur_val_example_id]}_{post_fix}", exist_ok=True)
            files = glob.glob(f"{val_image_out_dir}/{batch_text_prompts[cur_val_example_id]}_{post_fix}/*")
            
            print(f"[Validation {batch_text_prompts[cur_val_example_id]}_{post_fix}] Starts!.")
            
            vis_image_dict[cur_val_example_id] = [batch_text_prompts[cur_val_example_id]]
            
            final_image, vis_image_dict = iteration_wrapper(args, accelerator, batch_img_paths[cur_val_example_id], 
                                            main_unet, reference_unet, text_encoder, tokenizer, vae, noise_scheduler, subject_noise, weight_dtype, epoch, batch_text_prompts[cur_val_example_id], batch_origin_text_prompts[cur_val_example_id], batch_subject_prompt[cur_val_example_id], train_transforms, generator, vis_image_dict, val_prompt_idx=cur_val_example_id, val_image_out_dir = val_image_out_dir, post_fix=post_fix,pipeline=pipeline, clip_model=clip_model, clip_processor=clip_processor)
            print(f"[Validation {batch_text_prompts[cur_val_example_id]}_{post_fix}] ended.")
            
            
            
            del generator
            gc.collect()
            torch.cuda.empty_cache()
            
    for tracker in accelerator.trackers:
        if tracker.name == "tensorboard":
            for cur_val_example_id in range(num_val_example):
                text_prompt, simple_image, pure_text_image, final_image  = vis_image_dict[cur_val_example_id]
                np_images = np.stack([np.asarray(cv2.resize(np.array(img), (args.resolution, args.resolution))) for img in [pure_text_image, simple_image , final_image]])
                tracker.writer.add_images(f"Subject: {{batch_subject_prompt[cur_val_example_id]}}. Target: {batch_text_prompts[cur_val_example_id]} [Vanilla SD2.1 base / Full injection / Generated image]", np_images, epoch, dataformats="NHWC")
        else:
            logger.warn(f"image logging not implemented for {tracker.name}")

    
    del pipeline
    torch.cuda.empty_cache()
    return vis_image_dict


def loop_infer(args, batch_img_path, subject_features, vae, noise_scheduler, subject_noise, weight_dtype, batch_text_prompt, batch_subject_prompt, train_transforms, init_image, generator, sim_threshold=0.98, pipeline=None, val_image_out_dir=None, post_fix=None,clip_model=None, clip_processor=None, initial_image_size = None):
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
        # add noise and convert the generated image back to a sketch image
        noisy_latents = add_noise_to_image(noise_step = split_ratio * noise_scheduler.config.num_train_timesteps, args=args, img=loop_image, vae=vae, train_transforms=train_transforms, noise_scheduler=noise_scheduler)
        
        skipped_steps = int(args.infer_steps - split_ratio * args.infer_steps)
        with torch.no_grad():
            res = pipeline(
                batch_text_prompt, 
                num_inference_steps=args.infer_steps, 
                generator=generator, # don't assign random seed here, allow the model to denoise with some randomness in case of over-denoise during the interation.
                subject_features = subject_features, 
                image_paths = batch_img_path,
                subject_noise = subject_noise,
                weight_dtype = weight_dtype,
                train_transforms=train_transforms,
                subject_prompt = batch_subject_prompt,
                args=args,
                latents=noisy_latents,
                latents_steps=skipped_steps,
                guidance_scale= 7.5,
                # negative_prompt='dark, dim, blur, dizzy, defocus',
                inversed_intermediate_latents=None
            )
        loop_image = res["images"][0]
        origin_loop_image = loop_image.resize(initial_image_size)
        origin_loop_image.save(f"{val_image_out_dir}/{batch_text_prompt}_{post_fix}/loop_{cur_loop_num}.png")
        
        sims = 0
        sim = compute_clip_similarity(clip_model, clip_processor, image1=prev_image, image2=loop_image, device=subject_noise.device)
        prev_image = loop_image
        
        print(f"[Validation {batch_text_prompt}_{post_fix}][Loop {cur_loop_num}] Overall Similarity: {sim}.")
        cur_loop_num += 1
        
    torch.cuda.empty_cache()
    return loop_image

def extract_attention_params(attn1):
    return {
        'query_dim': attn1.query_dim,
        'num_attention_heads': attn1.heads,
        'dropout': attn1.dropout,
        'attention_head_dim': attn1.dim_head,
        'attention_bias': attn1.use_bias,
        'upcast_attention': attn1.upcast_attention,
        'attention_out_bias': attn1.out_bias,
        'cross_attention_dim': None,
    }
    
def initialize_adapter(params, args):
    return Attention_Adapter(
        query_dim=params['query_dim'],
        heads=params['num_attention_heads'],
        dim_head=params['attention_head_dim'],
        dropout=params['dropout'],
        bias=params['attention_bias'],
        cross_attention_dim=params['cross_attention_dim'],
        upcast_attention=params['upcast_attention'],
        out_bias=params['attention_out_bias'],
        residual_connection=args.residual_connection,
    )

def parse_args_from_yaml(config_path=None, config_file=None):
    assert (config_path is None) or (config_file is None), "Cannot have multiple config input!!"
    if config_file is None:
        if (config_path is None):
            parser = argparse.ArgumentParser(description="Example with configuration file.")
            parser.add_argument("--config", type=str, default="configs/sub_img_trans_unet.yaml", help="Path to the configuration file.")
            args = parser.parse_args()
            
            with open(args.config, 'r') as f:
                config = yaml.safe_load(f)
            del args
            del parser
        elif config_path is not None:
            with open(config_path, 'r') as f:
                config = yaml.safe_load(f)
    else:
        config = config_file

    parser = argparse.ArgumentParser(description="Simple example of a training script.")
    
    for key, value in config.items():
        parser.add_argument(f"--{key}", type=type(value), default=value)
    args = parser.parse_args()
    
    env_local_rank = int(os.environ.get("LOCAL_RANK", -1))
    if env_local_rank != -1 and env_local_rank != args.local_rank:
        args.local_rank = env_local_rank

    return args


def init_acclerator(args):
    logging_dir = os.path.join(args.output_dir, args.logging_dir)
    accelerator_project_config = ProjectConfiguration(project_dir=args.output_dir, logging_dir=logging_dir)
    accelerator = Accelerator(log_with=args.report_to, gradient_accumulation_steps=args.gradient_accumulation_steps,mixed_precision='no',project_config=accelerator_project_config,)
    
    if accelerator.is_local_main_process:
        datasets.utils.logging.set_verbosity_warning()
        transformers.utils.logging.set_verbosity_warning()
        diffusers.utils.logging.set_verbosity_info()
    else:
        datasets.utils.logging.set_verbosity_error()
        transformers.utils.logging.set_verbosity_error()
        diffusers.utils.logging.set_verbosity_error()
    
    return accelerator

def load_models_and_learnable_params(args, device):
    # load base models
    noise_scheduler = DDPMScheduler.from_pretrained(args.pretrained_model_name_or_path, subfolder="scheduler", local_files_only=True)
    tokenizer = CLIPTokenizer.from_pretrained(args.pretrained_model_name_or_path, subfolder="tokenizer")
    text_encoder = CLIPTextModel.from_pretrained(args.pretrained_model_name_or_path, subfolder="text_encoder", revision=args.revision, local_files_only=True)
    vae = AutoencoderKL.from_pretrained(args.pretrained_model_name_or_path, subfolder="vae", revision=args.revision, local_files_only=True)
    main_unet = UNet2DConditionModel_main.from_pretrained(args.pretrained_model_name_or_path, subfolder="unet", revision=args.non_ema_revision, local_files_only=True)
    reference_unet = UNet2DConditionModel_ref(args=args).from_pretrained(args.pretrained_model_name_or_path, subfolder="unet", revision=args.non_ema_revision, local_files_only=True)
    
    # clip_model, clip_processor = clip.load("ViT-B/32", device=device)
    clip_model, clip_processor = clip.load(args.clip_path, device=device)
    
    # clip_model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32", local_files_only=True)
    # clip_processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32", local_files_only=True)
    
    # freeze everything first
    vae.requires_grad_(False)
    text_encoder.requires_grad_(False)
    main_unet.requires_grad_(False)
    reference_unet.requires_grad_(False)
    
    # register Adapter to main_unet attention blocks, and also register some configs inside UNets, also optioanlly register trainable parameters, and set those params trainable
    register_adapter_and_configs(main_unet, reference_unet, args)
    
    return main_unet, reference_unet, noise_scheduler, tokenizer, text_encoder, vae, clip_model, clip_processor


def register_adapter_and_configs(main_unet, reference_unet, args):
    all_blocks = nn.ModuleList([])
    all_blocks.extend(main_unet.down_blocks)
    all_blocks.append(main_unet.mid_block)
    all_blocks.extend(main_unet.up_blocks)

    # register
    for num_down, unet_block in enumerate(all_blocks):
        if hasattr(unet_block, "has_cross_attention") and unet_block.has_cross_attention:
            for num_attn, attn in enumerate(unet_block.attentions):
                attn_1 = attn.transformer_blocks[0].attn1
                attn_2 = attn.transformer_blocks[0].attn2
                norm_1 = attn.transformer_blocks[0].norm1
                norm_2 = attn.transformer_blocks[0].norm2

                # extract Adapter parameters
                self_attention_module_params = extract_attention_params(attn_1)

                # initialize Adapter
                adapter = initialize_adapter(self_attention_module_params, args)
                
                # obtain norm parameters
                norm_eps = unet_block.attentions[num_attn].transformer_blocks[0].norm_eps
                norm_elementwise_affine = unet_block.attentions[num_attn].transformer_blocks[0].norm_elementwise_affine

                # initialize norm
                adapter_norm = nn.LayerNorm(self_attention_module_params['query_dim'], elementwise_affine=norm_elementwise_affine, eps=norm_eps)

                # init from text cross block
                copy_matched_parameters(attn_1, adapter)
                copy_matched_parameters(norm_1, adapter_norm)

                # set adapter trainable
                adapter.requires_grad_(True)

                # register blocks and configs
                adapter.args = args
                attn_1.args = args
                attn_2.args = args
                attn.transformer_blocks[0].args = args
                attn.transformer_blocks[0].adapter = adapter
                attn.transformer_blocks[0].adapter_norm = adapter_norm
                
    reference_unet.args = args            

    # (Optional) Register some weighting factors for the extracted features and set them trainable
    main_unet.learnable_weights = nn.Parameter(torch.ones(16)).requires_grad_(True)
    

def load_checkpoint(accelerator, args):
    # load pretrained model
    accelerator.print(f"Resuming from checkpoint {args.checkpoint_path}")
    accelerator.load_state(args.checkpoint_path)
    

def load_pipelines(vae, text_encoder, tokenizer, main_unet, noise_scheduler, weight_dtype, args):
    pipeline = StableDiffusionPipeline_main.from_pretrained(
        args.pretrained_model_name_or_path,
        vae=vae,
        text_encoder=text_encoder,
        tokenizer=tokenizer,
        unet=main_unet,
        safety_checker=None,
        revision=args.revision,
        variant=args.variant,
        torch_dtype=weight_dtype,
        local_files_only=True
    )
    pipeline.scheduler = noise_scheduler

    # we don't need inversion pipeline during training, please refer to infer.py for usage
    # inverse_pipeline = InversePipelinePartial.from_pretrained(args.pretrained_model_name_or_path, text_encoder=exclip, local_files_only=True)
    # inverse_pipeline.scheduler = DPMSolverMultistepInverseScheduler.from_config(inverse_pipeline.scheduler.config, local_files_only=True)
    return pipeline


def padding_subjects(input_tensor, padding_num):
    """
        input_tensor: tensor with shape [B, ...]
        padding num: number of batch to pad
    """
    if padding_num < 0:
        breakpoint()
    empty_tensor = torch.zeros_like(input_tensor)[:1, ...].repeat(padding_num, *([1] * (input_tensor.ndimension() - 1)))
    input_tensor = torch.concat((input_tensor, empty_tensor), dim=0)
    
    return input_tensor


def main(config_path=None, config_file=None):
    # prepare enironment and spaces
    weight_dtype = torch.float32
    args = parse_args_from_yaml(config_path=config_path, config_file=config_file)
    set_seed(args.seed)
    
    # Make one log on every process with the configuration for debugging.
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO,
    )
    
    # load accelerator
    accelerator = init_acclerator(args)
    
    # Handle the repository creation
    if accelerator.is_main_process:
        os.makedirs(args.output_dir, exist_ok=True)

    # load models, here, only main_UNet would contains trainable parameters while reference_UNet is merely a identical copy of SD2.1-base main_unet used for feature extraction
    main_unet, reference_unet, noise_scheduler, tokenizer, text_encoder, vae,  clip_model, clip_processor = load_models_and_learnable_params(args, accelerator.device)
    
    # Move text_encode and vae to gpu and cast to weight_dtype
    text_encoder.to(accelerator.device, dtype=weight_dtype)
    vae.to(accelerator.device, dtype=weight_dtype)
    reference_unet.to(accelerator.device, dtype=weight_dtype)
    
    # get trainable params for optimizer
    trainable_params = get_trainable_params(main_unet)
    trainable_params_name = get_trainable_params_name(main_unet)

    if args.scale_lr:
        args.learning_rate = (args.learning_rate * args.gradient_accumulation_steps * args.train_batch_size * accelerator.num_processes)
        
    optimizer = torch.optim.AdamW(
        trainable_params,
        lr=args.learning_rate,
        betas=(args.adam_beta1, args.adam_beta2),
        weight_decay=args.adam_weight_decay,
        eps=args.adam_epsilon,
    )
    
    train_transforms = transforms.Compose(
        [
            transforms.Resize((args.resolution, args.resolution), interpolation=transforms.InterpolationMode.BILINEAR),
            transforms.ToTensor(),
            transforms.Normalize([0.5], [0.5]),
        ]
    )
    
    dataset_to_train =[]
    if args.with_coco:
        coco2014_dataset = DatasetCOCOCap_unet(args.coco2014["data_path"], transform=train_transforms, max_len=args.num_sub_img, tokenizer=tokenizer, train_split=args.coco2014["train_split"], args=args, subset_size=args.coco2014['subset_size'])
        dataset_to_train.append(coco2014_dataset)
        
    if args.with_youtube_vis:
        youtubeVIS_dataset = YoutubeVISDataset_unet(image_dir=args.youtubeVIS['image_dir'], anno=args.youtubeVIS['anno'], meta=args.youtubeVIS['meta'], tokenizer=tokenizer, sub_size=args.resolution, transforms=train_transforms, ytbvis_subset_size=args.youtubeVIS['subset_size'], args=args)
        dataset_to_train.append(youtubeVIS_dataset)
    
    assert len(dataset_to_train) > 0, "No dataset is loaded!"
    
    train_dataset = ConcatDataset(dataset_to_train)

    
    def collate_fn(examples):
        target_image = torch.stack([example["target_image"] for example in examples]) 
        target_image = target_image.to(memory_format=torch.contiguous_format).float()
            
        input_ids = torch.stack([example["input_ids"] for example in examples])
        subject_input_ids = torch.vstack([padding_subjects(example["subject_input_ids"], args.num_sub_img - 1 if example["dataset_name"] == "youtubeVIS" else example["padding_num"]) for example in examples])
        
        subject_images = torch.vstack([padding_subjects(example["subject_images"], args.num_sub_img - 1 if example["dataset_name"] == "youtubeVIS" else example["padding_num"]) for example in examples]) # B * N, C, W, H
        # we need to mask out the black padded subject imagee
        padding_nums = torch.tensor([3 if example["dataset_name"] == "youtubeVIS" else example["padding_num"] for example in examples])
            
        # some training hyperparams
        timestep = torch.randint(0, noise_scheduler.config.num_train_timesteps, (args.train_batch_size, ))
        target_noise = torch.randn((args.train_batch_size, 4, 64, 64), dtype=weight_dtype)
        subject_noise = torch.randn((1, 4, 64, 64), dtype=weight_dtype)
        
        target_prompt = [example["target_prompt"] for example in examples]
        subject_prompt = [example["subject_prompt"] for example in examples]
        
        return {"target_image": target_image, "input_ids": input_ids, "subject_input_ids": subject_input_ids, "subject_images": subject_images, "padding_num": padding_nums, "dataset_name": examples[0]["dataset_name"], "timestep": timestep, "target_prompt": target_prompt, "subject_prompt": subject_prompt, "target_noise": target_noise, "subject_noise": subject_noise}
    
    # DataLoaders creation:
    train_dataloader = torch.utils.data.DataLoader(
        train_dataset,
        shuffle=True,
        collate_fn=collate_fn,
        batch_size=args.train_batch_size,
        num_workers=args.dataloader_num_workers,
        pin_memory=True
    )

    ## Scheduler and math around the number of training steps.
    # Check the PR https://github.com/huggingface/diffusers/pull/8312 for detailed explanation.
    num_warmup_steps_for_scheduler = args.lr_warmup_steps * accelerator.num_processes
    if args.max_train_steps is None:
        len_train_dataloader_after_sharding = math.ceil(len(train_dataloader) / accelerator.num_processes)
        num_update_steps_per_epoch = math.ceil(len_train_dataloader_after_sharding / args.gradient_accumulation_steps)
        num_training_steps_for_scheduler = (
            args.num_train_epochs * num_update_steps_per_epoch * accelerator.num_processes
        )
    else:
        num_training_steps_for_scheduler = args.max_train_steps * accelerator.num_processes
    
    lr_scheduler = get_scheduler(
        args.lr_scheduler,
        optimizer=optimizer,
        num_warmup_steps=num_warmup_steps_for_scheduler,
        num_training_steps=num_training_steps_for_scheduler,
    )
    
    # Prepare everything with our `accelerator`.
    main_unet, optimizer, lr_scheduler, train_dataloader = accelerator.prepare(
        main_unet, optimizer, lr_scheduler, train_dataloader
    )
    
    # Potentially load in the weights and states from a previous save
    if args.resume_from_checkpoint:
        if args.resume_from_checkpoint != "latest":
            path = os.path.basename(args.resume_from_checkpoint)
        else:
            # Get the most recent checkpoint
            dirs = os.listdir(args.output_dir)
            dirs = [d for d in dirs if d.startswith("checkpoint")]
            dirs = sorted(dirs, key=lambda x: int(x.split("-")[1]))
            path = dirs[-1] if len(dirs) > 0 else None

        if path is None:
            accelerator.print(
                f"Checkpoint '{args.resume_from_checkpoint}' does not exist. Starting a new training run."
            )
            args.resume_from_checkpoint = None
            initial_global_step = 0
        else:
            accelerator.print(f"Resuming from checkpoint {path}")
            load_path = os.path.join(args.output_dir, path)
            global_step = int(path.split("-")[1])
            accelerator.load_state(load_path)
            
            initial_global_step = global_step
            first_epoch = global_step // num_update_steps_per_epoch
            resumed = True # to mark if the model is freshly resumed from local
    else:
        initial_global_step = 0
        resumed = False
    
    
    
    # We need to recalculate our total training steps as the size of the training dataloader may have changed (after prepare).
    num_update_steps_per_epoch = math.ceil(len(train_dataloader) / args.gradient_accumulation_steps)
    if args.max_train_steps is None:
        args.max_train_steps = args.num_train_epochs * num_update_steps_per_epoch
        if num_training_steps_for_scheduler != args.max_train_steps * accelerator.num_processes:
            logger.warning(
                f"The length of the 'train_dataloader' after 'accelerator.prepare' ({len(train_dataloader)}) does not match "
                f"the expected length ({len_train_dataloader_after_sharding}) when the learning rate scheduler was created. "
                f"This inconsistency may result in the learning rate scheduler not functioning properly."
            )
    # Afterwards we recalculate our number of training epochs
    args.num_train_epochs = math.ceil(args.max_train_steps / num_update_steps_per_epoch)
    
    if accelerator.is_main_process:
        valid_types = (int, float, str, bool, torch.Tensor)
        tracker_config = {key: value for key, value in dict(vars(args)).items() if isinstance(value, valid_types)}
        accelerator.init_trackers(args.tracker_project_name, tracker_config)

    # Train!
    total_batch_size = args.train_batch_size * accelerator.num_processes * args.gradient_accumulation_steps

    logger.info("***** Running training *****")
    logger.info(f"  Num examples = {len(train_dataset)}")
    logger.info(f"  Num Epochs = {args.num_train_epochs}")
    logger.info(f"  Instantaneous batch size per device = {args.train_batch_size}")
    logger.info(f"  Total train batch size (w. parallel, distributed & accumulation) = {total_batch_size}")
    logger.info(f"  Gradient Accumulation steps = {args.gradient_accumulation_steps}")
    logger.info(f"  Total optimization steps = {args.max_train_steps}")
    global_step = 0
    first_epoch = 0

    progress_bar = tqdm(
        range(0, args.max_train_steps),
        initial=initial_global_step,
        desc="Steps",
        # Only show the progress bar once on each machine.
        disable=not accelerator.is_local_main_process,
    )
    
    if accelerator.is_main_process:
        batch_img_paths = []
        batch_text_prompts = []
        batch_origin_text_prompts = []
        batch_subject_prompts = []
        batch_variation_num = []
        for arg_name in vars(args):
            arg_value = getattr(args, arg_name)
            if isinstance(arg_name, str) and arg_name.startswith("subject_img_paths"):
                batch_img_paths.append(arg_value)
            if isinstance(arg_name, str) and arg_name.startswith("target_prompt"):
                batch_text_prompts.append(arg_value)
            if isinstance(arg_name, str) and arg_name.startswith("origin_target_prompt"):
                batch_origin_text_prompts.append(arg_value)
            if isinstance(arg_name, str) and arg_name.startswith("subject_prompt"):
                batch_subject_prompts.append(arg_value)
            if isinstance(arg_name, str) and arg_name.startswith("variation_num_"):
                batch_variation_num.append(arg_value)
    
    for epoch in range(first_epoch, args.num_train_epochs):
        train_loss = 0.0
        for step, batch in enumerate(train_dataloader):
            # load target related data
            target_image = batch["target_image"].to(weight_dtype)
            target_input_ids = batch["input_ids"]
            target_noise = batch['target_noise']
            target_timestep = batch['timestep']
            padding_nums = batch['padding_num']
            target_timestep = target_timestep.long()
            
            # load subject related data
            subject_images = batch["subject_images"].to(weight_dtype)
            subject_input_ids = batch["subject_input_ids"]
            subject_noise = batch['subject_noise']
            subject_timestep = torch.tensor(args.subject_timestep, device=subject_images.device).repeat(subject_images.shape[0])
            subject_timestep = subject_timestep.long()
            if subject_noise.shape[0] != subject_images.shape[0]:
                subject_noise = subject_noise.repeat(subject_images.shape[0], 1, 1, 1)
            
            breakpoint()
            
            # preprocess subject related features
            subject_latents = vae.encode(subject_images.to(weight_dtype)).latent_dist.sample()
            subject_latents = subject_latents * vae.config.scaling_factor
            noisy_subject_latents = noise_scheduler.add_noise(subject_latents, subject_noise, subject_timestep) # add [t=1]s to the latent
            subject_encoder_hidden_states = text_encoder(subject_input_ids, return_dict=False)[0]
            breakpoint()
            # obtain subject features from reference UNet for later usage
            _, subject_features = reference_unet(noisy_subject_latents, subject_timestep, subject_encoder_hidden_states, return_dict=False, args=args) 
            subject_features = [block_feat.reshape(args.train_batch_size,-1,block_feat.shape[-1]) for block_feat in subject_features] # bsz ,sub_image_patches (later concat to k and v), dim 
            
            # random drop subject feature
            if random.randint(1,10) / 10 > args.drop_reference_ratio:
                # turn to dummy inputs, and make sure to pad
                subject_features = [torch.zeros_like(subject_features[i]).to(weight_dtype) for i in range(16)]
                padding_nums = torch.tensor([args.num_sub_img] * args.train_batch_size)
            
            breakpoint()
            # preprocess target related features
            target_latents = vae.encode(target_image).latent_dist.sample() # [bsz, 4, 64, 64]
            target_latents = target_latents * vae.config.scaling_factor
            noisy_target_latents = noise_scheduler.add_noise(target_latents, target_noise, target_timestep)
            target_encoder_hidden_states = text_encoder(target_input_ids, return_dict=False)[0]
                
            # Get the target for loss depending on the prediction type
            if args.prediction_type is not None:
                # set prediction_type of scheduler if defined
                noise_scheduler.register_to_config(prediction_type=args.prediction_type)
            if noise_scheduler.config.prediction_type == "epsilon":
                target = target_noise
            elif noise_scheduler.config.prediction_type == "v_prediction":
                target = noise_scheduler.get_velocity(target_latents, target_noise, target_timestep)
            else:
                raise ValueError(f"Unknown prediction type {noise_scheduler.configprediction_type}")
                
            # construct attention mask for training
            training_attn_mask = None
            if padding_nums is not None:
                training_attn_mask = torch.zeros(args.train_batch_size, 2, 2 * (args.num_sub_img + 1)).to(noisy_target_latents.device)
                for i in range(args.train_batch_size):
                    training_attn_mask[i, :, - padding_nums[i] * 2:] = -10000.0
                training_attn_mask = training_attn_mask.to(weight_dtype)
            
            # obtain predicted noise
            model_pred = main_unet(noisy_target_latents, target_timestep, target_encoder_hidden_states,return_dict=False, subject_feats=subject_features, training_attn_mask=training_attn_mask)[0]
            
            if args.snr_gamma is None:
                loss = F.mse_loss(model_pred.float(), target.float(), reduction="mean")
            else:
                # Compute loss-weights as per Section 3.4 of https://arxiv.org/abs/2303.09556.
                # Since we predict the target_noise instead of x_0, the original formulation isslightly changed.
                # This is discussed in Section 4.2 of the same paper.
                snr = compute_snr(noise_scheduler, target_timestep)
                mse_loss_weights = torch.stack([snr, args.snr_gamma * torch.ones_like(target_timestep)], dim=1).min(dim=1)[0]
                if noise_scheduler.config.prediction_type == "epsilon":
                        mse_loss_weights = mse_loss_weights / snr
                elif noise_scheduler.config.prediction_type == "v_prediction":
                        mse_loss_weights = mse_loss_weights / (snr + 1)
                    
                loss = F.mse_loss(model_pred.float(), target.float(), reduction="none")
                loss = loss.mean(dim=list(range(1, len(loss.shape)))) * mse_loss_weights
                loss = loss.mean()
                
                # Gather the losses across all processes for logging (if we use distributedtraining).
            avg_loss = accelerator.gather(loss.repeat(args.train_batch_size)).mean()
            train_loss += avg_loss.item() / args.gradient_accumulation_steps
                
            accelerator.backward(loss)
            if accelerator.sync_gradients:
                accelerator.clip_grad_norm_(trainable_params, args.max_grad_norm)
            optimizer.step()
            lr_scheduler.step()
            optimizer.zero_grad()
                
            if step % args.validation_steps == 0:
                # display the
                for tracker in accelerator.trackers:
                    if tracker.name == "tensorboard":
                        unwrapped_main_unet = unwrap_model(main_unet, accelerator)
                        values = unwrapped_main_unet.learnable_weights.data.detach().cpu()
                        plt.figure(figsize=(10, 5))
                        plt.bar(range(1, 17), values.numpy())  # 创建柱状图
                        plt.title('learnable_weights')
                        tracker.writer.add_figure('learnable_weights', plt.gcf(), step)
                        plt.close()
                
            # Checks if the accelerator has performed an optimization step behind the scenesw
            if accelerator.sync_gradients:
                progress_bar.update(1)
                global_step += 1
                accelerator.log({"train_loss": train_loss}, step=global_step)
                train_loss = 0.0
                if global_step % args.checkpointing_steps == 0:
                    if accelerator.is_main_process:
                        # _before_ saving state, check if this save would set us overthe`checkpoints_total_limit`
                        if args.checkpoints_total_limit is not None:
                            checkpoints = os.listdir(args.output_dir)
                            checkpoints = [d for d in checkpoints if d.startswith("checkpoint")]
                            checkpoints = sorted(checkpoints, key=lambda x: int(x.split("-")[1]))
                            # before we save the new checkpoint, we need to have at_most_`checkpoints_total_limit - 1` checkpoints
                            if len(checkpoints) >= args.checkpoints_total_limit:
                                num_to_remove = len(checkpoints) - args.checkpoints_total_limit + 1
                                removing_checkpoints = checkpoints[0:num_to_remove]
                                logger.info(
                                        f"{len(checkpoints)} checkpoints already exist, removing {len(removing_checkpoints)} checkpoints"
                                )
                                logger.info(f"removing checkpoints: {', '.join(removing_checkpoints)}")
                                for removing_checkpoint in removing_checkpoints:
                                    removing_checkpoint = os.path.join(args.output_dir,removing_checkpoint)
                                    shutil.rmtree(removing_checkpoint)
                                        
                        save_path = os.path.join(args.output_dir, f"checkpoint-{global_step}")
                        
                        accelerator.save_state(save_path)
                        resumed = False
                        # accelerator.save_model(main_unet, save_path)
                        # save_random_state(os.path.join(save_path, "random_states_0.pth"))
                        logger.info(f"Saved state to {save_path}")
            
                logs = {"step_loss": loss.detach().item(), "lr": lr_scheduler.get_last_lr()[0]}
                progress_bar.set_postfix(**logs)
                if global_step >= args.max_train_steps:
                    break
            
            if args.target_prompt_1 is not None and step % args.validation_steps == 0 and accelerator.is_main_process:
                vis_images = log_validation_batch(accelerator.unwrap_model(vae), accelerator.unwrap_model(text_encoder), tokenizer, accelerator.unwrap_model(main_unet), reference_unet, noise_scheduler, subject_noise, train_transforms, args, accelerator, weight_dtype, epoch, batch_text_prompts, batch_origin_text_prompts, batch_subject_prompts, batch_img_paths, batch_variation_num, clip_model, clip_processor)
                    
                        
    accelerator.wait_for_everyone()
    if accelerator.is_main_process:
        if not resumed:
            print("Final saving!")
            save_path = os.path.join(args.output_dir, f"checkpoint-{global_step}")
            accelerator.save_model(main_unet, save_path)
            save_random_state(os.path.join(save_path, "random_states_0.pth"))
            logger.info(f"Final model saved state to {save_path}")
        
        print("Final eval!")
        vis_images = log_validation_batch(accelerator.unwrap_model(vae), accelerator.unwrap_model(text_encoder), tokenizer, accelerator.unwrap_model(main_unet), reference_unet, noise_scheduler, subject_noise, train_transforms, args, accelerator, weight_dtype, epoch, batch_text_prompts, batch_origin_text_prompts, batch_subject_prompts, batch_img_paths, batch_variation_num, clip_model, clip_processor)

    accelerator.end_training()
    os._exit(0)

if __name__ == "__main__":
    main(config_path=None, config_file=None)