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
import signal
import torch
import copy
from diffusers.pipelines.stable_diffusion.pipeline_stable_diffusion import retrieve_timesteps
import torch.nn as nn
from datetime import timedelta
from torch.cuda.amp import autocast, GradScaler
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
import argparse
import yaml
from PIL import Image
import gc
import clip
import transformers

from transformers import CLIPTextModel, CLIPTokenizer, CLIPProcessor, CLIPModel, AutoImageProcessor, AutoModel, AutoTokenizer, CLIPTextModelWithProjection
from transformers.utils import logging as transformers_logging
from transformers.utils import ContextManagers
from transformers.utils import logging as hf_logging
import wandb
import diffusers
from diffusers import (
    AutoencoderKL, DDPMScheduler, DDIMScheduler, StableDiffusionPipeline, UNet2DConditionModel, 
    DPMSolverSDEScheduler, DPMSolverMultistepInverseScheduler
)
from diffusers.utils import is_xformers_available, check_min_version, deprecate, is_wandb_available, make_image_grid, convert_state_dict_to_diffusers, check_min_version
from diffusers.training_utils import EMAModel, compute_snr, cast_training_params
from diffusers.optimization import get_scheduler
from diffusers.utils.testing_utils import enable_full_determinism
from diffusers.utils.torch_utils import is_compiled_module

from accelerate.utils import ProjectConfiguration, set_seed
from accelerate.state import AcceleratorState
from accelerate.logging import get_logger
from accelerate import Accelerator, InitProcessGroupKwargs

from mydatasets import DatasetCOCO_sdxl, Subject200k_dataset_sdxl, Subject200k_dataset_sdxl_jigsaw_sdxl
from mydatasets.datasets_anydoor import YoutubeVISDataset_unet_sdxl, VitonHDDataset_unet

from models.inversion_models import InversePipelinePartial, ExceptionCLIPTextModel, partial_inverse
from models.main_unet.unet_main import UNet2DConditionModel_main
from models.reference_unet.unet_ref import UNet2DConditionModel_ref
from models.main_unet.adapter import Attention_Adapter  # my model
from models.pipelines.pipline_sd_main import StableDiffusionPipeline_main
from models.pipelines.pipline_sdxl_main import StableDiffusionXLPipeline_main

from utils import extract_subject_features, extract_subject_features_sdxl, add_noise_to_image, calculate_dino_similarity, compute_clip_similarity, resize_image_to_fit_short, random_based_on_time, find_subsequence, prepare_mean_masks_each_word, generate_attn_masks_for_each_block, fill_bounding_rect, expand_foreground_hard, expand_foreground_soft

from accelerate.utils import DeepSpeedPlugin

transformers_logging.set_verbosity_error()
warnings.filterwarnings("ignore", category=FutureWarning, module="diffusers")
warnings.filterwarnings("ignore", category=FutureWarning, module="transformers")


# Will error if the minimal version of diffusers is not installed. Remove at your own risks.
check_min_version("0.27.0.dev0")

logger = get_logger(__name__, log_level="INFO")

device = "cpu" if not torch.cuda.is_available() else "cuda"

os.environ["CUDA_LAUNCH_BLOCKING"] = "1"

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


def iteration_wrapper(args, accelerator, batch_img_path, main_unet, reference_unet, text_encoder_one, text_encoder_two, tokenizer_one, tokenizer_two, vae,  noise_scheduler, subject_noise, weight_dtype, epoch, batch_text_prompt, batch_origin_text_prompt, batch_subject_prompt, train_transforms, generator, vis_image_dict, val_prompt_idx=None, val_image_out_dir=None, post_fix="", pipeline=None, clip_model=None, clip_processor=None):
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
    subject_features = extract_subject_features_sdxl(args, batch_img_path, reference_unet, text_encoder_one, text_encoder_two, tokenizer_one, tokenizer_two, vae,  noise_scheduler, None,  weight_dtype, train_transforms, text=batch_subject_prompt, subject_denoise_timestep=args.subject_timestep, device=reference_unet.device, generator=generator.manual_seed(int(args.seed)))

    # reshape subject features
    if subject_features[0].ndim == 2:
        for i in range(len(subject_features)):
            subject_features = [torch.cat((reference[None, :, :], reference[None, :, :]), dim=0) for reference in subject_features]
    elif subject_features[0].ndim == 3 and subject_features[0].shape[0] == 1:
        subject_features = [torch.vstack((reference, reference)) for reference in subject_features]
    
    # # inject subject features in in all timesteps, namely the full injection in the ablation study
    args.initial_loop = False
    args.skip_adapter_ratio = 0
    with torch.no_grad():
        res = pipeline(
            batch_text_prompt, 
            num_inference_steps=args.infer_steps, 
            generator=generator.manual_seed(int(args.seed)), 
            subject_features = subject_features, 
            subject_noise = subject_noise,
            weight_dtype = weight_dtype,
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
            subject_noise = subject_noise,
            weight_dtype = weight_dtype,
            args=args,
            latents=None,
            latents_steps=0 if args.initial_loop else None,
            guidance_scale=10
        ) 
        pure_text_image = res_origin["images"][0]
        pure_text_image.save(f"{val_image_out_dir}/{batch_text_prompt}_{post_fix}/pure_text_image.png")
        vis_image_dict[val_prompt_idx].append(pure_text_image)
        
        if args.init_image_path is None:
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
                subject_noise = subject_noise,
                weight_dtype = weight_dtype,
                args=args,
                latents=None,
                latents_steps=0 if args.initial_loop else None,
                guidance_scale=10
            )

            initial_image = res["images"][0]
        else:
            initial_image = Image.open(args.init_image_path).resize((args.resolution, args.resolution))
            
        initial_image_resized = initial_image 
        W, H = initial_image.size
        initial_image.save(f"{val_image_out_dir}/{batch_text_prompt}_{post_fix}/initial_loop.png")
        
    # Loop starts
    args.initial_loop = False
    args.skip_adapter_ratio = 0
    final_image = loop_infer(args, batch_img_path, subject_features, vae, noise_scheduler, subject_noise, weight_dtype, batch_text_prompt, batch_subject_prompt, train_transforms, initial_image_resized, generator, sim_threshold=0.99, pipeline=pipeline, val_image_out_dir=val_image_out_dir, post_fix=post_fix, clip_model=clip_model, clip_processor=clip_processor, initial_image_size=(W, H))
    vis_image_dict[val_prompt_idx].append(final_image)
    
    return final_image, vis_image_dict
    

def log_validation_batch(vae, text_encoder_one, text_encoder_two, tokenizer_one, tokenizer_two, main_unet, reference_unet, noise_scheduler, subject_noise, train_transforms, args, accelerator, weight_dtype, epoch, batch_text_prompts, batch_origin_text_prompts, 
batch_subject_prompt, batch_img_paths, batch_variation_num, clip_model, clip_processor):
    logger.info("Running validation... ")
    num_val_example = len(batch_text_prompts)
    
    if len(batch_origin_text_prompts) == 0:
        batch_origin_text_prompts = batch_text_prompts
    
    pipeline = load_pipelines(vae, text_encoder_one, text_encoder_two, tokenizer_one, tokenizer_two, main_unet, noise_scheduler, weight_dtype, args)
    pipeline.to(accelerator.device)
    pipeline.to(weight_dtype)
    vis_image_dict = {}

    with torch.no_grad():
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
                                            main_unet, reference_unet, text_encoder_one, text_encoder_two, tokenizer_one, tokenizer_two, vae, noise_scheduler, subject_noise, weight_dtype, epoch, batch_text_prompts[cur_val_example_id], batch_origin_text_prompts[cur_val_example_id], batch_subject_prompt[cur_val_example_id], train_transforms, generator, vis_image_dict, val_prompt_idx=cur_val_example_id, val_image_out_dir = val_image_out_dir, post_fix=post_fix,pipeline=pipeline, clip_model=clip_model, clip_processor=clip_processor)
            print(f"[Validation {batch_text_prompts[cur_val_example_id]}_{post_fix}] ended.")
            
            del generator
            gc.collect()
            torch.cuda.empty_cache()
            
    for tracker in accelerator.trackers:
        for cur_val_example_id in range(num_val_example):
            text_prompt, simple_image, pure_text_image, final_image  = vis_image_dict[cur_val_example_id]
            np_images = np.stack([np.asarray(cv2.resize(np.array(img), (args.resolution, args.resolution))) for img in [pure_text_image, simple_image , final_image]])
            if tracker.name == "tensorboard":
                    tracker.writer.add_images(f"Subject: {{batch_subject_prompt[cur_val_example_id]}}. Target: {batch_text_prompts[cur_val_example_id]} [Vanilla SD2.1 base / Full injection / Generated image]", np_images, epoch, dataformats="NHWC")
            elif tracker.name == "wandb":
                # 创建一个日志字典
                log_dict = {
                    f"Subject: {batch_subject_prompt[cur_val_example_id]}. Target: {batch_text_prompts[cur_val_example_id]} [Vanilla SD2.1 base / Full injection / Generated image]":
                        [wandb.Image(img, caption=f"Epoch {epoch}") for img in np_images]  # 逐张上传
                }
                # 记录到 wandb
                wandb.log(log_dict)
            
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

    # calculate the add noise/inversion steps
    infer_discrete_timesteps, num_inf_steps = retrieve_timesteps(noise_scheduler, args.infer_steps, device, None, None)
    infer_discrete_timesteps = infer_discrete_timesteps.cpu().numpy().tolist()

    noise_step = split_ratio * noise_scheduler.config.num_train_timesteps
    threshold_timestep = min(filter(lambda x: x <= noise_step, infer_discrete_timesteps), key=lambda x: abs(x - noise_step)) # 找到最近且小的值

    # find the index in timesteps and calculate how many are skiped 
    threshold_idx = infer_discrete_timesteps.index(threshold_timestep)
    skipped_steps = threshold_idx + 1


    while ((cur_loop_num < max_num_loop and sim < sim_threshold) or cur_loop_num < min_num_loop):

        noisy_latents = add_noise_to_image(noise_step = threshold_timestep, args=args, img=loop_image, vae=vae, train_transforms=train_transforms, noise_scheduler=noise_scheduler)

            
        with torch.no_grad():
            res = pipeline(
                batch_text_prompt if not args.do_editing else batch_text_prompt,
                # "",
                num_inference_steps=args.infer_steps, 
                # generator=generator,
                subject_features= subject_features,
                image_paths=batch_img_path,
                weight_dtype=weight_dtype,
                train_transforms=train_transforms,
                subject_prompts=batch_subject_prompt,
                args=args,
                latents=noisy_latents,
                latents_steps=skipped_steps,
                guidance_scale = 10,
                threshold_timestep=threshold_timestep,
            )
        loop_image = res["images"][0]
        origin_loop_image = loop_image.resize(initial_image_size)
        origin_loop_image.save(f"{val_image_out_dir}/{batch_text_prompt}_{post_fix}/loop_{cur_loop_num}.png")
        
        sims = 0
        # sim = compute_clip_similarity(clip_model, clip_processor, image1=prev_image, image2=loop_image, device=subject_noise.device)
        # prev_image = loop_image
        
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

    # deepspeed_plugin = DeepSpeedPlugin(hf_ds_config='accelerate_configs/deepspeed_config.json', zero3_init_flag=True)
    accelerator = Accelerator(
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        mixed_precision=args.mixed_precision,
        # deepspeed_plugin=deepspeed_plugin,
        log_with=args.report_to,
        project_config=accelerator_project_config,
        kwargs_handlers=[InitProcessGroupKwargs(backend="nccl", timeout=timedelta(seconds=60*60))]
    )

    # accelerator = Accelerator(log_with=args.report_to, gradient_accumulation_steps=args.gradient_accumulation_steps,mixed_precision=args.mixed_precision,project_config=accelerator_project_config,)
    
    if accelerator.is_local_main_process:
        datasets.utils.logging.set_verbosity_warning()
        transformers.utils.logging.set_verbosity_warning()
        diffusers.utils.logging.set_verbosity_info()
    else:
        datasets.utils.logging.set_verbosity_error()
        transformers.utils.logging.set_verbosity_error()
        diffusers.utils.logging.set_verbosity_error()
    
    return accelerator

def load_models_and_learnable_params(args, device, weight_dtype):
    # load base models
    noise_scheduler = DDPMScheduler.from_pretrained(args.pretrained_model_name_or_path, subfolder="scheduler", local_files_only=True, torch_dtype=weight_dtype)
    # Load the tokenizers
    tokenizer_one = AutoTokenizer.from_pretrained(args.pretrained_model_name_or_path,subfolder="tokenizer",revision=args.revision,use_fast=False, local_files_only=True, torch_dtype=weight_dtype)
    tokenizer_two = AutoTokenizer.from_pretrained(args.pretrained_model_name_or_path, subfolder="tokenizer_2", revision=args.revision, use_fast=False, local_files_only=True, torch_dtype=weight_dtype)
    
    text_encoder_one = CLIPTextModel.from_pretrained(args.pretrained_model_name_or_path, subfolder="text_encoder", revision=args.revision, variant=args.variant, local_files_only=True, torch_dtype=weight_dtype)
    text_encoder_two = CLIPTextModelWithProjection.from_pretrained(args.pretrained_model_name_or_path, subfolder="text_encoder_2", revision=args.revision, variant=args.variant, local_files_only=True, torch_dtype=weight_dtype)
    try:
        vae = AutoencoderKL.from_pretrained(args.vae_path, torch_dtype=torch.float16, local_files_only=True)
    except:
        vae = AutoencoderKL.from_pretrained(args.pretrained_model_name_or_path, subfolder="vae", local_files_only=True, torch_dtype=weight_dtype)

    main_unet = UNet2DConditionModel_main.from_pretrained(args.pretrained_model_name_or_path, subfolder="unet", revision=args.revision, local_files_only=True, torch_dtype=weight_dtype)
    reference_unet = UNet2DConditionModel_ref(args=args).from_pretrained(args.pretrained_model_name_or_path, subfolder="unet", revision=args.revision, local_files_only=True, torch_dtype=weight_dtype)
    
    # clip_model, clip_processor = clip.load("ViT-B/32", device=device)
    clip_model, clip_processor = clip.load(args.clip_path, device=device)
    
    # clip_model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32", local_files_only=True)
    # clip_processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32", local_files_only=True)
    
    # freeze everything first
    vae.requires_grad_(False)
    text_encoder_one.requires_grad_(False)
    text_encoder_two.requires_grad_(False)
    main_unet.requires_grad_(False)
    reference_unet.requires_grad_(False)
    
    # register Adapter to main_unet attention blocks, and also register some configs inside UNets, also optioanlly register trainable parameters, and set those params trainable
    num_of_adapters = register_adapter_and_configs(main_unet, reference_unet, args)
    
    return main_unet, reference_unet, noise_scheduler, tokenizer_one, tokenizer_two, text_encoder_one, text_encoder_two, vae, clip_model, clip_processor, num_of_adapters


def register_adapter_and_configs(main_unet, reference_unet, args):
    all_blocks = nn.ModuleList([])
    all_blocks.extend(main_unet.down_blocks)
    all_blocks.append(main_unet.mid_block)
    all_blocks.extend(main_unet.up_blocks)

    counter = 0
    # register
    for num_down, unet_block in enumerate(all_blocks):
        if hasattr(unet_block, "has_cross_attention") and unet_block.has_cross_attention:
            for num_attn, attn in enumerate(unet_block.attentions):
                for num_basics, basic_attn_processors in enumerate(attn.transformer_blocks):
                    
                    attn_1 = basic_attn_processors.attn1
                    attn_2 = basic_attn_processors.attn2
                    norm_1 = basic_attn_processors.norm1
                    norm_2 = basic_attn_processors.norm2
                    # breakpoint()
                    # extract Adapter parameters
                    self_attention_module_params = extract_attention_params(attn_1)

                    # initialize Adapter
                    adapter = initialize_adapter(self_attention_module_params, args)
                    
                    # obtain norm parameters
                    norm_eps = basic_attn_processors.norm_eps
                    norm_elementwise_affine = basic_attn_processors.norm_elementwise_affine

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
                    basic_attn_processors.args = args
                    basic_attn_processors.adapter = adapter
                    basic_attn_processors.adapter_norm = adapter_norm
                    counter += 1
                
                
    reference_unet.args = args            
    print(f"Registered {counter} adapters!")
    # (Optional) Register some weighting factors for the extracted features and set them trainable
    main_unet.learnable_weights = nn.Parameter(torch.ones(counter)).requires_grad_(True)
    return counter

def load_checkpoint(accelerator, args):
    # load pretrained model
    accelerator.print(f"Resuming from checkpoint {args.checkpoint_path}")
    accelerator.load_state(args.checkpoint_path)
    

def load_pipelines(vae, text_encoder_one, text_encoder_two, tokenizer_one, tokenizer_two, main_unet, noise_scheduler, weight_dtype, args):
    # pipeline = StableDiffusionPipeline_main.from_pretrained(
    #     args.pretrained_model_name_or_path,
    #     vae=vae,
    #     text_encoder=text_encoder,
    #     tokenizer=tokenizer,
    #     unet=main_unet,
    #     safety_checker=None,
    #     revision=args.revision,
    #     variant=args.variant,
    #     torch_dtype=weight_dtype,
    #     local_files_only=True
    # )
    
    pipeline = StableDiffusionXLPipeline_main.from_pretrained(
        args.pretrained_model_name_or_path,
        vae=vae,
        unet=main_unet,
        safety_checker=None,
        revision=args.revision,
        # variant=args.variant,
        torch_dtype=weight_dtype,
        local_files_only=True,
        # variant="fp16"
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

    weight_dtype = torch.float32
    if accelerator.mixed_precision == "fp16":
        weight_dtype = torch.float16
    elif accelerator.mixed_precision == "bf16":
        weight_dtype = torch.bfloat16
    
    # load models, here, only main_UNet would contains trainable parameters while reference_UNet is merely a identical copy of SD2.1-base main_unet used for feature extraction
    main_unet, reference_unet, noise_scheduler, tokenizer_one, tokenizer_two, text_encoder_one, text_encoder_two, vae, clip_model, clip_processor, num_of_adapters = load_models_and_learnable_params(args, accelerator.device, weight_dtype)
    
    # Move text_encode and vae to gpu and cast to weight_dtype
    text_encoder_one.to(accelerator.device, dtype=weight_dtype)
    text_encoder_two.to(accelerator.device, dtype=weight_dtype)
    
    vae.to(accelerator.device, dtype=weight_dtype)
    reference_unet.to(accelerator.device, dtype=weight_dtype)
    
    # get trainable params for optimizer
    trainable_params = get_trainable_params(main_unet)
    trainable_params_name = get_trainable_params_name(main_unet)

    if args.scale_lr:
        args.learning_rate = (args.learning_rate * args.gradient_accumulation_steps * args.train_batch_size * accelerator.num_processes)
        
    if accelerator.mixed_precision == "fp16":
        for params in trainable_params:
            params.to(torch.float16)
        cast_training_params([main_unet], dtype=torch.float16)
        pass
        
    elif accelerator.mixed_precision == "bf16":
        weight_dtype = torch.bfloat16
        
    
    train_transforms = transforms.Compose(
        [
            transforms.Resize((args.resolution, args.resolution), interpolation=transforms.InterpolationMode.BILINEAR),
            transforms.ToTensor(),
            transforms.Normalize([0.5], [0.5]),
        ]
    )
    
    main_unet = accelerator.prepare(main_unet)
    
    
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
    
    # save_path = os.path.join(args.output_dir, "Final")

    # main_unet = accelerator.unwrap_model(main_unet)
    # main_unet.save_pretrained(save_path, torch_dtype=torch.float16, is_main_process=accelerator.is_main_process, max_shard_size='50GB', safe_serialization=False)

    # logger.info(f"Final model saved state to {save_path}")
        
    state_dict = torch.load("/mnt/workspace/workgroup/duanzicheng.dzc/checkpoints/ezigen/train/train_sdxl_staged_timesteps_more_data_fp16/Final/diffusion_pytorch_model.bin")
    main_unet.load_state_dict(state_dict, strict=False)

    if accelerator.is_main_process:
        subject_noise = torch.randn((1, 4, args.resolution // 8, args.resolution // 8), dtype=weight_dtype)
        vis_images = log_validation_batch(vae, text_encoder_one, text_encoder_two, tokenizer_one, tokenizer_two, accelerator.unwrap_model(main_unet), reference_unet, noise_scheduler, subject_noise, train_transforms, args, accelerator, weight_dtype, 0, batch_text_prompts, batch_origin_text_prompts, batch_subject_prompts, batch_img_paths, batch_variation_num, clip_model, clip_processor)

if __name__ == "__main__":
    main(config_path=None, config_file=None)