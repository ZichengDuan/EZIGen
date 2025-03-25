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

from transformers import CLIPTextModel, CLIPTokenizer, CLIPProcessor, CLIPModel, AutoImageProcessor, AutoModel, AutoTokenizer, CLIPTextModelWithProjection
from transformers.utils import logging as transformers_logging
from transformers.utils import ContextManagers
from transformers.utils import logging as hf_logging

import diffusers
from diffusers import (
    AutoencoderKL, DDPMScheduler, DDIMScheduler, StableDiffusionPipeline, UNet2DConditionModel, PNDMScheduler,
    DPMSolverSDEScheduler, DPMSolverMultistepInverseScheduler, DPMSolverMultistepScheduler, EulerDiscreteScheduler, HeunDiscreteScheduler
)
from diffusers.utils import is_xformers_available, check_min_version, deprecate, is_wandb_available, make_image_grid, convert_state_dict_to_diffusers, check_min_version
from diffusers.training_utils import EMAModel, compute_snr
from diffusers.optimization import get_scheduler
from diffusers.utils.testing_utils import enable_full_determinism

from accelerate.utils import ProjectConfiguration, set_seed
from accelerate.state import AcceleratorState
from accelerate.logging import get_logger
from accelerate import Accelerator

from models.inversion_models import InversePipelinePartial, ExceptionCLIPTextModel, partial_inverse
from utils import extract_subject_features_sdxl, add_noise_to_image, calculate_dino_similarity, compute_clip_similarity, resize_image_to_fit_short
from models.main_unet.unet_main import UNet2DConditionModel_main
from models.reference_unet.unet_ref import UNet2DConditionModel_ref
from models.main_unet.adapter import Attention_Adapter  # my model
from models.pipelines.pipline_sd_main import StableDiffusionPipeline_main
from models.pipelines.pipline_sdxl_main import StableDiffusionXLPipeline_main

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


def loop_infer(args, subject_img_paths, subject_features, vae, noise_scheduler, weight_dtype, target_prompt, subject_prompts, train_transforms, generator, init_image, sim_threshold=0.98, pipeline=None, output_root=None, post_fix=None, reference_unet=None, exclip=None, inverse_pipeline=None, main_unet=None, clip_model=None, clip_processor=None, foreground_mask=None, initial_image_size = None, source_image_path=None, inference_attn_mask=None):
    """
    input: 
    """

    sim = 0
    cur_loop_num = 1
    split_ratio = args.split_ratio

    loop_image = init_image
    prev_image = init_image
    original_inversed_intermediate_latents = None

    
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

    # # to obtain backgrounds
    # inversed_noisy_latents, inversed_intermediate_latents = partial_inverse_xl(threshold_timestep, loop_image, inverse_pipeline, save_decoded=False, num_inference_steps=args.infer_steps)

    while ((cur_loop_num < max_num_loop and sim < sim_threshold) or cur_loop_num < min_num_loop):
        
        noisy_latents = add_noise_to_image(noise_step = threshold_timestep, args=args, img=loop_image, vae=vae, train_transforms=train_transforms, noise_scheduler=noise_scheduler)


        if args.do_editing:
            # to obtain backgrounds
            outs = inverse_pipeline(prompt = target_prompt, image = loop_image, num_inference_steps=args.infer_steps, guidance_scale = 1, threshold_timestep=threshold_timestep)
            inversed_noisy_latents, inversed_intermediate_latents = outs['images'], outs['intermediate_inversed_latents']

            # inversed_noisy_latents, inversed_intermediate_latents = partial_inverse_xl(threshold_timestep, loop_image, inverse_pipeline, save_decoded=False, num_inference_steps=args.infer_steps)
            # replaced the to-edit area with noisy latents
            # foreground_mask_edit = F.interpolate(foreground_mask.unsqueeze(0).unsqueeze(0), size=noisy_latents.shape[-2:])
            # noisy_latents = foreground_mask_edit * noisy_latents + (1 - foreground_mask_edit) * inversed_intermediate_latents[-1]

            noisy_latents = inversed_noisy_latents

            if original_inversed_intermediate_latents is None:
                original_inversed_intermediate_latents = inversed_intermediate_latents
            else:
                # only use the latents from the first round
                inversed_intermediate_latents = original_inversed_intermediate_latents
            
            # args.skip_adapter_ratio = 1
            # res_origin = pipeline(target_prompt, 
            #                         num_inference_steps=args.infer_steps,
            #                         generator=generator,
            #                         subject_features = None,
            #                         image_paths= subject_img_paths,
            #                         reference_unet=reference_unet,
            #                         weight_dtype= weight_dtype,
            #                         train_transforms=train_transforms,
            #                         subject_prompt= subject_prompts,
            #                         args=args,
            #                         latents=noisy_latents,
            #                         latents_steps=None,
            #                         negative_prompt="dark, blur, defoucus, lack of content, dizzy.",
            #                         guidance_scale=args.guidance_scale,
            #                         threshold_timestep=threshold_timestep
            #                         ) 
            # pure_text_image = res_origin["images"][0]
            # pure_text_image.save(f"{output_root}/{target_prompt}{post_fix}/recon_image_loop.png")

        args.skip_adapter_ratio = 0
        with torch.no_grad():
            res = pipeline(
                target_prompt, 
                num_inference_steps=args.infer_steps, 
                # generator=generator,
                subject_features= subject_features,
                image_paths=subject_img_paths,
                weight_dtype=weight_dtype,
                train_transforms=train_transforms,
                subject_prompts=subject_prompts,
                args=args,
                latents=noisy_latents,
                latents_steps=skipped_steps,
                foreground_mask=foreground_mask,
                guidance_scale = args.guidance_scale,
                inversed_intermediate_latents=inversed_intermediate_latents if args.do_editing else None,
                threshold_timestep=threshold_timestep,
                inference_attn_mask=inference_attn_mask,
                # negative_prompt="a cowboy riding a white horse."
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



def iteration_wrapper(args, accelerator, subject_img_paths, main_unet, reference_unet, text_encoder_one, text_encoder_two, tokenizer_one, tokenizer_two, vae,  noise_scheduler, weight_dtype, target_prompt, subject_prompts, train_transforms, generator=None, output_root=None, post_fix="", pipeline=None, exclip=None, inverse_pipeline=None, clip_model=None, clip_processor=None, source_image_path=None, foreground_mask_path=None):
    """
    1. load load prompts and ref image features
    2. initial loop, gte the initial image and hard masks (all)
    3. merge masks
    4. Start loop, in each loop:
        i: add noise to the given image
        ii: init pipeline
        iii: provide given noise and give mask
        iv: 
    """
    # load all subject feature
    subject_features  = extract_subject_features_sdxl(args, subject_img_paths, reference_unet, text_encoder_one, text_encoder_two, tokenizer_one, tokenizer_two, vae,  noise_scheduler, None,  weight_dtype, train_transforms, text=subject_prompts, subject_denoise_timestep=args.subject_denoise_timestep, device=reference_unet.device, generator=generator.manual_seed(int(args.seed)), visualize_denoised=False)

    # reshape subject feature for CFG
    if subject_features[0].ndim == 2:
        for i in range(len(subject_features)):
            subject_features = [torch.cat((subject_feature[None, :, :], subject_feature[None, :, :]), dim=0) for subject_feature in subject_features]
    elif subject_features[0].ndim == 3 and subject_features[0].shape[0] == 1:
        subject_features = [torch.vstack((subject_feature, subject_feature)) for subject_feature in subject_features]
    # ==========================================================================
    # # generate a image without iteration
    args.skip_adapter_ratio = 0
    generator.manual_seed(int(args.seed))
    res = pipeline(target_prompt, num_inference_steps=args.infer_steps, generator=generator,
                    subject_features = subject_features, 
                    image_paths= subject_img_paths,
                    reference_unet=reference_unet,
                    weight_dtype= weight_dtype,
                    train_transforms=train_transforms,
                    subject_prompt= subject_prompts,
                    args=args,
                    latents=None,
                    latents_steps=None,
                    negative_prompt="dark, blur, defoucus, lack of content, dizzy.",
                    guidance_scale=args.guidance_scale
                    )
    simple_img = res["images"][0]
    simple_img.save(f"{output_root}/{target_prompt}{post_fix}/simple_img.png")
    # ==========================================================================

    # ==========================================================================
    if args.do_editing and source_image_path:
        initial_image = Image.open(source_image_path).convert('RGB')
        initial_image = resize_image_to_fit_short(initial_image, short_size=512)
        W, H = initial_image.size
        initial_image_resized = initial_image.resize((512, 512), 1)
    else:
        # generate a image with pure text
        generator.manual_seed(int(args.seed))
        args.skip_adapter_ratio = 1
        res_origin = pipeline(target_prompt, 
                                num_inference_steps=args.infer_steps,
                                generator=generator,
                                subject_features = None,
                                image_paths= subject_img_paths,
                                reference_unet=reference_unet,
                                weight_dtype= weight_dtype,
                                train_transforms=train_transforms,
                                subject_prompt= subject_prompts,
                                args=args,
                                latents=None,
                                latents_steps=None,
                                negative_prompt="dark, blur, defoucus, lack of content, dizzy.",
                                guidance_scale=args.guidance_scale
                                ) 

        pure_text_image = res_origin["images"][0]
        pure_text_image.save(f"{output_root}/{target_prompt}{post_fix}/pure_text_image.png")
        generator.manual_seed(int(args.seed))
        if args.init_img_path is None:
            # decouple
            args.skip_adapter_ratio = 1 - args.split_ratio
            res = pipeline(target_prompt, num_inference_steps=args.infer_steps, generator=generator,
                            subject_features = subject_features,
                            image_paths= subject_img_paths,
                            reference_unet=reference_unet,
                            weight_dtype= weight_dtype,
                            train_transforms=train_transforms,
                            subject_prompt= subject_prompts,
                            args=args,
                            latents=None,
                            latents_steps=None,
                            negative_prompt="dark, blur, defoucus, lack of content, dizzy.",
                            guidance_scale=args.guidance_scale
                            ) 

            initial_image = res["images"][0]
        else:
            initial_image = Image.open(args.init_img_path).resize((args.resolution, args.resolution))
            
    initial_image_resized = initial_image
    W, H = initial_image.size

    initial_image.save(f"{output_root}/{target_prompt}{post_fix}/initial_loop.png")
    # ==========================================================================
    # merge masks
    if args.do_editing:
        foreground_mask = Image.open(foreground_mask_path).convert('RGB')
        foreground_mask = resize_image_to_fit_short(foreground_mask, short_size=512)
        foreground_mask = np.array(foreground_mask)[:, :, 0] // 255

        plt.imsave(f"{output_root}/{target_prompt}{post_fix}/mask_foreground.png", foreground_mask * 255)
        foreground_mask = cv2.resize(foreground_mask, (512, 512))
        foreground_mask = torch.tensor(foreground_mask).to(reference_unet.device)
    else:
        foreground_mask = None

    # start interation after decoupling operation
    args.skip_adapter_ratio = 0
    final_image = loop_infer(args, subject_img_paths, subject_features, vae, noise_scheduler, weight_dtype, target_prompt, subject_prompts, train_transforms, generator, initial_image_resized, sim_threshold=args.sim_threshold, pipeline=pipeline, output_root=output_root, post_fix=post_fix, reference_unet=reference_unet, exclip=exclip, inverse_pipeline=inverse_pipeline, main_unet=main_unet, clip_model=clip_model, clip_processor=clip_processor, foreground_mask=foreground_mask, initial_image_size=(W, H), source_image_path=source_image_path)
    return final_image


def load_config_and_args():
    parser = argparse.ArgumentParser(description="Command line argument parser")

    parser.add_argument('--config', type=str, required=True, help="Path to the YAML configuration file")
    parser.add_argument('--target_prompt', type=str, default=None, help="Target prompt string")
    parser.add_argument('--subject_prompts', type=str, default=None, help="Subject prompt string")
    parser.add_argument('--subject_img_paths', type=str, default=None, help="Subject image path")
    parser.add_argument('--init_img_path', type=str, default=None, help="Subject image path")

    # batch infer
    parser.add_argument('--batch_idx', type=int, default=0)
    parser.add_argument('--batch_total_num', type=int, default=-1)
    

    # Subject driven editing
    parser.add_argument('--do_editing', action='store_true')
    parser.add_argument('--foreground_mask_path', type=str, default=None)
    parser.add_argument('--source_image_path', type=str, default=None)
    
    # misc
    parser.add_argument('--split_ratio', type=float, default=0.5)
    parser.add_argument('--infer_steps', type=int, default=50)
    parser.add_argument('--guidance_scale', type=float, default=7.5)
    parser.add_argument('--sim_threshold', type=float, default=0.99)

    # output
    parser.add_argument('--output_root', type=str, default="experiments/debug_sdxl")
    parser.add_argument('--num_interations', type=int, default=-1)

    args = parser.parse_args()

    with open(args.config, 'r') as f:
        config = yaml.safe_load(f)

    for key, value in config.items():
        setattr(args, key, value)

    return args

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


def init_acclerator(args):
    accelerator_project_config = ProjectConfiguration(project_dir=args.output_root)
    accelerator = Accelerator(gradient_accumulation_steps=args.gradient_accumulation_steps,mixed_precision=args.mixed_precision,project_config=accelerator_project_config,)
    return accelerator


def load_models(args):
    # Load models
    # noise_scheduler = EulerDiscreteScheduler.from_pretrained(args.pretrained_model_name_or_path, subfolder="scheduler", local_files_only=True)
    # noise_scheduler = DPMSolverMultistepScheduler.from_pretrained(args.pretrained_model_name_or_path, subfolder="scheduler", solver_order=1, algorithm_type="dpmsolver++", local_files_only=True)
    # noise_scheduler = DPMSolverSDEScheduler.from_pretrained(args.pretrained_model_name_or_path, subfolder="scheduler", local_files_only=True)
    noise_scheduler = DDIMScheduler.from_pretrained(args.pretrained_model_name_or_path, subfolder="scheduler", local_files_only=True)
    # Load the tokenizers
    tokenizer_one = AutoTokenizer.from_pretrained(args.pretrained_model_name_or_path,subfolder="tokenizer",revision=args.revision,use_fast=False, local_files_only=True)
    tokenizer_two = AutoTokenizer.from_pretrained(args.pretrained_model_name_or_path, subfolder="tokenizer_2", revision=args.revision, use_fast=False, local_files_only=True)
    
    text_encoder_one = CLIPTextModel.from_pretrained(args.pretrained_model_name_or_path, subfolder="text_encoder", revision=args.revision, variant=args.variant, local_files_only=True)
    text_encoder_two = CLIPTextModelWithProjection.from_pretrained(args.pretrained_model_name_or_path, subfolder="text_encoder_2", revision=args.revision, variant=args.variant, local_files_only=True)
    
    if args.mixed_precision == "fp16":
        vae = AutoencoderKL.from_pretrained(args.vae_path, torch_dtype=torch.float16, local_files_only=True)
    else:
        vae = AutoencoderKL.from_pretrained(args.pretrained_model_name_or_path, subfolder="vae", local_files_only=True)
    main_unet = UNet2DConditionModel_main.from_pretrained(args.pretrained_model_name_or_path, subfolder="unet", revision=args.revision, local_files_only=True)
    reference_unet = UNet2DConditionModel_ref(args=args).from_pretrained(args.pretrained_model_name_or_path, subfolder="unet", revision=args.revision, local_files_only=True)
    
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

    return main_unet, reference_unet, noise_scheduler, tokenizer_one, tokenizer_two, text_encoder_one, text_encoder_two, vae, clip_model, clip_processor


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
    

def load_checkpoint(accelerator, args):
    # load pretrained model
    accelerator.print(f"Resuming from checkpoint {args.checkpoint_path}")
    accelerator.load_state(args.checkpoint_path)
    

def load_pipelines(vae, main_unet, noise_scheduler, weight_dtype, args):
    pipeline = StableDiffusionXLPipeline_main.from_pretrained(
        args.pretrained_model_name_or_path,
        vae=vae,
        unet=main_unet,
        safety_checker=None,
        revision=args.revision,
        variant=args.variant,
        torch_dtype=weight_dtype,
        local_files_only=True
    )
    pipeline.scheduler = noise_scheduler

    # inversion pipeline
    # inverse_pipeline = InversePipelinePartial.from_pretrained(args.pretrained_model_name_or_path, text_encoder=exclip, local_files_only=True)
    # inverse_pipeline.scheduler = DPMSolverMultistepInverseScheduler.from_config(inverse_pipeline.scheduler.config, local_files_only=True)
    return pipeline, None


def get_sub_batch(data_list, total_batches, current_batch):
    """
    将一个列表分成多个批次，并返回指定批次的子列表。

    参数：
    data_list (list): 要分批的列表
    total_batches (int): 总批次数量
    current_batch (int): 当前批次编号（从1开始）

    返回：
    list: 当前批次的子列表，如果输入不合法则返回空列表
    """
    if total_batches == -1:
        return data_list

    # 检查输入的有效性
    if not isinstance(data_list, list) or total_batches <= 0 or current_batch < 1 or current_batch > total_batches:
        return []

    # 计算每个批次的大小
    total_items = len(data_list)
    batch_size = total_items // total_batches
    remainder = total_items % total_batches

    # 计算当前批次的起始和结束索引
    start_index = (current_batch - 1) * batch_size + min(current_batch - 1, remainder)
    end_index = start_index + batch_size + (1 if current_batch <= remainder else 0)

    # 返回当前批次的子列表
    return data_list[start_index:end_index]


def prompts_n_subjects(batch_idx, batch_total_num):
    # classname, image path, identifier
    lives = [
      ["Chow Chow dog", "example_images/subjects/chowchow.png", "dog2"],
      ["small corgi dog", "example_images/subjects/dog2.png", "dog6"],
      ["gray-white border collie dog", "example_images/subjects/dog8_01.png", "dog8"],
      ["brown cat", "example_images/subjects/cat1.png", "cat1"],
      ["gray cat", "example_images/subjects/cat2.png", "cat2"],
      ["terracotta soldier", "example_images/subjects/terracotta_soldier.jpg", "terracotta_soldier"],
      ["robotic horse", "example_images/subjects/cyber_horse.png", "cyber_horse"],
      ["corgi", "example_images/subjects/dog_hd.png", "corgi_dog"],
    ]
    corgis = [
      ["corgi", "example_images/subjects/dog_hd.png", "corgi_police"],
    ]

    human = [
        ["woman", "example_images/subjects/lifeifei.png", "lifeifei"],
        ["man", "example_images/subjects/lecun.png", "lecun"],
        ["man", "example_images/subjects/000000023_white.png", "celeb"],
        ["man", "example_images/subjects/newton_white.png", "newton"],
        ["man", "example_images/subjects/einstein_white.png", "einstein"],
    ]

    small_objects = [
      ["white bowl", "example_images/subjects/berry_bowl.png", "berry_bowl"],
      ["red backpack", "example_images/subjects/backpack.png", "backpack1"],
      ["gray backpack", "example_images/subjects/dog_backpack.png", "backpack2"],
      ["pink sunglasses", "example_images/subjects/pink_galsses.png", "pink_sunglasses"],
      ["white tall boot", "example_images/subjects/fancy_boot.png", "fancy_boot"],
      ["yellow mug", "example_images/subjects/lego_mug.png", "mug"],
      ["red monster toy", "example_images/subjects/monster.png", "monster_toy"],
    ]

    huge_objects = [
      ["neuschwanstein castle", "example_images/subjects/german_castle.png", "castle"],
      ["modern Audi white sedan car", "example_images/subjects/sedan.png", "car"],
      ["castle", "example_images/subjects/hogwarts_white.png", "car"],
    ]

    lives_prompts = [
        "A {} wearing a sleek leather jacket, leaning casually against a brick wall, exuding an air of effortless cool.",
        "A {} adorned with a stylish floral bowtie, sitting elegantly on a windowsill, its fur glowing softly in the afternoon light.",
        "A {} in a cozy knitted sweater, curled up on a plush armchair by the fireplace, creating a sense of warmth and comfort.",
        "A {} dressed in a lightweight summer dress, strolls gracefully along a tree-lined path, enjoying the gentle breeze.",
        "A {} wearing a tailored suit, adjusting his cufflinks in front of a large mirror, depicting refinement and attention to detail.",
        "A {} in a chic bandana, playfully sprawled across a colorful rug in a bright living room, embodying joy and comfort.",
        "A {} wearing a cute raincoat, sitting patiently at the doorstep, with raindrops glistening on its shiny coat.",
        "A {} casually dressed in athletic wear, taking a moment to stretch in a sun-drenched park, ready for an invigorating run.",
        "A {} in a snug cardigan, sipping a mug of tea on a cozy balcony, enveloped in a serene morning atmosphere.",
        "A {} wearing a stylish beanie, curled up next to a stack of books on a recliner, highlighting a love for literature.",
        "A {} showcasing a fashionable collar, resting peacefully on a vintage suitcase, evoking a sense of adventure and travel.",
        "A {} dressed in a comfortable hoodie, lounging on a beige couch, surrounded by soft cushions and a warm blanket.",
        "A {} in an elegant dress, standing near a vintage piano, reflecting sophistication and grace in a softly lit room.",
        "A {} wearing a patterned scarf, basking in a sunbeam on a polished wooden table, creating an inviting and tranquil scene.",
        "A {} elegantly dressed in a smart blazer, holding a cup of coffee while standing in a bustling café, capturing urban life.",
        "A {} sporting a fashionable sweater, reclining in a sunny spot by the window, creating a peaceful and inviting atmosphere.",
        "A {} standing in a fire storm, surrounded by a bustling city, with a sense of urgency and action.",
        "A {} running in the mist in a magical forest, with a sense of magic and wonder.",
        "A {} standing in front of a neon light, showcasing a futuristic cityscape, with a sense of technology and connectivity.",
    ]

    human_prompts = [
        "A portrait of a {} wearing a sleek leather jacket, leaning casually against a brick wall, exuding an air of effortless cool.",
        "A portrait of a {} adorned with a stylish floral bowtie, sitting elegantly on a windowsill, its fur glowing softly in the afternoon light.",
        "A portrait of a {} in a cozy knitted sweater, curled up on a plush armchair by the fireplace, creating a sense of warmth and comfort.",
        "A portrait of a {} dressed in a lightweight summer dress, strolls gracefully along a tree-lined path, enjoying the gentle breeze.",
        "A portrait of a {} wearing a tailored suit, adjusting his cufflinks in front of a large mirror, depicting refinement and attention to detail.",
        "A portrait of a {} in a chic bandana, playfully sprawled across a colorful rug in a bright living room, embodying joy and comfort.",
        "A portrait of a {} wearing a cute raincoat, sitting patiently at the doorstep, with raindrops glistening on its shiny coat.",
        "A portrait of a {} casually dressed in athletic wear, taking a moment to stretch in a sun-drenched park, ready for an invigorating run.",
        "A portrait of a {} in a snug cardigan, sipping a mug of tea on a cozy balcony, enveloped in a serene morning atmosphere.",
        "A portrait of a {} wearing a stylish beanie, curled up next to a stack of books on a recliner, highlighting a love for literature.",
        "A portrait of a {} showcasing a fashionable collar, resting peacefully on a vintage suitcase, evoking a sense of adventure and travel.",
        "A portrait of a {} dressed in a comfortable hoodie, lounging on a beige couch, surrounded by soft cushions and a warm blanket.",
        "A portrait of a {} in an elegant dress, standing near a vintage piano, reflecting sophistication and grace in a softly lit room.",
        "A portrait of a {} wearing a patterned scarf, basking in a sunbeam on a polished wooden table, creating an inviting and tranquil scene.",
        "A portrait of a {} elegantly dressed in a smart blazer, holding a cup of coffee while standing in a bustling café, capturing urban life.",
        "A portrait of a {} sporting a fashionable sweater, reclining in a sunny spot by the window, creating a peaceful and inviting atmosphere."
    ]

    small_objects_prompts = [
        "A {} resting elegantly on a lavish picnic blanket spread across a vibrant meadow, surrounded by whimsical butterflies and songbirds.",
        "A {} showcased on a polished wooden countertop in a sun-drenched kitchen, its allure complemented by a backdrop of lush, green herbs in pots and the soft sound of a bubbling fountain outside.",
        "A {} gracefully perched on a vintage piano in a charming music room, sunlight streaming through large bay windows adorned with delicate lace curtains.",
        "A {} nestled among radiant sunflowers in a rolling field, with a gentle breeze carrying the sweet scent of blooming wildflowers.",
        "A {} prominently displayed on a grand, intricately carved altar in a serene forest temple, bathed in golden sunlight filtering through the ancient trees.",
        "A {} placed on an ornate coffee table in a luxurious lounge, surrounded by rich fabrics, sparkling glassware, and flickering candles that cast dancing shadows.",
        "A {} resting on a wrought-iron garden table in a quaint courtyard, framed by climbing vines and twinkling fairy lights illuminating the twilight.",
        "A {} suspended from a decorated archway at an outdoor wedding reception, where soft petals fall gently from above, creating a dreamlike atmosphere.",
        "A {} showcased inside a cozy winter cabin with a roaring fireplace, surrounded by wooden beams and shelves filled with rustic decor, creating a warm and inviting ambiance.",
        "A {} set atop a stylish outdoor bar during a vibrant sunset, casting shadows against colorful cocktails and delicious appetizers on the counter.",
        "A {} lying near a serene lake reflecting the moonlight, surrounded by lush greenery and the tranquil sound of water rippling against the shore.",
        "A {} nestled in a charming bookstore nook, where bookshelves curve around a cozy reading chair, sunlight filtering through the window, inviting calm and reflection.",
        "A {} sitting on an elegant spa treatment table, surrounded by aromatic candles and soft towels, with gentle music playing in the background to enhance relaxation.",
        "A {} floating gently in a beautiful pond adorned with lily pads, framed by willows that sway lightly in the breeze, evoking a sense of tranquility.",
        "A {} elegantly placed on a grand staircase in a historic mansion, where intricate details of the railings and walls create an air of sophistication.",
        "A {} displayed against the backdrop of a vibrant art gallery, with colorful abstract artworks illuminating the space and avant-garde installations sparking imagination."
    ]

    huge_objects_prompts = [
        "A majestic {} standing tall against a breathtaking sunset sky, its silhouette framed by vibrant hues of orange and purple.",
        "A grand {} blanketed in pristine snow, reflecting majestically in a serene, mirror-like lake surrounded by towering pines.",
        "An ancient {} looming over a dramatic, stormy landscape, its weathered stones telling tales of time amid swirling clouds.",
        "A breathtaking {} enveloped in a tapestry of colorful wildflowers in full bloom, creating a vivid contrast against the deep blue sky.",
        "A historic {} nestled gracefully in lush greenery, with ivy climbing its ancient walls beside a gently flowing river.",
        "A towering {} bathed in golden light at dusk, the sun casting long shadows that dance across its magnificent facade.",
        "A fairy-tale {} perched on a hill, overlooking a charming village aglow with twinkling lights, creating a scene straight out of a storybook.",
        "A rugged {} with adventurers exploring its rocky trails, the rugged cliffs offering breathtaking views of the surrounding landscape.",
        "A magnificent {} adorned with intricate spires, set under a glimmering starlit sky that enhances its ethereal beauty.",
        "A serene {} enveloped in a soft layer of wispy clouds, with bright blue skies creating an almost dreamlike tranquility.",
        "A sprawling {} by the wild sea, as waves crash against its weathered stones, adding a sense of motion to its enduring majesty.",
        "A dramatic {} with waterfalls cascading down its rocky face into a lush, verdant valley, surrounded by vibrant foliage.",
        "A beautiful {} illuminated by soft, warm lights at night, casting a magical glow that invites exploration and wonder.",
        "A giant {} rising majestically above a peaceful countryside, its presence commanding attention against a backdrop of rolling hills.",
        "A vibrant {} mirrored in a tranquil pond during sunset, the colors swirling together to create a masterpiece of nature's artistry."
    ]

    dog_police_templates = [
        "Photorealistic, real photo, 4k. A charming {} dressed in a police uniform, playfully trotting through a sunny park filled with colorful flowers and green grass, its tail wagging with joy.",
        "Photorealistic, 4k. A courageous {} in a stylish police outfit, sitting proudly on a park bench under a bright blue sky, surrounded by cheerful children playing.",
        "Photorealistic, real photo, 4k. A dapper {} wearing a police suit, patrolling the lush park on clear day, with trees swaying gently in the breeze and butterflies fluttering nearby.",
        "Photorealistic, real photo, 4k. A heroic {} dressed in a neat police uniform, enjoying a wonderful afternoon in the park, basking in the soft sunlight that filters through the leaves.",
        "Photorealistic, real photo, 4k. A delightful {} in a snazzy police outfit, posing confidently in the middle of a vibrant park, with fluffy clouds dotting the azure sky overhead.",
        "Photorealistic, real photo, 4k. An adorable {} clad in a police uniform, curiously exploring a national park, while colorful kites soar high in the clear sky and white clouds all around.",
        "Photorealistic, real photo, 4k. A playful {} in a bright police outfit, frolicking in a national park with children laughing and bright flowers blooming all around.",
        "Photorealistic, real photo, 4k. A courageous {} wearing a police badge, sitting at attention in a park, with birds chirping and trees standing tall under clear partly cloudy sky."
    ]

    seeds = [8721, 1318, 121365, 1982]
    pairs = []

    for small_object in small_objects:
      class_name, img_path, identifier = small_object
      for prompt in small_objects_prompts:
        for seed in seeds:
          pairs.append((prompt.format(class_name), f"A photo of a {class_name}", img_path, identifier, seed))
    
    for huge_object in huge_objects:
      class_name, img_path, identifier = huge_object
      for prompt in huge_objects_prompts:
        for seed in seeds:
          pairs.append((prompt.format(class_name), f"A photo of a {class_name}", img_path, identifier, seed))

    for live in lives:
      class_name, img_path, identifier = live
      for prompt in lives_prompts:
        for seed in seeds:
          pairs.append((prompt.format(class_name), f"A photo of a {class_name}", img_path, identifier, seed))

    for corgi in corgis:
      class_name, img_path, identifier = corgi
      for prompt in dog_police_templates:
        for seed in seeds:
          pairs.append((prompt, f"A photo of a {class_name}", img_path, identifier, seed))

    for hu in human:
      class_name, img_path, identifier = hu
      for prompt in human_prompts:
        for seed in seeds:
          pairs.append((prompt.format(class_name), f"A photo of a {class_name}", img_path, identifier, seed))
    
    for corgi in corgis:
        class_name, img_path, identifier = corgi
        for prompt in dog_police_templates:
            for seed in seeds:
                pairs.append((prompt.format(class_name), f"A photo of a {class_name}", img_path, identifier, seed))

    sub_pairs = get_sub_batch(pairs, batch_total_num, batch_idx)



    return sub_pairs


def main():
    args = load_config_and_args()

    # if args.seed is not None:
    #     generator = torch.Generator(device=device).manual_seed(args.seed)
    #     set_seed(args.seed)
    # else:
    #     generator = None


    # # load accelerator
    # accelerator = init_acclerator(args)

    # weight_dtype = torch.float32
    # if accelerator.mixed_precision == "fp16":
    #     weight_dtype = torch.float16

    # args = parse_args_from_yaml(config_path=config_path, config_file=config_file)
    # set_seed(args.seed)
    
    # load accelerator
    accelerator = init_acclerator(args)

    weight_dtype = torch.float32
    if accelerator.mixed_precision == "fp16":
        weight_dtype = torch.float16
    elif accelerator.mixed_precision == "bf16":
        weight_dtype = torch.bfloat16

    # load models
    main_unet, reference_unet, noise_scheduler, tokenizer_one, tokenizer_two, text_encoder_one, text_encoder_two, vae, clip_model, clip_processor = load_models(args)
    
    # register adapter to attention blocks
    register_adapter_and_configs(main_unet, reference_unet, args)
    
    
    # assign device
    text_encoder_one.to(device, dtype=weight_dtype)
    text_encoder_two.to(device, dtype=weight_dtype)
    vae.to(device, dtype=weight_dtype)
    reference_unet.to(device, dtype=weight_dtype)
    # exclip.to(device, dtype=weight_dtype)
    
    if args.checkpoint_path.endswith(".bin"):
        state_dict = torch.load(args.checkpoint_path, map_location="cpu")
        main_unet.load_state_dict(state_dict, strict=False)
    else:
        # # loading checkpoints
        main_unet = accelerator.prepare(main_unet)
        load_checkpoint(accelerator, args)
        main_unet = accelerator.unwrap_model(main_unet)
    
    # loading pipelines
    pipeline, inverse_pipeline = load_pipelines(accelerator.unwrap_model(vae), main_unet, noise_scheduler, weight_dtype, args)
    # inverse_pipeline.to(device, dtype=weight_dtype)
    pipeline.to(device, dtype=weight_dtype)

    # load batched data
    pairs = prompts_n_subjects(args.batch_idx, args.batch_total_num)[::-1]
    
    original_output_root = args.output_root
    for i, pair in enumerate(pairs):
        print(f"[{i + 1}/{len(pairs)}]")
        args.output_root = original_output_root

        target_prompt, subject_prompt, img_path, identifier, seed = pair

        generator = torch.Generator(device=device).manual_seed(seed)
        set_seed(seed)
        args.seed = seed
        subject_img_paths = img_path
        subject_prompts = subject_prompt
            
        # subject-driven editing
        foreground_mask_path = None
        source_image_path = None
        
        # misc preparation
        train_transforms = transforms.Compose(
            [
                transforms.Resize((args.resolution, args.resolution),interpolation=transforms.InterpolationMode.BILINEAR),
                transforms.ToTensor(),
                transforms.Normalize([0.5], [0.5]),
            ]
        )

        # with torch.autocast("cuda"):
        # I/O operations
        post_fix = ""
        if args.do_editing:
            post_fix += "_editing"
        post_fix += f"_seed_{seed}_{identifier}"

        args.output_root = os.path.join(args.output_root, str(args.batch_idx))

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
                "main_unet": main_unet,
                "reference_unet": reference_unet,
                "text_encoder_one": text_encoder_one,
                "text_encoder_two": text_encoder_two,
                "tokenizer_one": tokenizer_one,
                "tokenizer_two": tokenizer_two,
                "vae": vae,
                "noise_scheduler": noise_scheduler,
                "weight_dtype": weight_dtype,
                "generator": generator,
                "pipeline": pipeline,
                "exclip": None,
                "inverse_pipeline": inverse_pipeline,
                "clip_model": clip_model,
                "clip_processor": clip_processor,
        }

        iteration_wrapper(**kwargs)
    os._exit(0)














if __name__ == "__main__":
    main()
