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

from transformers import CLIPTextModel, CLIPTokenizer, CLIPProcessor, CLIPModel, AutoImageProcessor, AutoModel, AutoTokenizer, CLIPTextModelWithProjection, AutoConfig
from transformers.utils import logging as transformers_logging
from transformers.utils import ContextManagers
from transformers.utils import logging as hf_logging

import diffusers
from diffusers import (
    AutoencoderKL, DDPMScheduler, DDIMScheduler, StableDiffusionPipeline, UNet2DConditionModel, PNDMScheduler,
    DPMSolverSDEScheduler, DPMSolverMultistepInverseScheduler, DPMSolverMultistepScheduler, EulerDiscreteScheduler, HeunDiscreteScheduler,DDIMInverseScheduler
)
from models.inversion_models import InverseDDIMScheduler

from diffusers.utils import is_xformers_available, check_min_version, deprecate, is_wandb_available, make_image_grid, convert_state_dict_to_diffusers, check_min_version
from diffusers.training_utils import EMAModel, compute_snr
from diffusers.optimization import get_scheduler
from diffusers.utils.testing_utils import enable_full_determinism

from accelerate.utils import ProjectConfiguration, set_seed
from accelerate.state import AcceleratorState
from accelerate.logging import get_logger
from accelerate import Accelerator

from models.inversion_models import InversePipelinePartial, InversePipelineXLPartial, ExceptionCLIPTextModel, partial_inverse, partial_inverse_xl, ExceptionCLIPTextModelWithProj
from utils import extract_subject_features_sdxl, add_noise_to_image, calculate_dino_similarity, compute_clip_similarity, resize_image_to_fit_short, create_soft_mask
from models.main_unet.unet_main import UNet2DConditionModel_main
from models.reference_unet.unet_ref import UNet2DConditionModel_ref
from models.main_unet.adapter import Attention_Adapter  # my model
from models.pipelines.pipline_sd_main import StableDiffusionPipeline_main
from models.pipelines.pipline_sdxl_main import StableDiffusionXLPipeline_main
# from models.inversion_models.sdxl_inversions.pnp_pipeline import SDXLDDIMPipeline, SDXLDDIMPipeline_partial


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


def loop_infer(args, subject_img_paths, subject_features, vae, infer_noise_scheduler, weight_dtype, target_prompt, subject_prompts, train_transforms, generator, init_image, sim_threshold=0.98, pipeline=None, output_root=None, post_fix=None, reference_unet=None, exclip=None, inverse_pipeline=None, main_unet=None, clip_model=None, clip_processor=None, foreground_mask=None, initial_image_size = None, source_image_path=None, inference_attn_mask=None):
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
    infer_discrete_timesteps, num_inf_steps = retrieve_timesteps(infer_noise_scheduler, args.infer_steps, device, None, None)
    infer_discrete_timesteps = infer_discrete_timesteps.cpu().numpy().tolist()

    noise_step = split_ratio * infer_noise_scheduler.config.num_train_timesteps
    threshold_timestep = min(filter(lambda x: x <= noise_step, infer_discrete_timesteps), key=lambda x: abs(x - noise_step)) # 找到最近且小的值

    # find the index in timesteps and calculate how many are skiped 
    threshold_idx = infer_discrete_timesteps.index(threshold_timestep)
    skipped_steps = threshold_idx + 1

    # # to obtain backgrounds
    # inversed_noisy_latents, inversed_intermediate_latents = partial_inverse_xl(threshold_timestep, loop_image, inverse_pipeline, save_decoded=False, num_inference_steps=args.infer_steps)

    while ((cur_loop_num < max_num_loop and sim < sim_threshold) or cur_loop_num < min_num_loop):
        
        noisy_latents = add_noise_to_image(noise_step = threshold_timestep, args=args, img=loop_image, vae=vae, train_transforms=train_transforms, noise_scheduler=infer_noise_scheduler)

        if args.do_editing:
            # to obtain backgrounds
            # outs = inverse_pipeline(prompt = target_prompt, image = loop_image, num_inference_steps=args.infer_steps, guidance_scale = 1, threshold_timestep=threshold_timestep)
            # inversed_noisy_latents, inversed_intermediate_latents = outs['latents'], outs['inversed_intermediate_latents']
            inversed_noisy_latents, inversed_intermediate_latents = partial_inverse_xl(threshold_timestep, loop_image, inverse_pipeline, save_decoded=False, num_inference_steps=args.infer_steps)
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


def iteration_wrapper(args, accelerator, subject_img_paths, main_unet, reference_unet, text_encoder_one, text_encoder_two, tokenizer_one, tokenizer_two, vae, infer_noise_scheduler, add_noise_scheduler, weight_dtype, target_prompt, subject_prompts, train_transforms, generator=None, output_root=None, post_fix="", pipeline=None, exclip=None, inverse_pipeline=None, clip_model=None, clip_processor=None, source_image_path=None, foreground_mask_path=None, inference_attn_mask=None):
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
    subject_features  = extract_subject_features_sdxl(args, subject_img_paths, reference_unet, text_encoder_one, text_encoder_two, tokenizer_one, tokenizer_two, vae,  add_noise_scheduler, None,  weight_dtype, train_transforms, text=subject_prompts, subject_denoise_timestep=args.subject_denoise_timestep, device=reference_unet.device, generator=generator.manual_seed(int(args.seed)), visualize_denoised=False)
    # reshape subject feature for CFG, use a dummy input for CFG unconditional batch
    if subject_features[0].ndim == 2:
        for i in range(len(subject_features)):
            subject_features = [torch.cat((subject_feature[None, :, :], subject_feature[None, :, :]), dim=0) for subject_feature in subject_features]
    elif subject_features[0].ndim == 3 and subject_features[0].shape[0] == 1:
        subject_features = [torch.vstack((subject_feature, subject_feature)) for subject_feature in subject_features]
    # # ==========================================================================
    # # generate a image without iteration
    args.skip_adapter_ratio = 0
    generator.manual_seed(int(args.seed))
    # latents = torch.load("outputs/DDIM_inversion_normal_horse_full_step.pt")
    latents = None
    res = pipeline(target_prompt,
                    num_inference_steps=args.infer_steps,
                    generator=generator,
                    subject_features = subject_features, 
                    image_paths= subject_img_paths,
                    reference_unet=reference_unet,
                    weight_dtype= weight_dtype,
                    train_transforms=train_transforms,
                    subject_prompt= subject_prompts,
                    args=args,
                    latents=latents,
                    latents_steps=None,
                    negative_prompt="dark, blur, defoucus, lack of content, dizzy.",
                    guidance_scale=args.guidance_scale,
                    inference_attn_mask=inference_attn_mask
                )
    simple_img = res["images"][0]
    simple_img.save(f"{output_root}/{target_prompt}{post_fix}/simple_img.png")
    # # ==========================================================================
    # breakpoint()
    # ==========================================================================
    if args.do_editing and source_image_path:
        initial_image = Image.open(source_image_path).convert('RGB')
        initial_image = resize_image_to_fit_short(initial_image, short_size=args.resolution)
        W, H = initial_image.size
        initial_image_resized = initial_image.resize((args.resolution, args.resolution), 1)
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
            res = pipeline(target_prompt if not args.do_editing else target_prompt, 
                                num_inference_steps=args.infer_steps, 
                                generator=generator,
                                subject_features = subject_features,
                                image_paths= subject_img_paths,
                                reference_unet=reference_unet,
                                weight_dtype= weight_dtype,
                                train_transforms=train_transforms,
                                subject_prompt= subject_prompts,
                                args=args,
                                latents=latents,
                                latents_steps=None,
                                negative_prompt="dark, blur, defoucus, lack of content, dizzy.",
                                guidance_scale=args.guidance_scale,
                                inference_attn_mask=inference_attn_mask
                                ) 

            initial_image = res["images"][0]
        else:
            initial_image = Image.open(args.init_img_path).resize((args.resolution, args.resolution))
            
    initial_image_resized = initial_image
    W, H = initial_image.size

    initial_image.save(f"{output_root}/{target_prompt}{post_fix}/initial_loop.png")
    # ==========================================================================
    # merge masks
    if args.do_editing and foreground_mask_path is not None:
        foreground_mask = Image.open(foreground_mask_path).convert('RGB')
        foreground_mask = resize_image_to_fit_short(foreground_mask, short_size=args.resolution)

        foreground_mask = create_soft_mask(np.array(foreground_mask), sigma=5)

        foreground_mask = foreground_mask[:, :, 0]

        plt.imsave(f"{output_root}/{target_prompt}{post_fix}/mask_foreground.png", foreground_mask * 255)
        foreground_mask = cv2.resize(foreground_mask, (args.resolution, args.resolution))
        foreground_mask = torch.tensor(foreground_mask).to(reference_unet.device)
    else:
        foreground_mask = None
    
    # start interation after decoupling operation
    args.skip_adapter_ratio = 0
    final_image = loop_infer(args, subject_img_paths, subject_features, vae, infer_noise_scheduler, weight_dtype, target_prompt, subject_prompts, train_transforms, generator, initial_image_resized, sim_threshold=args.sim_threshold, pipeline=pipeline, output_root=output_root, post_fix=post_fix, reference_unet=reference_unet, exclip=exclip, inverse_pipeline=inverse_pipeline, main_unet=main_unet, clip_model=clip_model, clip_processor=clip_processor, foreground_mask=foreground_mask, initial_image_size=(W, H), source_image_path=source_image_path, inference_attn_mask=inference_attn_mask)
    return final_image


def load_config_and_args():
    parser = argparse.ArgumentParser(description="Command line argument parser")

    parser.add_argument('--config', type=str, required=True, help="Path to the YAML configuration file")
    parser.add_argument('--target_prompt', type=str, required=True, help="Target prompt string")
    parser.add_argument('--subject_prompts', type=str, required=True, help="Subject prompt string")
    parser.add_argument('--subject_img_paths', type=str, required=True, help="Subject image path")
    parser.add_argument('--init_img_path', type=str, default=None, help="Subject image path")
    
    # Subject driven editing
    parser.add_argument('--do_editing', action='store_true')
    parser.add_argument('--foreground_mask_path', type=str, default=None)
    parser.add_argument('--source_image_path', type=str, default=None)
    
    # misc
    parser.add_argument('--seed', type=int, default=0)
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
    accelerator_project_config = ProjectConfiguration(project_dir=args.output_dir)

    accelerator = Accelerator(
        mixed_precision=args.mixed_precision,
        project_config=accelerator_project_config,
    )
    return accelerator

def load_models(args, weight_type):
    # Load models
    # noise_scheduler = EulerDiscreteScheduler.from_pretrained(args.pretrained_model_name_or_path, subfolder="scheduler", local_files_only=True)
    # noise_scheduler = DPMSolverMultistepScheduler.from_pretrained(args.pretrained_model_name_or_path, subfolder="scheduler", solver_order=1, algorithm_type="dpmsolver++", local_files_only=True)
    # noise_scheduler = DPMSolverSDEScheduler.from_pretrained(args.pretrained_model_name_or_path, subfolder="scheduler", local_files_only=True)
    if not args.do_editing:
        infer_noise_scheduler = DDPMScheduler.from_pretrained(args.pretrained_model_name_or_path, subfolder="scheduler", local_files_only=True, torch_dtype=weight_type)
        add_noise_scheduler = DDPMScheduler.from_pretrained(args.pretrained_model_name_or_path, subfolder="scheduler", local_files_only=True, torch_dtype=weight_type)
    else:
        infer_noise_scheduler = DDPMScheduler.from_pretrained(args.pretrained_model_name_or_path, subfolder="scheduler", local_files_only=True, torch_dtype=weight_type)
        add_noise_scheduler = DDPMScheduler.from_pretrained(args.pretrained_model_name_or_path, subfolder="scheduler", local_files_only=True, torch_dtype=weight_type)
    
    # Load the tokenizers

    tokenizer_one = AutoTokenizer.from_pretrained(args.pretrained_model_name_or_path,subfolder="tokenizer",revision=args.revision,use_fast=False, local_files_only=True, torch_dtype=weight_type)
    tokenizer_two = AutoTokenizer.from_pretrained(args.pretrained_model_name_or_path, subfolder="tokenizer_2", revision=args.revision, use_fast=False, local_files_only=True, torch_dtype=weight_type)
    
    text_encoder_one = CLIPTextModel.from_pretrained(args.pretrained_model_name_or_path, subfolder="text_encoder", revision=args.revision, variant=args.variant, local_files_only=True, torch_dtype=weight_type)
    text_encoder_two = CLIPTextModelWithProjection.from_pretrained(args.pretrained_model_name_or_path, subfolder="text_encoder_2", revision=args.revision, variant=args.variant, local_files_only=True, torch_dtype=weight_type)
    if weight_type == torch.float16:
        vae = AutoencoderKL.from_pretrained(args.vae_path, torch_dtype=torch.float16, local_files_only=True)
    else:
        vae = AutoencoderKL.from_pretrained(args.pretrained_model_name_or_path, subfolder="vae", local_files_only=True, torch_dtype=weight_type)
    
    main_unet = UNet2DConditionModel_main.from_pretrained(args.pretrained_model_name_or_path, subfolder="unet", revision=args.revision, local_files_only=True, torch_dtype=weight_type)
    reference_unet = UNet2DConditionModel_ref(args=args).from_pretrained(args.pretrained_model_name_or_path, subfolder="unet", revision=args.revision, local_files_only=True, torch_dtype=weight_type) 
    
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

    return main_unet, reference_unet, infer_noise_scheduler, add_noise_scheduler, tokenizer_one, tokenizer_two, text_encoder_one, text_encoder_two, vae, clip_model, clip_processor


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
                
    if reference_unet is not None:
        reference_unet.args = args           
         
    print(f"Registered {counter} adapters!")
    # (Optional) Register some weighting factors for the extracted features and set them trainable
    main_unet.learnable_weights = nn.Parameter(torch.ones(counter)).requires_grad_(True)
    

def load_checkpoint(accelerator, args):
    # load pretrained model
    accelerator.print(f"Resuming from checkpoint {args.checkpoint_path}")
    accelerator.load_state(args.checkpoint_path)
    

def load_pipelines(vae, main_unet, infer_noise_scheduler, weight_dtype, args):
    exclip = ExceptionCLIPTextModel.from_pretrained(args.pretrained_model_name_or_path, subfolder="text_encoder")
    exclip_2 = ExceptionCLIPTextModelWithProj.from_pretrained(args.pretrained_model_name_or_path, subfolder="text_encoder_2")
    
    pipeline = StableDiffusionXLPipeline_main.from_pretrained(
        args.pretrained_model_name_or_path,
        vae=vae,
        unet=main_unet,
        safety_checker=None,
        revision=args.revision,
        variant=args.variant,
        torch_dtype=weight_dtype,
        local_files_only=True,
    )
    # if args.do_editing:
    #     pipeline.text_encoder = exclip
    #     pipeline.text_encoder_2=exclip_2

    pipeline.scheduler = infer_noise_scheduler

    # inversion pipeline
    inverse_pipeline = InversePipelineXLPartial.from_pretrained(args.pretrained_model_name_or_path, vae=vae, local_files_only=True)
    # # inverse_pipeline.scheduler = DPMSolverMultistepInverseScheduler.from_config(inverse_pipeline.scheduler.config, local_files_only=True)
    # inverse_pipeline.scheduler = DDIMInverseScheduler.from_config(inverse_pipeline.scheduler.config)

    # inverse_pipeline = SDXLDDIMPipeline_partial.from_pretrained(args.pretrained_model_name_or_path, use_safetensors=True, torch_dtype=weight_dtype).to("cuda")
    inverse_pipeline.scheduler = DDIMInverseScheduler.from_config(inverse_pipeline.scheduler.config)

    return pipeline, inverse_pipeline


def main():
    args = load_config_and_args()

    if args.seed is not None:
        generator = torch.Generator(device=device).manual_seed(args.seed)
        set_seed(args.seed)
    else:
        generator = None


    # # load accelerator
    # accelerator = init_acclerator(args)

    # weight_dtype = torch.float32
    # if accelerator.mixed_precision == "fp16":
    #     weight_dtype = torch.float16

    # args = parse_args_from_yaml(config_path=config_path, config_file=config_file)
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
    main_unet, reference_unet, infer_noise_scheduler, add_noise_scheduler, tokenizer_one, tokenizer_two, text_encoder_one, text_encoder_two, vae, clip_model, clip_processor = load_models(args, weight_dtype)
    
    # register adapter to attention blocks
    register_adapter_and_configs(main_unet, reference_unet, args)
    
    
    # assign device
    main_unet.to(device, dtype=weight_dtype)
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
    pipeline, inverse_pipeline = load_pipelines(vae, main_unet, infer_noise_scheduler, weight_dtype, args)
    inverse_pipeline.to(device, dtype=weight_dtype)
    pipeline.to(device, dtype=weight_dtype)

    # subject driven generation
    if "|" in args.subject_img_paths:
        subject_img_paths = args.subject_img_paths.split("|")
    else:
        subject_img_paths = args.subject_img_paths
    
    if "|" in args.subject_prompts:
        subject_prompts = args.subject_prompts.split("|")
    else:
        subject_prompts = args.subject_prompts
        
    target_prompt = args.target_prompt
    # subject-driven editing
    foreground_mask_path = args.foreground_mask_path
    source_image_path = args.source_image_path
    
    # misc preparation
    train_transforms = transforms.Compose(
        [
            transforms.Resize((args.resolution, args.resolution),interpolation=transforms.InterpolationMode.BILINEAR),
            transforms.ToTensor(),
            transforms.Normalize([0.5], [0.5]),
        ]
    )

    # make sure subject feature does not interfere with unconditional generation
    # inference_attn_mask = torch.zeros(1, 2, 2 * 2).to(device)
    # inference_attn_mask[0, 0, - 2:] = -50000.0
    # inference_attn_mask = inference_attn_mask.to(weight_dtype)


    # with torch.autocast("cuda"):
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
                "main_unet": main_unet,
                "reference_unet": reference_unet,
                "text_encoder_one": text_encoder_one,
                "text_encoder_two": text_encoder_two,
                "tokenizer_one": tokenizer_one,
                "tokenizer_two": tokenizer_two,
                "vae": vae,
                "infer_noise_scheduler": infer_noise_scheduler,
                "add_noise_scheduler": add_noise_scheduler,
                "weight_dtype": weight_dtype,
                "generator": generator,
                "pipeline": pipeline,
                "exclip": None,
                "inverse_pipeline": inverse_pipeline,
                "clip_model": clip_model,
                "clip_processor": clip_processor,
                # "inference_attn_mask": inference_attn_mask,
    }
    iteration_wrapper(**kwargs)
    os._exit(0)


if __name__ == "__main__":
    main()
