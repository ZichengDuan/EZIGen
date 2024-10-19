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
from numba import cuda
import argparse
import yaml
import shutil
import random
from PIL import Image
import os
import gc
import clip

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

from accelerate.utils import ProjectConfiguration, set_seed
from accelerate.state import AcceleratorState
from accelerate.logging import get_logger
from accelerate import Accelerator

from models.inversion_models import InversePipelinePartial, ExceptionCLIPTextModel, partial_inverse
from models.utils import extract_subject_features, add_noise_to_image, calculate_dino_similarity, compute_clip_similarity, resize_image_to_fit_short
from models.main_unet.unet_main import UNet2DConditionModel_main
from models.reference_unet.unet_ref import UNet2DConditionModel_ref
from models.main_unet.adapter import Attention_Adapter  # my model
from models.pipelines.pipline_sd_main import StableDiffusionPipeline_main

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


def loop_infer(args, subject_img_path, subject_features, vae, noise_scheduler, weight_dtype, target_prompt, subject_prompt, train_transforms, generator, init_image, sim_threshold=0.98, pipeline=None, output_root=None, post_fix=None, reference_unet=None, exclip=None, inverse_pipeline=None, unet=None, clip_model=None, clip_preprocess=None, foreground_mask=None, initial_image_size = None, source_image_path=None):
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

    while ((cur_loop_num < max_num_loop and sim < sim_threshold) or cur_loop_num < min_num_loop):
        if args.do_editing:
            noisy_latents, inversed_intermediate_latents = partial_inverse(split_ratio, loop_image, exclip, inverse_pipeline, unet=None, save_decoded=False, num_inference_steps=args.infer_steps)
            if original_inversed_intermediate_latents is None:
                original_inversed_intermediate_latents = inversed_intermediate_latents
            else:
                # only use the latents from the first round
                inversed_intermediate_latents = original_inversed_intermediate_latents
        else:
            noisy_latents = add_noise_to_image(split_ratio * 1000, generator, args, loop_image, vae, train_transforms, noise_scheduler=noise_scheduler)

        skipped_steps = int(args.infer_steps - split_ratio * args.infer_steps)

        with torch.no_grad():
            res = pipeline(
                target_prompt if not args.do_editing else target_prompt,
                num_inference_steps=args.infer_steps, generator=generator,
                references= subject_features,
                image_paths=subject_img_path,
                weight_dtype=weight_dtype,
                train_transforms=train_transforms,
                ref_text=subject_prompt,
                args=args,
                latents=noisy_latents,
                latents_steps=skipped_steps,
                reference_unet=reference_unet,
                foreground_mask=foreground_mask,
                guidance_scale = 2 if args.do_editing else args.guidance_scale,
                inversed_intermediate_latents=inversed_intermediate_latents if args.do_editing else None
            )
        loop_image = res["images"][0]
        origin_loop_image = loop_image.resize(initial_image_size)
        origin_loop_image.save(f"{output_root}/{target_prompt}{post_fix}/loop_{cur_loop_num}.png")

        sim = compute_clip_similarity(clip_model, clip_preprocess, image1=prev_image, image2=loop_image, device=device)

        prev_image = loop_image
        print(f"[Validation {target_prompt}{post_fix}][Loop {cur_loop_num}] Overall Similarity: {sim}.")

        cur_loop_num += 1

    torch.cuda.empty_cache()
    return loop_image


def iteration_wrapper(args, accelerator, subject_img_path, unet, reference_unet, text_encoder, tokenizer, vae,  noise_scheduler, weight_dtype, target_prompt, subject_prompt, train_transforms, generator=None, output_root=None, post_fix="", pipeline=None, exclip=None, inverse_pipeline=None, clip_model=None, clip_preprocess=None, source_image_path=None, foreground_mask_path=None):
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
    # load all ref unet feature
    subject_features  = extract_subject_features(args, subject_img_path, reference_unet, text_encoder, tokenizer, vae, noise_scheduler, None,  weight_dtype, train_transforms, text=subject_prompt, subject_denoise_timestep=args.subject_denoise_timestep, device=reference_unet.device, generator=generator)

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
                    references = subject_features, 
                    image_paths= subject_img_path,
                    reference_unet=reference_unet,
                    weight_dtype= weight_dtype,
                    train_transforms=train_transforms,
                    ref_text= subject_prompt,
                    args=args,
                    latents=None,
                    latents_steps=None,
                    # negative_prompt='Blur, dizzy, defocus',
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
        res_origin = pipeline(target_prompt, num_inference_steps=args.infer_steps, generator=generator,
                                references = None,
                                image_paths= subject_img_path,
                                reference_unet=reference_unet,
                                weight_dtype= weight_dtype,
                                train_transforms=train_transforms,
                                ref_text= subject_prompt,
                                args=args,
                                latents=None,
                                latents_steps=None,
                                foreground_mask= None,
                                guidance_scale=args.guidance_scale
                                ) 

        pure_text_image = res_origin["images"][0]
        pure_text_image.save(f"{output_root}/{target_prompt}{post_fix}/pure_text_image.png")
        generator.manual_seed(int(args.seed))

        # decouple
        args.skip_adapter_ratio = 1 - args.split_ratio
        res = pipeline(target_prompt, num_inference_steps=args.infer_steps, generator=generator,
                        references = subject_features,
                        image_paths= subject_img_path,
                        reference_unet=reference_unet,
                        weight_dtype= weight_dtype,
                        train_transforms=train_transforms,
                        ref_text= subject_prompt,
                        args=args,
                        latents=None,
                        latents_steps=None,
                        foreground_mask= None,
                        guidance_scale=args.guidance_scale
                        ) 

        initial_image = res["images"][0]

    initial_image_resized = initial_image
    W, H = initial_image.size

    initial_image.save(f"{output_root}/{target_prompt}{post_fix}/initial_loop.png")
    # ==========================================================================
    # merge maskds
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
    final_image = loop_infer(args, subject_img_path, subject_features, vae, noise_scheduler, weight_dtype, target_prompt, subject_prompt, train_transforms, generator, initial_image_resized, sim_threshold=args.sim_threshold, pipeline=pipeline, output_root=output_root, post_fix=post_fix, reference_unet=reference_unet, exclip=exclip, inverse_pipeline=inverse_pipeline, unet=unet, clip_model=clip_model, clip_preprocess=clip_preprocess, foreground_mask=foreground_mask, initial_image_size=(W, H), source_image_path=source_image_path)

    return final_image


def load_config_and_args():
    parser = argparse.ArgumentParser(description="Command line argument parser")

    parser.add_argument('--config', type=str, required=True, help="Path to the YAML configuration file")
    parser.add_argument('--tar_prompt', type=str, required=True, help="Target prompt string")
    parser.add_argument('--sub_prompt', type=str, required=True, help="Subject prompt string")
    parser.add_argument('--sub_img_path', type=str, required=True, help="Subject image path")
    
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
    parser.add_argument('--output_root', type=str, default="experiments/debug_gradio")
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
    noise_scheduler = DDPMScheduler.from_pretrained(args.pretrained_model_name_or_path, subfolder="scheduler", local_files_only=True)
    tokenizer = CLIPTokenizer.from_pretrained(args.pretrained_model_name_or_path, subfolder="tokenizer")
    text_encoder = CLIPTextModel.from_pretrained(args.pretrained_model_name_or_path, subfolder="text_encoder", revision=args.revision, local_files_only=True)
    vae = AutoencoderKL.from_pretrained(args.pretrained_model_name_or_path, subfolder="vae", revision=args.revision, local_files_only=True)
    unet = UNet2DConditionModel_main.from_pretrained(args.pretrained_model_name_or_path, subfolder="unet", revision=args.non_ema_revision, local_files_only=True)
    reference_unet = UNet2DConditionModel_ref(args=args).from_pretrained(args.pretrained_model_name_or_path, subfolder="unet", revision=args.non_ema_revision, local_files_only=True)
    exclip = ExceptionCLIPTextModel.from_pretrained("stabilityai/stable-diffusion-2-1-base", subfolder="text_encoder", local_files_only=True)
    clip_model, clip_preprocess = load_clip_model(device)

    return noise_scheduler, tokenizer, text_encoder, vae, unet, reference_unet, exclip, clip_model, clip_preprocess


def register_adapter(unet, reference_unet, args):
    all_blocks = nn.ModuleList([])
    all_blocks.extend(unet.down_blocks)
    all_blocks.append(unet.mid_block)
    all_blocks.extend(unet.up_blocks)

    for num_down, unet_block in enumerate(all_blocks):
        if hasattr(unet_block, "has_cross_attention") and unet_block.has_cross_attention:
            for num_attn, attn in enumerate(unet_block.attentions):
                attn_1 = attn.transformer_blocks[0].attn1
                attn_2 = attn.transformer_blocks[0].attn2
                norm_1 = attn.transformer_blocks[0].norm1
                norm_2 = attn.transformer_blocks[0].norm2

                # extract Adapter parameters
                self_attention_module_params = extract_attention_params(attn_1)

                # initialize adapter
                adapter = initialize_adapter(self_attention_module_params, args)

                # obtain norm parameters
                norm_eps = unet_block.attentions[num_attn].transformer_blocks[0].norm_eps
                norm_elementwise_affine = unet_block.attentions[num_attn].transformer_blocks[0].norm_elementwise_affine

                # initialize norm
                adapter_norm = nn.LayerNorm(self_attention_module_params['query_dim'], elementwise_affine=norm_elementwise_affine, eps=norm_eps)

                # init from text cross block
                copy_matched_parameters(attn_1, adapter)
                copy_matched_parameters(norm_1, adapter_norm)

                adapter.args = args
                attn_1.args = args
                attn_2.args = args
                attn.transformer_blocks[0].args = args
                attn.transformer_blocks[0].adapter = adapter
                attn.transformer_blocks[0].adapter_norm = adapter_norm

    reference_unet_learnable_weights = nn.Parameter(torch.ones(16))
    reference_unet.learnable_weights = reference_unet_learnable_weights
    reference_unet.args = args


def load_checkpoint(accelerator, args):
    # load pretrained model
    accelerator.print(f"Resuming from checkpoint {args.checkpoint_path}")
    accelerator.load_state(args.checkpoint_path)
    

def load_pipelines(vae, text_encoder, tokenizer, unet, noise_scheduler, exclip, weight_dtype, args):
    pipeline = StableDiffusionPipeline_main.from_pretrained(
        args.pretrained_model_name_or_path,
        vae=vae,
        text_encoder=text_encoder,
        tokenizer=tokenizer,
        unet=unet,
        safety_checker=None,
        revision=args.revision,
        variant=args.variant,
        torch_dtype=weight_dtype,
        local_files_only=True
    )
    pipeline.scheduler = noise_scheduler

    # inversion pipeline
    inverse_pipeline = InversePipelinePartial.from_pretrained(args.pretrained_model_name_or_path, text_encoder=exclip, local_files_only=True)
    inverse_pipeline.scheduler = DPMSolverMultistepInverseScheduler.from_config(inverse_pipeline.scheduler.config, local_files_only=True)
    return pipeline, inverse_pipeline




def main():
    # prepare enironment and spaces
    weight_dtype = torch.float32
    args = load_config_and_args()
    os.makedirs(args.output_root, exist_ok=True)
    if args.seed is not None:
        generator = torch.Generator(device=device).manual_seed(args.seed)
        set_seed(args.seed)
    else:
        generator = None

    # load accelerator
    accelerator = init_acclerator(args)

    # load models
    noise_scheduler, tokenizer, text_encoder, vae, unet, reference_unet, exclip, clip_model, clip_preprocess = load_models(args)

    # register adapter to attention blocks
    register_adapter(unet, reference_unet, args)
    
    # prepare models using accelerator
    unet, reference_unet = accelerator.prepare(unet, reference_unet)
    
    # assign device
    text_encoder.to(device, dtype=weight_dtype)
    vae.to(device, dtype=weight_dtype)
    reference_unet.to(device, dtype=weight_dtype)
    exclip.to(device, dtype=weight_dtype)
    
    # loading checkpoints
    load_checkpoint(accelerator, args)

    # loading pipelines
    pipeline, inverse_pipeline = load_pipelines(accelerator.unwrap_model(vae), accelerator.unwrap_model(text_encoder), tokenizer, accelerator.unwrap_model(unet), noise_scheduler, exclip, weight_dtype, args)
    inverse_pipeline.to(device, dtype=weight_dtype)
    pipeline.to(device, dtype=weight_dtype)

    # subject driven generation
    subject_img_path = args.sub_img_path
    subject_prompt = args.sub_prompt
    target_prompt = args.tar_prompt

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

    with torch.autocast("cuda"):
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
            "subject_img_path": subject_img_path,
            "target_prompt": target_prompt,
            "subject_prompt": subject_prompt,
            "train_transforms": train_transforms,
            "source_image_path": source_image_path,
            "foreground_mask_path": foreground_mask_path,
            "output_root": args.output_root,

            # system
            "post_fix": post_fix,
            "args": args,
            "accelerator": accelerator,
            "unet": unet,
            "reference_unet": reference_unet,
            "text_encoder": text_encoder,
            "tokenizer": tokenizer,
            "vae": vae,
            "noise_scheduler": noise_scheduler,
            "weight_dtype": weight_dtype,
            "generator": generator,
            "pipeline": pipeline,
            "exclip": exclip,
            "inverse_pipeline": inverse_pipeline,
            "clip_model": clip_model,
            "clip_preprocess": clip_preprocess,
        }
        
        iteration_wrapper(**kwargs)















if __name__ == "__main__":
    main()
