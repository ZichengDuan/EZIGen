import sys
sys.path.append("..")
sys.path.append(".")
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
from diffusers.image_processor import VaeImageProcessor
from diffusers.pipelines.stable_diffusion.pipeline_stable_diffusion import retrieve_timesteps
import torch.nn as nn
from datetime import timedelta
from peft import LoraConfig, get_peft_model
from peft.tuners.lora import LoraLayer
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
from diffusers.training_utils import compute_density_for_timestep_sampling, compute_loss_weighting_for_sd3
from transformers import CLIPTextModel, CLIPTokenizer, CLIPProcessor, CLIPModel, AutoImageProcessor, AutoModel, AutoTokenizer, CLIPTextModelWithProjection, T5EncoderModel, T5TokenizerFast
from transformers.utils import logging as transformers_logging
from transformers.utils import ContextManagers
from transformers.utils import logging as hf_logging
import wandb
import diffusers
from diffusers import (
    AutoencoderKL, DDPMScheduler, DDIMScheduler, StableDiffusionPipeline, UNet2DConditionModel, 
    DPMSolverSDEScheduler, DPMSolverMultistepInverseScheduler
)
from models import FluxTransformer2DModel, FlowMatchEulerDiscreteScheduler

from diffusers.utils import is_xformers_available, check_min_version, deprecate, is_wandb_available, make_image_grid, convert_state_dict_to_diffusers, check_min_version
from diffusers.training_utils import EMAModel, compute_snr, cast_training_params
from diffusers.optimization import get_scheduler
from diffusers.utils.testing_utils import enable_full_determinism
from diffusers.utils.torch_utils import is_compiled_module

from accelerate.utils import ProjectConfiguration, set_seed
from accelerate.state import AcceleratorState
from accelerate.logging import get_logger
from accelerate import Accelerator, InitProcessGroupKwargs

from mydatasets import DatasetCOCO_sdxl, Subject200k_dataset_sdxl, Subject200k_dataset_sdxl_jigsaw_sdxl, Subject200k_dataset_parquet_collection2
from mydatasets.datasets_anydoor import YoutubeVISDataset_unet_sdxl, VitonHDDataset_unet

from models.inversion_models import InversePipelinePartial, ExceptionCLIPTextModel, partial_inverse
# from models.flux_transformer.unet_main import UNet2DConditionModel_main
from models.reference_unet.unet_ref import UNet2DConditionModel_ref
from models.main_unet.adapter import Attention_Adapter  # my model
from models.pipelines.pipline_sd_main import StableDiffusionPipeline_main
from models.pipelines.pipline_sdxl_main import StableDiffusionXLPipeline_main
from models.pipelines import FluxPipeline_main

from utils import extract_subject_features, extract_subject_features_sdxl, add_noise_to_image, calculate_dino_similarity, compute_clip_similarity, resize_image_to_fit_short, random_based_on_time, find_subsequence, prepare_mean_masks_each_word, generate_attn_masks_for_each_block, fill_bounding_rect, expand_foreground_hard, expand_foreground_soft, get_sigmas, encode_prompt, add_noise_to_image_flux, prepare_latents, pack_latents, unpack_latents, compute_text_embeddings, calculate_shift

from accelerate.utils import DeepSpeedPlugin

transformers_logging.set_verbosity_error()
warnings.filterwarnings("ignore", category=FutureWarning, module="diffusers")
warnings.filterwarnings("ignore", category=FutureWarning, module="transformers")


# Will error if the minimal version of diffusers is not installed. Remove at your own risks.
check_min_version("0.27.0.dev0")

logger = get_logger(__name__, log_level="INFO")

device = "cpu" if not torch.cuda.is_available() else "cuda"

os.environ["CUDA_LAUNCH_BLOCKING"] = "1"

def get_gpu_usage(device):
    allocated = torch.cuda.memory_allocated(device) / (1024 ** 3)
    total = torch.cuda.get_device_properties(device).total_memory / (1024 ** 3)
    return f"[{allocated:.1f}G/{total:.1f}G]"

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


def iteration_wrapper(args, accelerator, batch_img_path, pipeline_modules, flux_transformer_copy, noise_scheduler_copy, subject_noise, weight_dtype, epoch, batch_text_prompt, batch_origin_text_prompt, batch_subject_prompt, train_transforms, generator, vis_image_dict, val_prompt_idx=None, val_image_out_dir=None, post_fix="", pipeline=None, clip_model=None, clip_processor=None):
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
    # subject_features = extract_subject_features_sdxl(args, batch_img_path, reference_unet, text_encoder_one, text_encoder_two, tokenizer_one, tokenizer_two, vae,  noise_scheduler, None,  weight_dtype, train_transforms, text=batch_subject_prompt, subject_denoise_timestep=args.subject_timestep, device=reference_unet.device, generator=generator.manual_seed(int(args.seed)))

    ### Get subject feature
    # read subject image
    
    subject_image = Image.open(batch_img_path[0]).convert('RGB')
    subject_image = train_transforms(subject_image)

    noise_step =args.subject_timestep
    noisy_subject_latents = add_noise_to_image_flux(subject_image.unsqueeze(0), pipeline_modules["vae"], noise_step, noise_scheduler_copy, noise=subject_noise)

    # get subject text emb
    subject_prompt_embeds, subject_pooled_prompt_embeds, subject_text_ids = compute_text_embeddings(batch_text_prompt, [pipeline_modules["text_encoder"], pipeline_modules["text_encoder_2"]], [pipeline_modules["tokenizer"], pipeline_modules["tokenizer_2"]])

    latent_image_ids = prepare_latents(8, 1, args.resolution, args.resolution, weight_dtype, accelerator.device)
    noisy_subject_latents = pack_latents(noisy_subject_latents, 1, 16, int(args.resolution/8), int(args.resolution/8))

    guidance = torch.full([1], 3.5, device=accelerator.device, dtype=weight_dtype)
    guidance = guidance.expand(noisy_subject_latents.shape[0])

    # with torch.no_grad():
    # extract subject image features from flux
    with torch.no_grad():
        _, subject_features = pipeline_modules["transformer"](
                hidden_states=noisy_subject_latents,
                # YiYi notes: divide it by 1000 for now because we scale it by 1000 in the transforme rmodel (we should not keep it but I want to keep the inputs same for the model for testing)
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
    # subject_features = [subject_feature[:, -noisy_subject_latents.shape[1]:, :].to(weight_dtype) for subject_feature in subject_features]
    # # reshape subject features
    # if subject_features[0].ndim == 2:
    #     for i in range(len(subject_features)):
    #         subject_features = [torch.cat((reference[None, :, :], reference[None, :, :]), dim=0) for reference in subject_features]
    # elif subject_features[0].ndim == 3 and subject_features[0].shape[0] == 1:
    #     subject_features = [torch.vstack((reference, reference)) for reference in subject_features]
    # subject_features = None
    # # inject subject features in in all timesteps, namely the full injection in the ablation study
    args.initial_loop = False
    args.skip_adapter_ratio = 0
    with torch.no_grad():
        res = pipeline(
            batch_text_prompt, 
            num_inference_steps=args.infer_steps, 
            generator=generator.manual_seed(int(args.seed)), 
            subject_features = subject_features, 
            weight_dtype = weight_dtype,
            args=args,
            height=args.resolution,
            width=args.resolution,
            guidance_scale = 3.5,
            is_simple=True
        )
        simple_img = res["images"][0]
        simple_img.save(f"{val_image_out_dir}/{batch_text_prompt}_{post_fix}/simple_img.png")
        vis_image_dict[val_prompt_idx].append(simple_img)
        # # ==========================================================================
        
        # # ==========================================================================
        # generate using vanilla Stable Diffusion 2.1 base,
        # i.e. totally drop the subject guidance and the adapter, and so-called "pure_text_image"
        res_origin = pipeline(
            batch_text_prompt, 
            num_inference_steps=args.infer_steps, 
            generator=generator.manual_seed(int(args.seed)), 
            subject_features = None, 
            weight_dtype = weight_dtype,
            args=args,
            guidance_scale = 3.5,
            height=args.resolution,
            width=args.resolution,
            is_pure_text=True
        ) 
        pure_text_image = res_origin["images"][0]
        pure_text_image.save(f"{val_image_out_dir}/{batch_text_prompt}_{post_fix}/pure_text_image.png")
        vis_image_dict[val_prompt_idx].append(pure_text_image)
        
        # calculate the add noise/inversion steps

        sigmas = np.linspace(1.0, 1 / args.infer_steps, args.infer_steps)
        image_seq_len = noisy_subject_latents[0].shape[1]
        mu = calculate_shift(
            image_seq_len,
            pipeline_modules["scheduler"].config.get("base_image_seq_len", 256),
            pipeline_modules["scheduler"].config.get("max_image_seq_len", 4096),
            pipeline_modules["scheduler"].config.get("base_shift", 0.5),
            pipeline_modules["scheduler"].config.get("max_shift", 1.15),
        )

        infer_discrete_timesteps, num_inf_steps = retrieve_timesteps(pipeline_modules["scheduler"], args.infer_steps, accelerator.device, sigmas=sigmas, mu=mu)

        infer_discrete_timesteps = infer_discrete_timesteps.cpu().numpy().tolist()

        noise_step = args.split_ratio * pipeline_modules["scheduler"].config.num_train_timesteps
        threshold_timestep = min(filter(lambda x: x <= noise_step, infer_discrete_timesteps), key=lambda x: abs(x - noise_step)) # 找到最近且小的值
        
        # find the index in timesteps and calculate how many are skiped 
        threshold_idx = infer_discrete_timesteps.index(threshold_timestep)
        skipped_steps = threshold_idx + 1

        if args.init_image_path is None:
            # generate the sketch image and do the first iteration
            # Note that we combine the sketch image generation and the first iteration together for
            #   simple implemetation.
            args.initial_loop = True
            res = pipeline(
                batch_text_prompt, 
                num_inference_steps=args.infer_steps, 
                generator=generator.manual_seed(int(args.seed)), 
                subject_features = subject_features, 
                weight_dtype = weight_dtype,
                args=args,
                guidance_scale = 3.5,
                threshold_timestep=threshold_timestep,
                height=args.resolution,
                width=args.resolution,
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

    final_image = loop_infer(args, batch_img_path, subject_features, pipeline_modules["vae"], pipeline_modules["scheduler"], noise_scheduler_copy, subject_noise, weight_dtype, batch_text_prompt, batch_subject_prompt, train_transforms, initial_image_resized, generator, sim_threshold=0.99, pipeline=pipeline, val_image_out_dir=val_image_out_dir, post_fix=post_fix, clip_model=clip_model, clip_processor=clip_processor, initial_image_size=(W, H), threshold_timestep=threshold_timestep)
    
    vis_image_dict[val_prompt_idx].append(final_image)
    
    return final_image, vis_image_dict
    
@torch.no_grad()
def log_validation_batch(pipeline_modules, flux_transformer_copy, noise_scheduler_copy, subject_noise, train_transforms, args, accelerator, weight_dtype, epoch, batch_text_prompts, batch_origin_text_prompts, 
batch_subject_prompt, batch_img_paths, batch_variation_num, clip_model, clip_processor):
    logger.info("Running validation... ")
    num_val_example = len(batch_text_prompts)
    
    if len(batch_origin_text_prompts) == 0:
        batch_origin_text_prompts = batch_text_prompts
    pipeline = load_pipelines(pipeline_modules, flux_transformer_copy, weight_dtype, args)
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
            
            final_image, vis_image_dict = iteration_wrapper(args, accelerator, batch_img_paths[cur_val_example_id], pipeline_modules, flux_transformer_copy,noise_scheduler_copy, subject_noise, weight_dtype, epoch, batch_text_prompts[cur_val_example_id], batch_origin_text_prompts[cur_val_example_id], batch_subject_prompt[cur_val_example_id], train_transforms, generator, vis_image_dict, val_prompt_idx=cur_val_example_id, val_image_out_dir = val_image_out_dir, post_fix=post_fix,pipeline=pipeline, clip_model=clip_model, clip_processor=clip_processor)

            print(f"[Validation {batch_text_prompts[cur_val_example_id]}_{post_fix}] ended.")
            
            del generator
            gc.collect()
            torch.cuda.empty_cache()
            
    # for tracker in accelerator.trackers:
    #     for cur_val_example_id in range(num_val_example):
    #         text_prompt, simple_image, pure_text_image, final_image  = vis_image_dict[cur_val_example_id]
    #         np_images = np.stack([np.asarray(cv2.resize(np.array(img), (args.resolution, args.resolution))) for img in [pure_text_image, simple_image , final_image]])
    #         if tracker.name == "tensorboard":
    #                 tracker.writer.add_images(f"Subject: {{batch_subject_prompt[cur_val_example_id]}}. Target: {batch_text_prompts[cur_val_example_id]} [Vanilla SD2.1 base / Full injection / Generated image]", np_images, epoch, dataformats="NHWC")
    #         elif tracker.name == "wandb":
    #             # 创建一个日志字典
    #             log_dict = {
    #                 f"Subject: {batch_subject_prompt[cur_val_example_id]}. Target: {batch_text_prompts[cur_val_example_id]} [Vanilla SD2.1 base / Full injection / Generated image]":
    #                     [wandb.Image(img, caption=f"Epoch {epoch}") for img in np_images]  # 逐张上传
    #             }
    #             # 记录到 wandb
    #             wandb.log(log_dict)
            
    #         else:
    #             logger.warn(f"image logging not implemented for {tracker.name}")

    
    del pipeline
    torch.cuda.empty_cache()
    return vis_image_dict


def loop_infer(args, batch_img_path, subject_features, vae, noise_scheduler, noise_scheduler_copy, subject_noise, weight_dtype, batch_text_prompt, batch_subject_prompt, train_transforms, init_image, generator, threshold_timestep, sim_threshold=0.98, pipeline=None, val_image_out_dir=None, post_fix=None,clip_model=None, clip_processor=None, initial_image_size = None):
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

    # # calculate the add noise/inversion steps
    # infer_discrete_timesteps, num_inf_steps = retrieve_timesteps(noise_scheduler, args.infer_steps, device, None, None)
    # infer_discrete_timesteps = infer_discrete_timesteps.cpu().numpy().tolist()

    # noise_step = split_ratio * noise_scheduler.config.num_train_timesteps
    # threshold_timestep = min(filter(lambda x: x <= noise_step, infer_discrete_timesteps), key=lambda x: abs(x - noise_step)) # 找到最近且小的值

    # # find the index in timesteps and calculate how many are skiped 
    # threshold_idx = infer_discrete_timesteps.index(threshold_timestep)
    # skipped_steps = threshold_idx + 1

    while ((cur_loop_num < max_num_loop and sim < sim_threshold) or cur_loop_num < min_num_loop):
        noisy_latents = add_noise_to_image_flux(img=loop_image, vae=vae, noise_step = threshold_timestep, noise_scheduler=noise_scheduler_copy , train_transforms=train_transforms)
        noisy_latents = pack_latents(noisy_latents, 1, 16, int(args.resolution/8), int(args.resolution/8))

        with torch.no_grad():
            res = pipeline(
                batch_text_prompt if not args.do_editing else batch_text_prompt,
                num_inference_steps=args.infer_steps, 
                subject_features = subject_features,
                weight_dtype=weight_dtype,
                args=args,
                latents=noisy_latents,
                guidance_scale = 3.5,
                threshold_timestep=threshold_timestep,
                height=args.resolution,
                width=args.resolution,
                generator=generator.manual_seed(int(args.seed))
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


def init_accelerator(args):
    logging_dir = os.path.join(args.output_dir, args.logging_dir)
    accelerator_project_config = ProjectConfiguration(project_dir=args.output_dir, logging_dir=logging_dir)
    
    deepspeed_plugin = DeepSpeedPlugin(hf_ds_config=args.deepspeed_config_path, zero3_init_flag=False)
    deepspeed_plugin.deepspeed_config['train_micro_batch_size_per_gpu'] = args.train_batch_size
    
    # deepspeed_plugin = DeepSpeedPlugin(hf_ds_config='accelerate_configs/deepspeed_config.json', zero3_init_flag=True)
    accelerator = Accelerator(
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        mixed_precision=args.mixed_precision,
        deepspeed_plugin=deepspeed_plugin,
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
    noise_scheduler = FlowMatchEulerDiscreteScheduler.from_pretrained(args.pretrained_model_name_or_path, subfolder="scheduler", local_files_only=True)
    noise_scheduler_copy = copy.deepcopy(noise_scheduler)
    # Load the tokenizers
    tokenizer_one = CLIPTokenizer.from_pretrained(args.pretrained_model_name_or_path,subfolder="tokenizer",revision=args.revision,use_fast=False, local_files_only=True, torch_dtype=weight_dtype)
    tokenizer_two = T5TokenizerFast.from_pretrained(args.pretrained_model_name_or_path, subfolder="tokenizer_2", revision=args.revision, use_fast=False, local_files_only=True, torch_dtype=weight_dtype)
    text_encoder_one = CLIPTextModel.from_pretrained(args.pretrained_model_name_or_path, subfolder="text_encoder", revision=args.revision, variant=args.variant, local_files_only=True, torch_dtype=weight_dtype).to(device)
    text_encoder_two = T5EncoderModel.from_pretrained(args.pretrained_model_name_or_path, subfolder="text_encoder_2", revision=args.revision, variant=args.variant, local_files_only=True, torch_dtype=weight_dtype).to(device)

    vae = AutoencoderKL.from_pretrained(args.pretrained_model_name_or_path, subfolder="vae", revision=args.revision, variant=args.variant, local_files_only=True, torch_dtype=weight_dtype).to(device)
    
    flux_transformer = FluxTransformer2DModel.from_pretrained(args.pretrained_model_name_or_path, subfolder="transformer", torch_dtype=weight_dtype, local_files_only=True, ).to(device)
    
    # clip_model, clip_processor = clip.load("ViT-B/32", device=device)
    clip_model, clip_processor = clip.load(args.clip_path, device=device)
    
    # clip_model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32", local_files_only=True)
    # clip_processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32", local_files_only=True)
    
    # freeze everything first
    vae.requires_grad_(False)
    text_encoder_one.requires_grad_(False)
    text_encoder_two.requires_grad_(False)
    flux_transformer.requires_grad_(False)
    
    # register Adapter to flux_transformer attention blocks, and also register some configs inside UNets, also optioanlly register trainable parameters, and set those params trainable
    # num_of_adapters = register_adapter_and_configs(flux_transformer, reference_unet, args)
    num_of_adapters = 0
    return flux_transformer, noise_scheduler_copy, noise_scheduler, tokenizer_one, tokenizer_two, text_encoder_one, text_encoder_two, vae, clip_model, clip_processor, num_of_adapters


def register_adapter_and_configs(flux_transformer, reference_unet, args):
    all_blocks = nn.ModuleList([])

    try:
        assert args.only_up == True
        all_blocks.extend(flux_transformer.up_blocks)
    except:
        all_blocks.extend(flux_transformer.down_blocks)
        all_blocks.append(flux_transformer.mid_block)
        all_blocks.extend(flux_transformer.up_blocks)

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
    flux_transformer.learnable_weights = nn.Parameter(torch.ones(counter)).requires_grad_(True)
    return counter

def load_checkpoint(accelerator, args):
    # load pretrained model
    accelerator.print(f"Resuming from checkpoint {args.checkpoint_path}")
    accelerator.load_state(args.checkpoint_path)
    
def collate_fn(examples):
        target_image = torch.stack([example["target_image"] for example in examples]) 
        target_image = target_image.to(memory_format=torch.contiguous_format).float()
            
        # input_ids_one = torch.stack([example["input_ids"] for example in examples])
        # subject_input_ids_one = torch.vstack([padding_subjects(example["subject_input_ids"], example["padding_num"]) for example in examples])
        
        # input_ids_two = torch.stack([example["input_ids_two"] for example in examples])
        # subject_input_ids_two = torch.vstack([padding_subjects(example["subject_input_ids_two"], example["padding_num"]) for example in examples])
        
        subject_images = torch.vstack([example["subject_images"] for example in examples]) # B * N, C, W, H
        # we need to mask out the black padded subject imagee
        # padding_nums = torch.tensor([example["padding_num"] for example in examples])
        
        # if not args.with_staged_timestep:
        #     timesteps = torch.randint(0, 1000, (args.train_batch_size, ))
        # else:
        #     # some training hyperparams
        #     timesteps = []
        #     for example in examples: # following anydoor [0 ... High Res ... T/2 ... Low Res ... T]
        #         if example["dataset_name"] in ["vitonHD", "youtubeVIS", "coco2014"]: # low quality for the early stages 
        #             timesteps.append(min(random.randint(int(noise_scheduler.config.num_train_timesteps // 3), noise_scheduler.config.num_train_timesteps - 1), 999))
        #         else:
        #             t = random.randint(0, int(2 * noise_scheduler.config.num_train_timesteps // 3 - 1))
        #             timesteps.append(min(t, 999))
            
        #     timesteps = torch.tensor(timesteps).reshape((args.train_batch_size,))

        # target_noise = torch.randn((args.train_batch_size, 16, args.resolution // 8, args.resolution // 8), dtype=weight_dtype)
        # subject_noise = torch.randn((1, 16, args.resolution // 8, args.resolution // 8), dtype=weight_dtype)
        target_prompt = [example["target_prompt"] for example in examples]
        subject_prompt = [example["subject_prompt"][0] if type(example["subject_prompt"]) == list else example["subject_prompt"] for example in examples]
        subject_prompt = [example["subject_prompt"] for example in examples]
        return {"target_image": target_image, "input_ids_one": None, "input_ids_two": None,  "subject_input_ids_one": None, "subject_input_ids_two": None, "subject_images": subject_images, "padding_num": None, "dataset_name": [example["dataset_name"] for example in examples], "timesteps": None, "target_prompt": target_prompt, "subject_prompt": subject_prompt, "target_noise": None, "subject_noise": None}

def load_pipelines(pipeline_modules, flux_transformer_copy, weight_dtype, args):
    # pipeline = StableDiffusionPipeline_main.from_pretrained(
    #     args.pretrained_model_name_or_path,
    #     vae=vae,
    #     text_encoder=text_encoder,
    #     tokenizer=tokenizer,
    #     unet=flux_transformer,
    #     safety_checker=None,
    #     revision=args.revision,
    #     variant=args.variant,
    #     torch_dtype=weight_dtype,
    #     local_files_only=True
    # )
    pipeline = FluxPipeline_main.from_pretrained(
        args.pretrained_model_name_or_path,
        safety_checker=None,
        torch_dtype=weight_dtype,
        local_files_only=True,
        **pipeline_modules
        # variant="fp16"
    )

    pipeline.flux_transformer_copy = flux_transformer_copy

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
    torch.autograd.set_detect_anomaly(True)
    # load accelerator
    accelerator = init_accelerator(args)
    
    # Handle the repository creation
    if accelerator.is_main_process:
        os.makedirs(args.output_dir, exist_ok=True)
    
    weight_dtype = torch.float32
    if accelerator.mixed_precision == "fp16":
        weight_dtype = torch.float16
    elif accelerator.mixed_precision == "bf16":
        weight_dtype = torch.bfloat16
    
    # load models, here, only flux_transformer would contains trainable parameters while reference_UNet is merely a identical copy of SD2.1-base flux_transformer used for feature extraction
    flux_transformer, noise_scheduler_copy, noise_scheduler, tokenizer_one, tokenizer_two, text_encoder_one, text_encoder_two, vae, clip_model, clip_processor, num_of_adapters = load_models_and_learnable_params(args, accelerator.device, weight_dtype)
    
    flux_transformer_copy = None

    vae_scale_factor = 2 ** (len(vae.config.block_out_channels) - 1) if vae is not None else 8
    image_processor = VaeImageProcessor(vae_scale_factor=vae_scale_factor * 2)

    def apply_lora_to_all_linears(module, r=8, alpha=32):
        """
        对传入模块中所有 nn.Linear 层添加 LoRA。
        注意：操作是 in-place。
        """
        config = LoraConfig(
            r=r,
            lora_alpha=alpha,
            target_modules=["to_q", "to_k", "to_v", "to_out.0"],  # 关键：匹配所有 nn.Linear
            # target_modules=[
            #     "to_q", "to_k", "to_v",
            #     "add_q_proj", "add_k_proj", "add_v_proj",
            #     "to_out.0",             
            #     "to_add_out",           
            #     "ff.net.0.proj",       
            #     "ff.net.2",
            #     "ff_context.net.0.proj",
            #     "ff_context.net.2"
            # ],
            bias="none",
            task_type="FEATURE_EXTRACTION"
        )
        return get_peft_model(module, config)

    
    # # set which parameter is trainable, the addtional qkv
    # flux_transformer.cross_attn_transformer_blocks = copy.deepcopy(flux_transformer.transformer_blocks)
    # flux_transformer.add_module("cross_attn_transformer_blocks", flux_transformer.cross_attn_transformer_blocks)
    # flux_transformer.cross_attn_transformer_blocks.requires_grad_(True)

    # for i, block in enumerate(flux_transformer.cross_attn_transformer_blocks):
    #     # block = flux_transformer.cross_attn_transformer_blocks[i]
    #     block.sub_norm = copy.deepcopy(block.norm)
    #     # block.sub_proj_mlp = copy.deepcopy(block.proj_mlp)
    #     # block.sub_act_mlp = copy.deepcopy(block.act_mlp)

    #     block.add_module("sub_norm", block.sub_norm)
    #     # block.add_module("sub_proj_mlp", block.sub_proj_mlp)
    #     # block.add_module("sub_act_mlp", block.sub_act_mlp)
    #     block.sub_norm.requires_grad_(True)

    # # # 假设 flux_transformer 是一个完整模型
    # for i in range(len(flux_transformer.cross_attn_transformer_blocks)):
    #     block = flux_transformer.cross_attn_transformer_blocks[i]
    #     flux_transformer.cross_attn_transformer_blocks[i] = apply_lora_to_all_linears(block, r=64, alpha=64)
    
    # for i in range(len(flux_transformer.transformer_blocks)):
    #     block = flux_transformer.transformer_blocks[i]
    #     flux_transformer.transformer_blocks[i] = apply_lora_to_all_linears(block, r=64, alpha=64)


    # breakpoint()
    for i, block in enumerate(flux_transformer.transformer_blocks):
        # flux_transformer.transformer_blocks[i].cross_attn = copy.deepcopy(block.attn)
        # nsformer.transformer_blocks[i].add_module("cross_attn", flux_transformer.transformer_blocks[i].cross_attn)
        block.sub_norm = copy.deepcopy(block.norm1)
        block.add_module("sub_norm", block.sub_norm)
        block.sub_norm.requires_grad_(True)
        
        attn = block.attn
        attn.sub_to_k = copy.deepcopy(attn.to_k)
        attn.add_module("sub_to_k", attn.sub_to_k)
        attn.sub_to_k.requires_grad_(True)

        attn.sub_to_v = copy.deepcopy(attn.to_v)
        attn.add_module("sub_to_v", attn.sub_to_v)
        attn.sub_to_v.requires_grad_(True)

        # attn.sub_norm_k = copy.deepcopy(attn.norm_k)
        # attn.add_module("sub_norm_k", attn.sub_norm_k)
        # attn.sub_norm_k.requires_grad_(True)

        # attn.to_k.requires_grad_(True)
        # attn.to_v.requires_grad_(True)
        
        
        
    for i, block in enumerate(flux_transformer.single_transformer_blocks):
        block.sub_norm = copy.deepcopy(block.norm)
        block.add_module("sub_norm", block.sub_norm)
        block.sub_norm.requires_grad_(True)
        
        attn = block.attn
        attn.sub_to_k = copy.deepcopy(attn.to_k)
        attn.add_module("sub_to_k", attn.sub_to_k)
        attn.sub_to_k.requires_grad_(True)

        attn.sub_to_v = copy.deepcopy(attn.to_v)
        attn.add_module("sub_to_v", attn.sub_to_v)
        attn.sub_to_v.requires_grad_(True)

        # attn.sub_norm_k = copy.deepcopy(attn.norm_k)
        # attn.add_module("sub_norm_k", attn.sub_norm_k)
        # attn.sub_norm_k.requires_grad_(True)
        
        # attn.to_k.requires_grad_(True)
        # attn.to_v.requires_grad_(True)

    # get trainable params for optimizer
    trainable_params = get_trainable_params(flux_transformer)
    trainable_params_name = get_trainable_params_name(flux_transformer)
    
    if args.scale_lr:
        args.learning_rate = (args.learning_rate * args.gradient_accumulation_steps * args.train_batch_size * accelerator.num_processes)
        
    optimizer = torch.optim.AdamW(
        trainable_params,
        lr=args.learning_rate,
        betas=(args.adam_beta1, args.adam_beta2),
        weight_decay=args.adam_weight_decay,
        eps=args.adam_epsilon,
    )
    
    if accelerator.mixed_precision == "fp16":
        for params in trainable_params:
            params.to(torch.float16)
        cast_training_params([flux_transformer], dtype=torch.float16)
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
    
    dataset_to_train =[]
    if args.with_coco:
        coco2014_dataset = DatasetCOCO_sdxl(args.coco2014["data_path"], transform=train_transforms, max_len=args.num_sub_img, tokenizer_one=tokenizer_one, tokenizer_two=tokenizer_two, train_split=args.coco2014["train_split"], args=args, subset_size=args.coco2014['subset_size'])
        dataset_to_train.append(coco2014_dataset)
        
    if args.with_youtube_vis:
        youtubeVIS_dataset = YoutubeVISDataset_unet_sdxl(image_dir=args.youtubeVIS['image_dir'], anno=args.youtubeVIS['anno'], meta=args.youtubeVIS['meta'], tokenizer_one=tokenizer_one, tokenizer_two=tokenizer_two, sub_size=args.resolution, transforms=train_transforms, ytbvis_subset_size=args.youtubeVIS['subset_size'], args=args)
        dataset_to_train.append(youtubeVIS_dataset)

    if args.with_vitonhd:
        vitonHD_dataset = VitonHDDataset_unet(image_dir=args.vitonHD_dataset['image_dir'], tokenizer_one=tokenizer_one, tokenizer_two=tokenizer_two, sub_size=args.resolution, transforms=train_transforms, vitonhd_subset_size=args.vitonHD_dataset['subset_size'], args=args)
        dataset_to_train.append(vitonHD_dataset)
    
    if args.with_subject200k:
        # subject200k_dataset_sdxl = Subject200k_dataset_sdxl(args.subject200k["data_path"], transform=train_transforms, max_len=4, tokenizer_one=tokenizer_one, tokenizer_two=tokenizer_two, subset_size=args.subject200k['subset_size'], args=args)
        
        subject200k_dataset_sdxl = Subject200k_dataset_parquet_collection2(transform=train_transforms, subset_size=args.subject200k['subset_size'])
        dataset_to_train.append(subject200k_dataset_sdxl)

    if args.with_subject200k_jigsaw:
        subject200k_dataset_sdxl_jigsaw_sdxl = Subject200k_dataset_sdxl_jigsaw_sdxl(args.subject200k_jigsaw["data_path"], transform=train_transforms, max_len=4, tokenizer=tokenizer_one, tokenizer_two=tokenizer_two, subset_size=args.subject200k_jigsaw['subset_size'], args=args)
        dataset_to_train.append(subject200k_dataset_sdxl_jigsaw_sdxl)
    assert len(dataset_to_train) > 0, "No dataset is loaded!"
    
    train_dataset = ConcatDataset(dataset_to_train)
    
    
    # DataLoaders creation:
    train_dataloader = torch.utils.data.DataLoader(
        train_dataset,
        shuffle=True,
        collate_fn=collate_fn,
        batch_size=args.train_batch_size,
        num_workers=args.dataloader_num_workers,
        pin_memory=True,
        multiprocessing_context='fork',
        drop_last=True
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
    flux_transformer, optimizer, lr_scheduler, train_dataloader = accelerator.prepare(
        flux_transformer, optimizer, lr_scheduler, train_dataloader
    )
    
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
    
    # Potentially load in the weights and states from a previous save
    resumed = False
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
            
            # breakpoint()
            # if accelerator.is_main_process:
            #     flux_transformer = accelerator.unwrap_model(flux_transformer)
            #     flux_transformer.save_pretrained("/mnt/workspace/workgroup/duanzicheng.dzc/checkpoints/ezigen/debug/fp16_ckpt/", torch_dtype=torch.float16, is_main_process=accelerator.is_main_process, max_shard_size='50GB', safe_serialization=False)
            #     sys.exit(0)

            initial_global_step = global_step
            first_epoch = global_step // num_update_steps_per_epoch
            resumed = True # to mark if the model is freshly resumed from local
    else:
        initial_global_step = 0
        resumed = True
    

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
    

    # save_path = os.path.join(args.output_dir, "Final")

    # flux_transformer = accelerator.unwrap_model(flux_transformer)
    # flux_transformer.save_pretrained(save_path, torch_dtype=torch.float16, is_main_process=accelerator.is_main_process, max_shard_size='50GB', safe_serialization=False)

    # logger.info(f"Final model saved state to {save_path}")
        
    # state_dict = torch.load("/mnt/workspace/workgroup/duanzicheng.dzc/checkpoints/ezigen/train/train_sdxl_staged_timesteps_more_data_fp16/Final/diffusion_pytorch_model.bin")
    # flux_transformer.load_state_dict(state_dict, strict=False)


    # if accelerator.is_main_process:
    #     subject_noise = torch.randn((1, 4, args.resolution // 8, args.resolution // 8), dtype=weight_dtype)
    #     if (resumed==True or (args.target_prompt_1 is not None and global_step % args.validation_steps == 0)) and accelerator.is_main_process:
    #         vis_images = log_validation_batch(vae, text_encoder_one, text_encoder_two, tokenizer_one, tokenizer_two, accelerator.unwrap_model(flux_transformer), reference_unet, noise_scheduler, subject_noise, train_transforms, args, accelerator, weight_dtype, 0, batch_text_prompts, batch_origin_text_prompts, batch_subject_prompts, batch_img_paths, batch_variation_num, clip_model, clip_processor)
    # accelerator.wait_for_everyone()
    
    pipeline_modules = {
        "vae": vae,
        "text_encoder": text_encoder_one,
        "text_encoder_2": text_encoder_two,
        "tokenizer": tokenizer_one,
        "tokenizer_2": tokenizer_two,
        "scheduler": noise_scheduler,
    }
    subject_noise = torch.randn((1, 16, args.resolution // 8, args.resolution // 8), dtype=weight_dtype)
    # pipeline_modules["transformer"] = accelerator.unwrap_model(flux_transformer)
    # vis_images = log_validation_batch(pipeline_modules, noise_scheduler_copy, subject_noise, train_transforms, args, accelerator, weight_dtype, 1, batch_text_prompts, batch_origin_text_prompts, batch_subject_prompts, batch_img_paths, batch_variation_num, clip_model, clip_processor)
    
    for epoch in range(first_epoch, args.num_train_epochs):
        train_loss = 0.0
        for step, batch in enumerate(train_dataloader):
            ### load target related data
            target_image = batch["target_image"].to(weight_dtype)
            # target_input_ids_one = batch["input_ids_one"]
            # target_input_ids_two = batch["input_ids_two"]
            # target_noise = batch['target_noise'].to(weight_dtype)
            # target_timestep = batch['timesteps'].to(weight_dtype)
            # t_mask = target_timestep >= 1000
            # target_timestep[t_mask] = torch.randint(500, 999, (t_mask.sum(),)).to(weight_dtype).to(accelerator.device)

            prompt_embeds, pooled_prompt_embeds, text_ids = compute_text_embeddings(batch["target_prompt"], [text_encoder_one, text_encoder_two], [tokenizer_one, tokenizer_two])

            ### Get target noisy latent input
            model_input = vae.encode(target_image).latent_dist.sample()
            model_input = (model_input - vae.config.shift_factor) * vae.config.scaling_factor
            
            # Sample noise that we'll add to the latents
            noise = torch.randn_like(model_input, dtype=weight_dtype)
            bsz = model_input.shape[0]
            # Sample a random timestep for each image, for weighting schemes where we sample timesteps non-uniformly
            u = compute_density_for_timestep_sampling(weighting_scheme=None,batch_size=bsz,logit_mean=0.0,logit_std=1.0,mode_scale=1.29)
            indices = (u * noise_scheduler_copy.config.num_train_timesteps).long()
            timesteps = noise_scheduler_copy.timesteps[indices].to(device=accelerator.device, dtype=weight_dtype)
            # Add noise according to flow matching. zt = (1 - texp) * x + texp * z1
            # breakpoint()
            sigmas = get_sigmas(timesteps, noise_scheduler_copy, n_dim=model_input.ndim, dtype=weight_dtype)
            noisy_model_input = (1.0 - sigmas) * model_input + sigmas * noise
            # handle guidance, guidance = torch.tensor([args.guidance_scale], device=accelerator.device)
            guidance = torch.full([1], 3.5, device=accelerator.device, dtype=weight_dtype)
            guidance = guidance.expand(model_input.shape[0])
            
            noisy_model_input = pack_latents(noisy_model_input, bsz, 16, int(args.resolution/vae_scale_factor), int(args.resolution/vae_scale_factor))

            # ### Get subject feature as the target
            # subject_images = batch["subject_images"].to(weight_dtype)
            # subject_images_latent = vae.encode(subject_images).latent_dist.sample()
            # subject_images_latent = (subject_images_latent - vae.config.shift_factor) * vae.config.scaling_factor
            # sub_timesteps = torch.tensor([1], dtype=weight_dtype)
            # sub_sigmas = get_sigmas(sub_timesteps, noise_scheduler_copy, n_dim=subject_images.ndim, dtype=weight_dtype)
            # noisy_subject_images_latent = (1.0 - sub_sigmas) * subject_images_latent + sub_sigmas * noise #sub_sigma 0.0010

            # breakpoint()
            ### Get subject feature
            subject_images = batch["subject_images"].to(weight_dtype)
            # add noise to subject image latent
            noise_step =args.subject_timestep
            noisy_subject_latents = add_noise_to_image_flux(subject_images, vae, noise_step, noise_scheduler_copy, noise=noise) # mean: 0.3184

            # get subject text emb
            subject_prompt_embeds, subject_pooled_prompt_embeds, subject_text_ids = compute_text_embeddings(batch["subject_prompt"], [text_encoder_one, text_encoder_two], [tokenizer_one, tokenizer_two])
            
            latent_image_ids = prepare_latents(
                vae_scale_factor,
                bsz,
                args.resolution,
                args.resolution,
                weight_dtype,
                accelerator.device,
            )

            noisy_subject_latents = pack_latents(noisy_subject_latents, bsz, 16, int(args.resolution/vae_scale_factor), int(args.resolution/vae_scale_factor))
            with torch.no_grad():
                # extract subject image features from flux
                sub_noise_pred, subject_features = flux_transformer(
                    hidden_states=noisy_subject_latents,
                    # YiYi notes: divide it by 1000 for now because we scale it by 1000 in the transforme rmodel (we should not keep it but I want to keep the inputs same for the model for testing)
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

                def visualize_denoised(sub_noise_pred):
                    sub_latents = noise_scheduler.step(sub_noise_pred, torch.tensor([noise_step + 1]), noisy_subject_latents, return_dict=False)[0]
                    latents_out = unpack_latents(sub_latents, args.resolution, args.resolution,vae_scale_factor)
                    latents_out = (latents_out / vae.config.scaling_factor) + vae.config.shift_factor
                    image = vae.decode(latents_out.to(vae.dtype), return_dict=False)[0]
                    image = image_processor.postprocess(image, output_type='pil')[0]
                    return image

                # breakpoint()

            # subject_features = [subject_feature[:, -noisy_subject_latents.shape[1]:, :] for subject_feature in subject_features]
            
            model_pred, _ = flux_transformer(
                hidden_states=noisy_model_input,
                # YiYi notes: divide it by 1000 for now because we scale it by 1000 in the transforme rmodel (we should not keep it but I want to keep the inputs same for the model for testing)
                timestep=timesteps / 1000,
                guidance=guidance,
                pooled_projections=pooled_prompt_embeds,
                encoder_hidden_states=prompt_embeds,
                txt_ids=torch.zeros(prompt_embeds.shape[1], 3).to(device=accelerator.device, dtype=weight_dtype),
                img_ids=latent_image_ids,
                return_dict=False,
                subject_features=subject_features,
                is_train=True
            )
            
            model_pred = unpack_latents(model_pred,height=args.resolution,width=args.resolution,vae_scale_factor=vae_scale_factor,)
            
            weighting = compute_loss_weighting_for_sd3(weighting_scheme=None, sigmas=sigmas)

            # flow matching loss
            target = noise - model_input

            # Compute regular loss.
            flux_loss = torch.mean((weighting.float() * (model_pred.float() - target.float()) ** 2).reshape(target.shape[0], -1),1)
            flux_loss = flux_loss.mean()

            avg_loss = accelerator.gather(flux_loss.repeat(args.train_batch_size)).mean()
            train_loss += avg_loss.item() / args.gradient_accumulation_steps

            # breakpoint()
            accelerator.backward(flux_loss)
            if accelerator.sync_gradients:
                accelerator.clip_grad_norm_(trainable_params, args.max_grad_norm)
            optimizer.step()
            lr_scheduler.step()
            optimizer.zero_grad()
                
            # Checks if the accelerator has performed an optimization step behind the scenesw
            if accelerator.sync_gradients:
                progress_bar.update(1)
                global_step += 1
                accelerator.log({"train_loss": train_loss}, step=global_step)
                train_loss = 0.0
                if global_step % args.checkpointing_steps == 0:
                    if args.checkpoints_total_limit is not None:
                        checkpoints = os.listdir(args.output_dir)
                        checkpoints = [d for d in checkpoints if d.startswith("checkpoint")]
                        checkpoints = sorted(checkpoints, key=lambda x: int(x.split("-")[1]))
                        # before we save the new checkpoint, we need to have at_most_`checkpoints_total_limit - 1` checkpoints
                        if len(checkpoints) >= args.checkpoints_total_limit:
                            num_to_remove = len(checkpoints) - args.checkpoints_total_limit + 1
                            removing_checkpoints = checkpoints[0:num_to_remove]
                            logger.info(f"{len(checkpoints)} checkpoints already exist, removing {len(removing_checkpoints)} checkpoints")
                            logger.info(f"removing checkpoints: {', '.join(removing_checkpoints)}")
                            try:
                                for removing_checkpoint in removing_checkpoints:
                                    removing_checkpoint = os.path.join(args.output_dir,removing_checkpoint)
                                    shutil.rmtree(removing_checkpoint)
                            except:
                                pass
                                  
                    save_path = os.path.join(args.output_dir, f"checkpoint-{global_step}")
                        
                    accelerator.save_state(save_path)
                    
                    unwrapped_flux_transformer = accelerator.unwrap_model(flux_transformer)
                    unwrapped_flux_transformer.save_pretrained(save_path, torch_dtype=torch.float16, is_main_process=accelerator.is_main_process, max_shard_size='100GB', safe_serialization=False)

                    resumed = False
                    # accelerator.save_model(flux_transformer, save_path)
                    # save_random_state(os.path.join(save_path, "random_states_0.pth"))
                    logger.info(f"Saved state to {save_path}")
            
                logs = {"step_loss": flux_loss.detach().item(), "lr": lr_scheduler.get_last_lr()[0], "GPU Mem": f"{get_gpu_usage(accelerator.device)}"}
                progress_bar.set_postfix(**logs)
                if global_step >= args.max_train_steps:
                    break
            
            if (resumed==True or (args.target_prompt_1 is not None and global_step % args.validation_steps == 0)) and accelerator.is_main_process:
                pipeline_modules["transformer"] = accelerator.unwrap_model(flux_transformer)
                vis_images = log_validation_batch(pipeline_modules, flux_transformer_copy, noise_scheduler_copy, subject_noise, train_transforms, args, accelerator, weight_dtype, 1, batch_text_prompts, batch_origin_text_prompts, batch_subject_prompts, batch_img_paths, batch_variation_num, clip_model, clip_processor)
                resumed = False

    # if accelerator.is_main_process:
    if not resumed or args.num_train_epochs <= initial_global_step:
        print("Final saving!")
        save_path = os.path.join(args.output_dir, f"checkpoint-{global_step}")
        accelerator.save_state(save_path)
        
        flux_transformer = accelerator.unwrap_model(flux_transformer)
        # flux_transformer.save_pretrained(save_path, torch_dtype=weight_dtype, is_main_process=accelerator.is_main_process, max_shard_size='100GB', safe_serialization=False)
        for i, block in enumerate(flux_transformer.cross_attn_transformer_blocks):
            block.save_pretrained(f"{save_path}/block_{i}")

        logger.info(f"Final model saved state to {save_path}")
        
        # state_dict = torch.load(os.path.join(save_path, "diffusion_pytorch_model.bin"))
        # flux_transformer.load_state_dict(state_dict, strict=False)

        print("Final eval!")
        if accelerator.is_main_process:
            pipeline_modules["transformer"] = accelerator.unwrap_model(flux_transformer)
            vis_images = log_validation_batch(pipeline_modules, flux_transformer_copy, noise_scheduler_copy, subject_noise, train_transforms, args, accelerator, weight_dtype, 1, batch_text_prompts, batch_origin_text_prompts, batch_subject_prompts, batch_img_paths, batch_variation_num, clip_model, clip_processor)

        # sys.exit()
        # os.kill(os.getpid(), signal.SIGKILL)
        
    accelerator.wait_for_everyone()
    accelerator.end_training()
    sys.exit()
    os.kill(os.getpid(), signal.SIGKILL)
    sys.exit()

if __name__ == "__main__":
    main(config_path=None, config_file=None)