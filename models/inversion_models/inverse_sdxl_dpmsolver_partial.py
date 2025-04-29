import torch
import argparse
import sys
sys.path.append("..")
sys.path.append(".")
from PIL import Image
import PIL
from .inverse_pipeline_xl_partial import InversePipelineXLPartial
from diffusers import StableDiffusionPipeline, DDIMScheduler, PNDMScheduler, EulerDiscreteScheduler, DPMSolverMultistepInverseScheduler, DPMSolverMultistepScheduler,StableDiffusionXLPipeline, DDPMScheduler
from .pipline_sd_partial import *
from .schedulers import InverseDDIMScheduler, InversePNDMScheduler, InverseEulerDiscreteScheduler
from .clip import ExceptionCLIPTextModel, ExceptionCLIPTextModelWithProj

if torch.cuda.is_available():
    device = "cuda"
else:
    device = "cpu"

def partial_inverse_xl(threshold_timestep, img: PIL.Image, pipe, unet=None, save_decoded=False, num_inference_steps=20):
    prompt_str = ""
    outputs = pipe(
        prompt_str, 
        guidance_scale=1,
        num_inference_steps=num_inference_steps,
        image=img,
        threshold_timestep=threshold_timestep,
        output_type="latent",
    )
    
    noise_image, noise, decode_image, inversed_intermediate_latents = outputs["images"][0], outputs["noise"][0], outputs["decode_images"][0], outputs["inversed_intermediate_latents"]
    
    if save_decoded:
        denoise_pipe = StableDiffusionXLPipeline_main.from_pretrained("stabilityai/stable-diffusion-xl-base-1.0").to(device)
        denoise_pipe.scheduler = DDIMScheduler.from_config(denoise_pipe.scheduler.config)
        
        denoise_pipe.unet = pipe.unet
        pipe.scheduler = DDPMScheduler.from_config(pipe.scheduler.config, local_files_only=True)
        outputs = denoise_pipe(
            prompt_str, 
            guidance_scale=1,
            num_inference_steps=num_inference_steps,
            latents=noise.unsqueeze(0),
            noise_step=threshold_timestep
        ) 
        recon_image = outputs["images"][0]
        recon_image.save(f"adapter_recon.jpg")
        breakpoint()
    return noise.unsqueeze(0), inversed_intermediate_latents




