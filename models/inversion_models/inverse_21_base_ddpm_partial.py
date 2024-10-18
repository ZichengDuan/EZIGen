import torch
import argparse
import sys
sys.path.append("..")
sys.path.append(".")
from PIL import Image
import PIL
from .inverse_pipeline_partial import InversePipelinePartial
from diffusers import StableDiffusionPipeline, DDIMScheduler, PNDMScheduler, EulerDiscreteScheduler, DPMSolverMultistepInverseScheduler, DPMSolverMultistepScheduler,StableDiffusionXLPipeline, DDPMScheduler
from .pipline_sd_partial import *
from .schedulers import InverseDDIMScheduler, InversePNDMScheduler, InverseEulerDiscreteScheduler
from .clip import ExceptionCLIPTextModel, ExceptionCLIPTextModelWithProj

if torch.cuda.is_available():
    device = "cuda"
else:
    device = "cpu"

def partial_inverse(noise_step, img: PIL.Image, exclip, pipe, unet=None, save_decoded=False, num_inference_steps=20):
    if unet:
        pipe.unet = unet
    prompt_str = ""
    outputs = pipe(
        prompt_str, 
        guidance_scale=1,
        num_inference_steps=num_inference_steps,
        image=img,
        noise_step=noise_step
    )
    
    noise_image, noise, decode_image, inversed_intermediate_latents = outputs["images"][0], outputs["noise"][0], outputs["decode_images"][0], outputs["inversed_intermediate_latents"]
    
    if save_decoded and unet:
        denoise_pipe = StableDiffusionPipelinePartial.from_pretrained("stabilityai/stable-diffusion-2-1-base", text_encoder=exclip, local_files_only=True).to("cuda:0")
        denoise_pipe.unet = unet
        pipe.scheduler = DDPMScheduler.from_config(pipe.scheduler.config, local_files_only=True)
        outputs = denoise_pipe(
            prompt_str, 
            guidance_scale=1,
            num_inference_steps=num_inference_steps,
            latents=noise.unsqueeze(0),
            noise_step=noise_step
        ) 
        recon_image = outputs["images"][0]
        recon_image.save(f"adapter_recon.jpg")
    
    return noise.unsqueeze(0), inversed_intermediate_latents




