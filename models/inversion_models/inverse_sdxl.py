import sys
sys.path.append("..")
sys.path.append(".")
import torch
import argparse
from PIL import Image
from models.inversion_models.inverse_pipeline_xl import InversePipelineXL
from models.inversion_models.inverse_pipeline_xl_partial import InversePipelineXLPartial
from diffusers import StableDiffusionPipeline, DDIMScheduler, PNDMScheduler, EulerDiscreteScheduler, DPMSolverMultistepInverseScheduler, DPMSolverMultistepScheduler, StableDiffusionXLPipeline, DDIMInverseScheduler
from models.main_unet.adapter import Attention_Adapter  # my model
from models.pipelines import StableDiffusionXLPipeline_main
from models.main_unet import UNet2DConditionModel_main
from transformers import AutoTokenizer
# from model.inversion_models.schedulers import InverseDDIMScheduler, InversePNDMScheduler, InverseEulerDiscreteScheduler
from models.inversion_models.clip import ExceptionCLIPTextModel, ExceptionCLIPTextModelWithProj
import numpy as np
import torch.nn as nn
import argparse

if torch.cuda.is_available():
    device = "cuda"
else:
    device = "cpu"

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
                    # copy_matched_parameters(attn_1, adapter)
                    # copy_matched_parameters(norm_1, adapter_norm)

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


def inverse_sdxl(args):
    exclip = ExceptionCLIPTextModel.from_pretrained(args.model_path, subfolder="text_encoder").to(device)
    exclip_2 = ExceptionCLIPTextModelWithProj.from_pretrained(args.model_path, subfolder="text_encoder_2").to(device)
    pipe = InversePipelineXL.from_pretrained(args.model_path, text_encoder = exclip, text_encoder_2=exclip_2).to(device)
    # pipe.scheduler = DPMSolverMultistepInverseScheduler.from_config(pipe.scheduler.config, local_files_only=True)
    # pipe = StableDiffusionXLPipeline.from_pretrained(args.model_path, text_encoder = exclip, text_encoder_2=exclip_2).to(device)
    # pipe.scheduler = InverseDDIMScheduler.from_config(pipe.scheduler.config)
    pipe.scheduler = DDIMInverseScheduler.from_config(pipe.scheduler.config)
    # pipe.scheduler = InversePNDMScheduler.from_config(pipe.scheduler.config)
    # pipe.scheduler = DPMSolverMultistepInverseScheduler.from_config(pipe.scheduler.config)

    image = Image.open(args.input_image).resize((1024,1024), Image.Resampling.LANCZOS).convert("RGB")
    # x0 = np.array(image)/255
    # x0 = torch.from_numpy(x0).permute(2, 0, 1).unsqueeze(dim=0).repeat(1, 1, 1, 1).to(device)
    # x0 = (x0 - 0.5) * 2.
    # with torch.no_grad():
    #     img_latents = pipe.vae.encode(x0.float()).latent_dist.sample().to(device)
    #     img_latents *= pipe.vae.config.scaling_factor
    
    prompt_str = ""
    outputs = pipe(
        prompt_str, 
        guidance_scale=1,
        num_inference_steps=args.num_inference_steps,
        # output_type="latent",
        image = image
    )

    noisy_latent = outputs["noise"][0]
    print(noisy_latent.mean(), noisy_latent.std())
    
    del pipe
    
    denoise_pipe = StableDiffusionXLPipeline.from_pretrained(args.model_path, text_encoder = exclip, text_encoder_2=exclip_2).to(device)
    denoise_pipe.scheduler = DDIMScheduler.from_config(denoise_pipe.scheduler.config)
    # denoise_pipe.scheduler = DPM`SolverMultistepScheduler.from_config(denoise_pipe.scheduler.config)

    
    prompt_str = ""
    outputs = denoise_pipe(
        prompt_str, 
        guidance_scale=1,
        num_inference_steps=args.num_inference_steps,
        latents=noisy_latent.unsqueeze(0)
    ) 
    recon_image = outputs["images"][0]
    recon_image.save(args.results_folder + "recon.jpg")


def inverse_sdxl_partial(args):
    args.do_editing = True
    args.add_before_ca = True
    args.skip_adapter_ratio = 1
    args.infer_steps = args.num_inference_steps
    args.residual_connection = True
    
    # exclip = ExceptionCLIPTextModel.from_pretrained(args.model_path, subfolder="text_encoder").to(device)
    # exclip_2 = ExceptionCLIPTextModelWithProj.from_pretrained(args.model_path, subfolder="text_encoder_2").to(device)
    pipe = InversePipelineXLPartial.from_pretrained(args.model_path).to(device)
    pipe.scheduler = DDIMInverseScheduler.from_config(pipe.scheduler.config)
    # pipe.scheduler = DPMSolverMultistepInverseScheduler.from_config(pipe.scheduler.config)
    
    denoise_pipe = StableDiffusionXLPipeline_main.from_pretrained(args.model_path).to(device)
    denoise_pipe.scheduler = DDIMScheduler.from_config(denoise_pipe.scheduler.config)
    

    image = Image.open(args.input_image).resize((1024,1024), Image.Resampling.LANCZOS).convert("RGB")  
    
    # pipe.unet = main_unet
    pipe.args = args
    pipe.to(device)
    
    threshold_timestep = args.split_ratio * 1000
    prompt_str = ""
    
    outputs = pipe(
        prompt_str, 
        guidance_scale=1,
        num_inference_steps=args.num_inference_steps,
        image=image,
        threshold_timestep=threshold_timestep,
        # output_type="latent",
    )
    
    
    noise_image, noise, decode_image, inversed_intermediate_latents = outputs["images"][0], outputs["noise"][0], outputs["decode_images"][0], outputs["inversed_intermediate_latents"]
    print(noise.mean(), noise.std())

    noise_image.save(args.results_folder + "noisy_image_partial2.jpg")

    args.skip_adapter_ratio = 0
    denoise_pipe.args = args
    
    # load main unet and its adapters
    main_unet = UNet2DConditionModel_main.from_pretrained(args.model_path, subfolder="unet", local_files_only=True).to(device)
    register_adapter_and_configs(main_unet, reference_unet=None, args=args)
    
    state_dict = torch.load("experiments/trained_sdxl/diffusion_pytorch_model.bin", map_location="cpu")
    main_unet.load_state_dict(state_dict, strict=False)
    
    denoise_pipe.unet = main_unet
    denoise_pipe.to(device)
    prompt_str = ""
    outputs = denoise_pipe(
        prompt_str, 
        guidance_scale=2,
        num_inference_steps=args.num_inference_steps,
        latents=noise.unsqueeze(0),
        inversed_intermediate_latents=inversed_intermediate_latents if args.do_editing else None,
        threshold_timestep=threshold_timestep,
        args=args
    ) 
    recon_image = outputs["images"][0]
    recon_image.save(args.results_folder + "recon_partial2.jpg")

def arg_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument('--input_image', type=str, default='example_images/source_images_with_masks/dog_car.png')
    parser.add_argument('--results_folder', type=str, default='outputs/')
    parser.add_argument('--num_inference_steps', type=int, default=20)
    parser.add_argument('--split_ratio', type=int, default=0.8)
    parser.add_argument('--model_path', type=str, default="stabilityai/stable-diffusion-xl-base-1.0")
    parser.add_argument('--config', type=str, default="")
    args = parser.parse_args()

    return args

if __name__ == "__main__":
    args = arg_parser()
    inverse_sdxl_partial(args)
    # inverse_sdxl(args)