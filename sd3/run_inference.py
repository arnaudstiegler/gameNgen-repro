from diffusers import AutoencoderKL, DDIMScheduler, UNet2DConditionModel
from huggingface_hub import hf_hub_download
from diffusers import (
    AutoencoderKL,
    DDIMScheduler,
    UNet2DConditionModel,
    DDPMScheduler,
)
import torch
from typing import List
from tqdm import tqdm
from diffusers.image_processor import VaeImageProcessor
import numpy as np
import random
from safetensors import safe_open
import torch

REPO_NAME = 'arnaudstiegler/sd-model-gameNgen'
torch.manual_seed(9052924)
np.random.seed(9052924)
random.seed(9052924)

def read_action_embedding_from_safetensors(file_path: str):
    with safe_open(file_path, framework="pt", device="cpu") as f:
        embedding_weight = f.get_tensor("weight")

    num_embeddings, embedding_dim = embedding_weight.shape
    action_embedding = torch.nn.Embedding(num_embeddings, embedding_dim)
    action_embedding.weight.data = embedding_weight
    return action_embedding

def get_latents(
    noise_scheduler: DDPMScheduler,
    prev_frames: List[torch.Tensor],
    device: torch.device,
    dtype=torch.float32,
):

    latents = torch.concat(prev_frames, dim=1).to(device).type(dtype)
    # scale the initial noise by the standard deviation required by the scheduler
    latents = latents * noise_scheduler.init_noise_sigma
    return latents

def run_inference(images):
    device = torch.device(
        "cuda"
        if torch.cuda.is_available()
        else "mps"
        if torch.backends.mps.is_available()
        else "cpu"
    )
    
    unet = (
        UNet2DConditionModel.from_pretrained(REPO_NAME, subfolder="unet")
        .eval()
        .to(device)
    )
    vae = AutoencoderKL.from_pretrained(REPO_NAME, subfolder="vae").eval().to(device)
    noise_scheduler = DDIMScheduler.from_pretrained(
        REPO_NAME, subfolder="noise_scheduler"
    )

    file_path = hf_hub_download(
        repo_id=REPO_NAME, filename="action_embedding_model.safetensors"
    )
    action_embedding = (
        read_action_embedding_from_safetensors(file_path).eval().to(device)
    )

    # Defining all config-related variables here
    batch_size = 1
    do_classifier_free_guidance = True
    actions = [1]
    num_inference_steps = 50
    generator = torch.Generator(device=device)
    dtype = next(unet.parameters()).dtype
    vae_scale_factor = 2 ** (len(vae.config.block_out_channels) - 1)
    height = unet.config.sample_size * vae_scale_factor
    width = unet.config.sample_size * vae_scale_factor
    guidance_scale = 7.5

    image_processor = VaeImageProcessor(vae_scale_factor=vae_scale_factor)

    with torch.no_grad():
        noise_scheduler.set_timesteps(num_inference_steps, device=device)
        timesteps = noise_scheduler.timesteps

        prev_frames = []
        for img in images:
            original_image = img.reshape(1, 3, width, height).to(device)
            encoded_image = vae.encode(original_image).latent_dist.sample()
            encoded_image *= vae.config.scaling_factor
            prev_frames.append(encoded_image)
        
        prev_frames.append(torch.randn_like(prev_frames[0]))
        
        latents = get_latents(
            noise_scheduler,
            prev_frames,
            device,
            dtype,
        )
        images = []
        for i, t in tqdm(enumerate(timesteps), total=num_inference_steps):
            # expand the latents if we are doing classifier free guidance
            latent_model_input = (
                torch.cat([latents] * 2) if do_classifier_free_guidance else latents
            )
            latent_model_input = noise_scheduler.scale_model_input(
                latent_model_input, t
            )
            action_hidden_states = action_embedding(
                torch.tensor([[1, 1]]).to(device)
            ).repeat(2, 1, 1)

            noise_pred = unet(
                latent_model_input,
                t,
                encoder_hidden_states=action_hidden_states,
                timestep_cond=None,
                return_dict=False,
            )[0]
            # perform guidance
            if do_classifier_free_guidance:
                noise_pred_uncond, noise_pred_text = noise_pred.chunk(2)
                noise_pred = noise_pred_uncond + guidance_scale * (
                    noise_pred_text - noise_pred_uncond
                )
            latent = latents[:, -2:-1, ...]
            latent = noise_scheduler.step(
                noise_pred, t, latent, generator=None, return_dict=False
            )[0]
           

            print(latent.shape)
            image = vae.decode(
                latent / vae.config.scaling_factor, return_dict=False, generator=generator
            )[0]

            image = image_processor.postprocess(
                image.detach(), output_type="pil", do_denormalize=[True] * image.shape[0]
            )
            image[0].save(f"./image_steps_{i}.png")
            images.append(image)
    return images
            

if __name__ == "__main__":
    images = [torch.randn(3, 512, 512) for _ in range(9)]
    generated_images = run_inference(images)
    for index, image in enumerate(generated_images):
        with open(f"inference_test/image_steps_{index}.png", "wb") as f:
            image[0].save(f, "PNG")

