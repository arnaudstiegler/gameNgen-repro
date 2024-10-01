import argparse
import base64
import io
import random
from typing import List, Optional

import numpy as np
import torch
from datasets import DatasetDict, load_dataset
from diffusers import (AutoencoderKL, DDIMScheduler, DDPMScheduler,
                       UNet2DConditionModel)
from diffusers.image_processor import VaeImageProcessor
from diffusers.utils.torch_utils import randn_tensor
from huggingface_hub import hf_hub_download
from PIL import Image
from safetensors import safe_open
from torchvision import transforms
from tqdm import tqdm
from transformers import CLIPImageProcessor, CLIPTextModel, CLIPTokenizer
import os
from config_sd import BUFFER_SIZE, HEIGHT, REPO_NAME, WIDTH

torch.manual_seed(9052924)
np.random.seed(9052924)
random.seed(9052924)

repo_name = "CompVis/stable-diffusion-v1-4"


def read_action_embedding_from_safetensors(file_path: str):
    with safe_open(file_path, framework="pt", device="cpu") as f:
        embedding_weight = f.get_tensor("weight")

    num_embeddings, embedding_dim = embedding_weight.shape
    action_embedding = torch.nn.Embedding(num_embeddings, embedding_dim)
    action_embedding.weight.data = embedding_weight
    return action_embedding


def encode_prompt(
    tokenizer: CLIPTokenizer,
    text_encoder: CLIPTextModel,
    prompts: List[str],
    negative_prompt: List[str],
    batch_size: int,
    device: str,
    do_classifier_free_guidance: bool,
    num_images_per_prompt: int,
):
    assert isinstance(prompts, list), f"Expected list but received: {type(prompts)}"
    text_inputs = tokenizer(
        prompts,
        padding="max_length",
        max_length=tokenizer.model_max_length,
        truncation=True,
        return_tensors="pt",
    )
    # NB: the diffusers pipeline is seemingly not using attention_masks for the text encoder
    prompt_embeds = text_encoder(text_inputs.input_ids.to(device))
    prompt_embeds = prompt_embeds[0]

    bs_embed, seq_len, _ = prompt_embeds.shape
    # duplicate text embeddings for each generation per prompt, using mps friendly method
    prompt_embeds = prompt_embeds.repeat(1, num_images_per_prompt, 1)
    prompt_embeds = prompt_embeds.view(bs_embed * num_images_per_prompt, seq_len, -1)

    prompt_embeds_dtype = prompt_embeds.dtype

    if do_classifier_free_guidance:
        uncond_tokens = [""] * batch_size if not negative_prompt else negative_prompt
        uncond_input = tokenizer(
            uncond_tokens,
            padding="max_length",
            max_length=tokenizer.model_max_length,
            truncation=True,
            return_tensors="pt",
        )
        negative_prompt_embeds = text_encoder(uncond_input.input_ids.to(device))
        negative_prompt_embeds = negative_prompt_embeds[0]

        seq_len = negative_prompt_embeds.shape[1]

        negative_prompt_embeds = negative_prompt_embeds.to(
            dtype=prompt_embeds_dtype, device=device
        )

        negative_prompt_embeds = negative_prompt_embeds.repeat(
            1, num_images_per_prompt, 1
        )
        negative_prompt_embeds = negative_prompt_embeds.view(
            batch_size * num_images_per_prompt, seq_len, -1
        )

    else:
        negative_prompt_embeds = None

    return prompt_embeds, negative_prompt_embeds


def encode_conditioning_frames(
    vae: AutoencoderKL, conditioning_frames: list[Image.Image], dtype: torch.dtype
):
    conditioning_frames = [
        Image.open(io.BytesIO(base64.b64decode(img))) for img in conditioning_frames
    ]
    transform = transforms.ToTensor()
    conditioning_frames_tensor = torch.stack(
        [transform(image.convert("RGB")) for image in conditioning_frames]
    )
    conditioning_frames = vae.encode(
        conditioning_frames_tensor.to(device=vae.device, dtype=dtype)
    ).latent_dist.sample()
    conditioning_frames = conditioning_frames * vae.config.scaling_factor
    return conditioning_frames


def get_latents(
    noise_scheduler: DDPMScheduler,
    batch_size: int,
    height: int,
    width: int,
    num_channels_latents: int,
    vae_scale_factor: int,
    device: torch.device,
    dtype=torch.float32,
):
    # TODO: here we need to 1) generate a random tensor for the last frame and 2) concatenate the conditioning frames to it

    shape = (
        batch_size,
        num_channels_latents // BUFFER_SIZE,
        int(height) // vae_scale_factor,
        int(width) // vae_scale_factor,
    )
    latents = randn_tensor(shape, generator=None, device=device, dtype=dtype)

    # scale the initial noise by the standard deviation required by the scheduler
    latents = latents * noise_scheduler.init_noise_sigma
    return latents


def run_inference_with_params(
    unet,
    vae,
    noise_scheduler,
    action_embedding,
    tokenizer,
    text_encoder,
    batch,
    device,
    num_inference_steps=30,
    do_classifier_free_guidance=True,
    guidance_scale=7.5,
    skip_image_conditioning=False,
    skip_action_conditioning=False,
):
    vae_scale_factor = 2 ** (len(vae.config.block_out_channels) - 1)
    image_processor = VaeImageProcessor(vae_scale_factor=vae_scale_factor)
    with torch.no_grad():
        # Modify these lines to work with a single item
        images = batch["pixel_values"].unsqueeze(0)  # Add batch dimension
        actions = batch["input_ids"].unsqueeze(0)  # Add batch dimension
        if skip_image_conditioning:
            conditioning_frames = None
            batch_size, channels, height, width = images.shape
            #TODO: batched inference, stop hardcoding dimensions
            batch_size=1
            latent_channels=4
            latent_height=64
            latent_width=64
        else:
            # Reshape and encode conditioning frames
            batch_size, buffer_size, channels, height, width = images.shape
            conditioning_frames = images[:, : BUFFER_SIZE - 1].reshape(
                -1, channels, height, width
            )

            conditioning_frames_latents = vae.encode(
                conditioning_frames.to(device)
            ).latent_dist.sample()
            conditioning_frames_latents = (
                conditioning_frames_latents * vae.config.scaling_factor
            )

            # Reshape conditioning_frames_latents back to include batch and buffer dimensions
            _, latent_channels, latent_height, latent_width = (
                conditioning_frames_latents.shape
            )
            conditioning_frames_latents = conditioning_frames_latents.reshape(
                batch_size,
                BUFFER_SIZE - 1,
                latent_channels,
                latent_height,
                latent_width,
            )

        # Generate initial noise for the last frame
        latents = torch.randn(
            (batch_size, latent_channels, latent_height, latent_width), device=device
        )
        latents = latents * noise_scheduler.init_noise_sigma

        if not skip_image_conditioning:
            # Concatenate conditioning frames with the noisy last frame
            latents = torch.cat(
                [conditioning_frames_latents, latents.unsqueeze(1)], dim=1
            )

        # Prepare timesteps
        noise_scheduler.set_timesteps(num_inference_steps, device=device)
        timesteps = noise_scheduler.timesteps

        if not skip_action_conditioning:
            if do_classifier_free_guidance:
                # Not sure what to do for the negative prompt in that case
                encoder_hidden_states = action_embedding(actions.to(device))
            else:
                encoder_hidden_states = action_embedding(actions.to(device))
        else:
            if do_classifier_free_guidance:
                encoder_hidden_states = text_encoder(torch.stack([tokenizer.encode(["doom image, high quality, 4k, high resolution"], return_tensors="pt"), tokenizer.encode(["low quality, low poly, low resolution"], return_tensors="pt")]).to(device))[0]
            else:
                encoder_hidden_states = text_encoder(tokenizer.encode(["doom image, high quality, 4k, high resolution"], return_tensors="pt").to(device))[0]


        latents = latents.view(1, -1, latent_height, latent_width)

        # Denoising loop
        for _, t in enumerate(timesteps):
            latent_model_input = (
                torch.cat([latents] * 2) if do_classifier_free_guidance else latents
            )
            latent_model_input = noise_scheduler.scale_model_input(latent_model_input, t)

            if not skip_action_conditioning:
                # Predict noise
                noise_pred = unet(
                    latent_model_input,
                    t,
                    encoder_hidden_states=encoder_hidden_states,
                    timestep_cond=None,
                    return_dict=False,
                )[0]
                # Perform denoising step on the last frame only
                reshaped_frames = latents.reshape(
                    batch_size,
                    BUFFER_SIZE,
                    latent_channels,
                    latent_height,
                    latent_width,
                )
                last_frame = reshaped_frames[:, -1]
                denoised_last_frame = noise_scheduler.step(
                    noise_pred, t, last_frame, return_dict=False
                )[0]
                reshaped_frames[:, -1] = denoised_last_frame
                latents = reshaped_frames.reshape(
                    batch_size, -1, latent_height, latent_width
                )
                #TODO: unfinished here

            else:
                noise_pred = unet(
                    latent_model_input,
                    t,
                    encoder_hidden_states=encoder_hidden_states,
                    timestep_cond=None,
                    return_dict=False,
                )[0]

                if do_classifier_free_guidance:
                    noise_pred_uncond, noise_pred_text = noise_pred.chunk(2)
                    noise_pred = noise_pred_uncond + guidance_scale * (
                        noise_pred_text - noise_pred_uncond
                    )

                latents = noise_scheduler.step(
                    noise_pred, t, latents, generator=None, return_dict=False
                )[0]

        # Decode the last frame
        image = vae.decode(
            latents / vae.config.scaling_factor, return_dict=False
        )[0]

        # Post-process the image
        image = image_processor.postprocess(
            image.detach(), output_type="pil", do_denormalize=[True] * image.shape[0]
            )
        image[0].save(f"./image_steps{t}.png")

    return image[0]


# if __name__ == "__main__":
#     device = torch.device(
#         "cuda"
#         if torch.cuda.is_available()
#         else "mps"
#         if torch.backends.mps.is_available()
#         else "cpu"
#     )
#     batch = load_dataset("P-H-B-D-a16z/ViZDoom-Deathmatch-PPO")["test"][0]
#     unet = (
#         UNet2DConditionModel.from_pretrained(args.output_dir, subfolder="unet")
#         .eval()
#         .to(device)
#     )
#     vae = (
#         AutoencoderKL.from_pretrained(args.output_dir, subfolder="vae")
#         .eval()
#         .to(device)
#     )
#     noise_scheduler = DDIMScheduler.from_pretrained(
#         args.output_dir, subfolder="scheduler"
#     )

#     action_embedding = read_action_embedding_from_safetensors(
#         os.path.join(args.output_dir, "action_embedding_model.safetensors")
#     )

    # TODO: bring back
    # run_inference_with_params(
    #     unet,
    #     vae,
    #     noise_scheduler,
    #     action_embedding,
    #     tokenizer,
    #     text_encoder,
    #     batch,
    # )
