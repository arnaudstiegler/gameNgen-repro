import base64
import io
import random
from typing import List, Optional

import numpy as np
import torch
from diffusers import (DDIMScheduler, DDPMScheduler,
                       UNet2DConditionModel)
from diffusers.image_processor import VaeImageProcessor
from diffusers.utils.torch_utils import randn_tensor
from torch.amp import autocast


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
    shape = (
        batch_size,
        num_channels_latents,
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
    skip_action_conditioning=False,
):
    assert batch["pixel_values"].shape[0] == 1, "Batch size must be 1"

    generator = torch.Generator(device=device)
    vae_scale_factor = 2 ** (len(vae.config.block_out_channels) - 1)
    image_processor = VaeImageProcessor(vae_scale_factor=vae_scale_factor)
    with torch.no_grad(), autocast(device_type='cuda', dtype=torch.float32):
        images = batch["pixel_values"]
        actions = batch["input_ids"]
        batch_size = images.shape[0]

        if not skip_action_conditioning:
            if do_classifier_free_guidance:
                # Not sure what to do for the negative prompt in that case
                encoder_hidden_states = action_embedding(actions.to(device=device))
            else:
                encoder_hidden_states = action_embedding(actions.to(device))
        else:
            if do_classifier_free_guidance:
                positive_prompt = tokenizer.encode(VALIDATION_PROMPT, return_tensors="pt", padding="max_length", max_length=tokenizer.model_max_length)
                negative_prompt = tokenizer.encode("", return_tensors="pt", padding="max_length", max_length=tokenizer.model_max_length)
                
                # BEWARE: should be unconditional first, then conditional
                encoder_hidden_states = text_encoder(torch.stack([negative_prompt, positive_prompt]).squeeze(1).to(device))[0]
            else:
                encoder_hidden_states = text_encoder(tokenizer.encode(VALIDATION_PROMPT, return_tensors="pt").to(device))[0]



        # Prepare timesteps
        noise_scheduler.set_timesteps(num_inference_steps, device=device)
        timesteps = noise_scheduler.timesteps

        num_channels_latents = unet.config.in_channels
        # Generate initial noise for the last frame
        latents = get_latents(
            noise_scheduler,
            batch_size,
            HEIGHT,
            WIDTH,
            num_channels_latents,
            vae_scale_factor,
            device,
            dtype=unet.dtype,
        )

        # Denoising loop
        for _, t in enumerate(timesteps):
            latent_model_input = (
                torch.cat([latents] * 2) if do_classifier_free_guidance else latents
            )
            latent_model_input = noise_scheduler.scale_model_input(latent_model_input, t)

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
            latents / vae.config.scaling_factor, return_dict=False, generator=generator
        )[0]

        # Post-process the image
        image = image_processor.postprocess(
            image.detach(), output_type="pil", do_denormalize=[True] * image.shape[0]
            )
        image[0].save(f"./image_steps{t}.png")

    return image[0]


def run_inference_img_conditioning_with_params(
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
    skip_action_conditioning=False,
):
    assert batch["pixel_values"].shape[0] == 1, "Batch size must be 1"
    vae_scale_factor = 2 ** (len(vae.config.block_out_channels) - 1)
    image_processor = VaeImageProcessor(vae_scale_factor=vae_scale_factor)
    with torch.no_grad(), autocast(device_type='cuda', dtype=torch.float32):
        images = batch["pixel_values"]
        actions = batch["input_ids"]
        latent_height = HEIGHT//vae_scale_factor
        latent_width = WIDTH//vae_scale_factor
        num_channels_latents = vae.config.latent_channels
    
        # Reshape and encode conditioning frames
        batch_size, _, channels, height, width = images.shape
        conditioning_frames = images[:, : BUFFER_SIZE].reshape(
            -1, channels, height, width
        )

        conditioning_frames_latents = vae.encode(
            conditioning_frames.to(device)
        ).latent_dist.sample()
        conditioning_frames_latents = (
            conditioning_frames_latents * vae.config.scaling_factor
        )
        # Reshape conditioning_frames_latents back to include batch and buffer dimensions
        conditioning_frames_latents = conditioning_frames_latents.reshape(
            batch_size,
            BUFFER_SIZE,
            num_channels_latents,
            latent_height,
            latent_width,
        )
        
        # Generate initial noise for the last frame
        latents = get_latents(
            noise_scheduler,
            batch_size,
            HEIGHT,
            WIDTH,
            num_channels_latents,
            vae_scale_factor,
            device,
            dtype=unet.dtype,
        )

        # Prepare timesteps
        noise_scheduler.set_timesteps(num_inference_steps, device=device)
        timesteps = noise_scheduler.timesteps

        if not skip_action_conditioning:
            if do_classifier_free_guidance:
                # Repeat the encoder hidden states for the unconditional case
                # We don't "uncondition" on the action embedding
                encoder_hidden_states = action_embedding(actions.to(device)).repeat(2,1,1)
            else:
                encoder_hidden_states = action_embedding(actions.to(device))
        else:
            if do_classifier_free_guidance:
                positive_prompt = tokenizer.encode(VALIDATION_PROMPT, return_tensors="pt", padding="max_length", max_length=tokenizer.model_max_length)
                negative_prompt = tokenizer.encode("", return_tensors="pt", padding="max_length", max_length=tokenizer.model_max_length)
                encoder_hidden_states = text_encoder(torch.stack([positive_prompt, negative_prompt]).squeeze(1).to(device))[0]
            else:
                encoder_hidden_states = text_encoder(tokenizer.encode(VALIDATION_PROMPT, return_tensors="pt").to(device))[0]


        latents = torch.cat(
            [conditioning_frames_latents, latents.unsqueeze(1)], dim=1
        )
        # FOld the conditioning frames into the channel dimension
        latents = latents.view(1, -1, latent_height, latent_width)

        # Denoising loop
        for _, t in enumerate(timesteps):
            if do_classifier_free_guidance:
                # In case of classifier free guidance, the unconditional case is without conditioning frames
                uncond_latents = latents.clone()
                uncond_latents[:, :BUFFER_SIZE] = torch.zeros_like(uncond_latents[:, :BUFFER_SIZE])
                # BEWARE: order is important, the unconditional case should come first
                latent_model_input = torch.cat([uncond_latents, latents])
            else:
                latent_model_input = latents
            latent_model_input = noise_scheduler.scale_model_input(latent_model_input, t)

            # Predict noise
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

            # Perform denoising step on the last frame only
            reshaped_frames = latents.reshape(
                batch_size,
                BUFFER_SIZE+1,
                num_channels_latents,
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
            # The conditioning frames should not be modified by the denoising process
            assert torch.all(conditioning_frames_latents == reshaped_frames[:, :BUFFER_SIZE])

        # only take the last frame
        image = vae.decode(
            reshaped_frames[:, -1] / vae.config.scaling_factor, return_dict=False
        )[0]

        # Post-process the image
        image = image_processor.postprocess(
            image.detach(), output_type="pil", do_denormalize=[True] * image.shape[0]
            )
        image[0].save(f"./image_steps{t}.png")

    return image[0]