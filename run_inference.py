import argparse
import base64
import io
import random

import numpy as np
import torch
from diffusers import AutoencoderKL, DDPMScheduler
from diffusers.image_processor import VaeImageProcessor
from diffusers.utils.torch_utils import randn_tensor
from PIL.Image import Image
from config_sd import (
    BUFFER_SIZE,
    HEIGHT,
    WIDTH,
    VALIDATION_PROMPT,
    TRAINING_DATASET_DICT,
    ZERO_OUT_ACTION_CONDITIONING_PROB,
    CFG_GUIDANCE_SCALE,
)
from sd3.model import get_model, load_model
from torch.amp import autocast
from PIL import Image
from sd3.model import load_model
from dataset import get_single_batch


torch.manual_seed(9052924)
np.random.seed(9052924)
random.seed(9052924)


def encode_conditioning_frames(
    vae: AutoencoderKL, images: torch.Tensor, vae_scale_factor: int, dtype: torch.dtype
) -> torch.Tensor:
    batch_size, _, channels, height, width = images.shape
    context_frames = images[:, :BUFFER_SIZE].reshape(-1, channels, height, width)
    conditioning_frame_latents = vae.encode(
        context_frames.to(device=vae.device, dtype=dtype)
    ).latent_dist.sample()
    conditioning_frame_latents = conditioning_frame_latents * vae.config.scaling_factor

    # Reshape context latents
    conditioning_frame_latents = conditioning_frame_latents.reshape(
        batch_size,
        BUFFER_SIZE,
        vae.config.latent_channels,
        height // vae_scale_factor,
        width // vae_scale_factor,
    )
    return conditioning_frame_latents


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


def next_latent(
    unet,
    vae,
    noise_scheduler,
    action_embedding,
    context_latents,  # New parameter: pre-encoded context frames
    device,
    num_inference_steps=30,
    do_classifier_free_guidance=True,
    guidance_scale=7.5,
    skip_action_conditioning=False,
    actions=None,  # New parameter: action conditioning
):
    batch_size = context_latents.shape[0]
    latent_height = context_latents.shape[-2]
    latent_width = context_latents.shape[-1]
    num_channels_latents = context_latents.shape[2]

    vae_scale_factor = 2 ** (len(vae.config.block_out_channels) - 1)
    with torch.no_grad(), autocast(device_type="cuda", dtype=torch.float32):
        # Generate initial noise for the target frame
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
                encoder_hidden_states = action_embedding(actions.to(device)).repeat(
                    2, 1, 1
                )
            else:
                encoder_hidden_states = action_embedding(actions.to(device))

        latents = torch.cat([context_latents, latents.unsqueeze(1)], dim=1)

        # Fold the conditioning frames into the channel dimension
        latents = latents.view(batch_size, -1, latent_height, latent_width)

        # Denoising loop
        for _, t in enumerate(timesteps):
            if do_classifier_free_guidance:
                # In case of classifier free guidance, the unconditional case is without conditioning frames
                uncond_latents = latents.clone()
                uncond_latents[:, :BUFFER_SIZE] = torch.zeros_like(
                    uncond_latents[:, :BUFFER_SIZE]
                )
                # BEWARE: order is important, the unconditional case should come first
                latent_model_input = torch.cat([uncond_latents, latents])
            else:
                latent_model_input = latents
            latent_model_input = noise_scheduler.scale_model_input(
                latent_model_input, t
            )

            # Predict noise
            noise_pred = unet(
                latent_model_input,
                t,
                encoder_hidden_states=encoder_hidden_states,
                timestep_cond=None,
                class_labels=torch.zeros(batch_size, dtype=torch.long).to(device),
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
                BUFFER_SIZE + 1,
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

        # Return the final latents of the target frame only
        reshaped_frames = latents.reshape(
            batch_size,
            BUFFER_SIZE + 1,
            num_channels_latents,
            latent_height,
            latent_width,
        )
        return reshaped_frames[:, -1]


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
) -> Image:
    assert batch["pixel_values"].shape[0] == 1, "Batch size must be 1"
    vae_scale_factor = 2 ** (len(vae.config.block_out_channels) - 1)
    image_processor = VaeImageProcessor(vae_scale_factor=vae_scale_factor)
    batch_size = batch["pixel_values"].shape[0]
    with torch.no_grad(), autocast(device_type="cuda", dtype=torch.float32):
        actions = batch["input_ids"]
        latent_height = HEIGHT // vae_scale_factor
        latent_width = WIDTH // vae_scale_factor
        num_channels_latents = vae.config.latent_channels

        conditioning_frames_latents = encode_conditioning_frames(
            vae,
            images=batch["pixel_values"],
            vae_scale_factor=vae_scale_factor,
            dtype=torch.float32,
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
                encoder_hidden_states = action_embedding(actions.to(device)).repeat(
                    2, 1, 1
                )
            else:
                encoder_hidden_states = action_embedding(actions.to(device))
        else:
            if do_classifier_free_guidance:
                positive_prompt = tokenizer.encode(
                    VALIDATION_PROMPT,
                    return_tensors="pt",
                    padding="max_length",
                    max_length=tokenizer.model_max_length,
                )
                negative_prompt = tokenizer.encode(
                    "",
                    return_tensors="pt",
                    padding="max_length",
                    max_length=tokenizer.model_max_length,
                )
                encoder_hidden_states = text_encoder(
                    torch.stack([positive_prompt, negative_prompt])
                    .squeeze(1)
                    .to(device)
                )[0]
            else:
                encoder_hidden_states = text_encoder(
                    tokenizer.encode(VALIDATION_PROMPT, return_tensors="pt").to(device)
                )[0]

        latents = torch.cat([conditioning_frames_latents, latents.unsqueeze(1)], dim=1)
        # FOld the conditioning frames into the channel dimension
        latents = latents.view(1, -1, latent_height, latent_width)

        # Denoising loop
        for _, t in enumerate(timesteps):
            if do_classifier_free_guidance:
                # In case of classifier free guidance, the unconditional case is without conditioning frames
                uncond_latents = latents.clone()
                uncond_latents[:, :BUFFER_SIZE] = torch.zeros_like(
                    uncond_latents[:, :BUFFER_SIZE]
                )
                # BEWARE: order is important, the unconditional case should come first
                latent_model_input = torch.cat([uncond_latents, latents])
            else:
                latent_model_input = latents
            latent_model_input = noise_scheduler.scale_model_input(
                latent_model_input, t
            )

            # Predict noise
            noise_pred = unet(
                latent_model_input,
                t,
                encoder_hidden_states=encoder_hidden_states,
                timestep_cond=None,
                class_labels=torch.zeros(batch_size, dtype=torch.long).to(device),
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
                BUFFER_SIZE + 1,
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
            assert torch.all(
                conditioning_frames_latents == reshaped_frames[:, :BUFFER_SIZE]
            )

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


def main(
    model_folder: str, skip_image_conditioning: bool, skip_action_conditioning: bool
) -> None:
    device = torch.device(
        "cuda"
        if torch.cuda.is_available()
        else "mps"
        if torch.backends.mps.is_available()
        else "cpu"
    )

    if not model_folder:
        unet, vae, action_embedding, noise_scheduler, tokenizer, text_encoder = (
            get_model(
                action_embedding_dim=17, skip_image_conditioning=skip_image_conditioning
            )
        )
    else:
        unet, vae, action_embedding, noise_scheduler, tokenizer, text_encoder = (
            load_model(model_folder)
        )

    unet = unet.to(device)
    vae = vae.to(device)
    action_embedding = action_embedding.to(device)

    batch = get_single_batch(TRAINING_DATASET_DICT["small"])

    img = run_inference_img_conditioning_with_params(
        unet,
        vae,
        noise_scheduler,
        action_embedding,
        tokenizer,
        text_encoder,
        batch,
        device=device,
        skip_action_conditioning=skip_action_conditioning,
        do_classifier_free_guidance=False,
        guidance_scale=CFG_GUIDANCE_SCALE,
        num_inference_steps=50,
    )
    img.save("output.png")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Run inference with customizable parameters"
    )
    parser.add_argument(
        "--skip_image_conditioning", action="store_true", help="Skip image conditioning"
    )
    parser.add_argument(
        "--skip_action_conditioning",
        action="store_true",
        help="Skip action conditioning",
    )
    parser.add_argument(
        "--model_folder",
        type=str,
        help="Path to the folder containing the model weights",
    )
    args = parser.parse_args()

    main(
        model_folder=args.model_folder,
        skip_image_conditioning=args.skip_image_conditioning,
        skip_action_conditioning=args.skip_action_conditioning,
    )
