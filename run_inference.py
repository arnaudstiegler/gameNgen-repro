import argparse
import random

import numpy as np
import torch
from diffusers import AutoencoderKL, DDPMScheduler, UNet2DConditionModel
from diffusers.image_processor import VaeImageProcessor
from diffusers.utils.torch_utils import randn_tensor
from PIL import Image
from torch.amp import autocast

from config_sd import (
    BUFFER_SIZE,
    CFG_GUIDANCE_SCALE,
    HEIGHT,
    TRAINING_DATASET_DICT,
    WIDTH,
    DEFAULT_NUM_INFERENCE_STEPS,
)
from dataset import get_single_batch
from model import load_model

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


def get_initial_noisy_latent(
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
    unet: UNet2DConditionModel,
    vae: AutoencoderKL,
    noise_scheduler: DDPMScheduler,
    action_embedding: torch.nn.Embedding,
    context_latents: torch.Tensor,
    actions: torch.Tensor,
    device: torch.device,
    num_inference_steps: int = DEFAULT_NUM_INFERENCE_STEPS,
    do_classifier_free_guidance: bool = True,
    guidance_scale: float = CFG_GUIDANCE_SCALE,
    skip_action_conditioning: bool = False,
):
    batch_size = context_latents.shape[0]
    latent_height = context_latents.shape[-2]
    latent_width = context_latents.shape[-1]
    num_channels_latents = context_latents.shape[2]

    vae_scale_factor = 2 ** (len(vae.config.block_out_channels) - 1)
    with torch.no_grad(), autocast(device_type="cuda", dtype=torch.float32):
        # Generate initial noise for the target frame
        latents = get_initial_noisy_latent(
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

            # The conditioning frames should not be modified by the denoising process
            assert torch.all(context_latents == reshaped_frames[:, :BUFFER_SIZE])

        # Return the final latents of the target frame only
        reshaped_frames = latents.reshape(
            batch_size,
            BUFFER_SIZE + 1,
            num_channels_latents,
            latent_height,
            latent_width,
        )
        return reshaped_frames[:, -1]


def decode_and_postprocess(
    vae: AutoencoderKL, image_processor: VaeImageProcessor, latents: torch.Tensor
) -> Image:
    image = vae.decode(latents / vae.config.scaling_factor, return_dict=False)[0]

    image = image_processor.postprocess(
        image.detach(), output_type="pil", do_denormalize=[True] * image.shape[0]
    )[0]
    return image


def run_inference_img_conditioning_with_params(
    unet,
    vae,
    noise_scheduler,
    action_embedding,
    tokenizer,
    text_encoder,
    batch,
    device,
    num_inference_steps=DEFAULT_NUM_INFERENCE_STEPS,
    do_classifier_free_guidance=True,
    guidance_scale=CFG_GUIDANCE_SCALE,
    skip_action_conditioning=False,
) -> Image:
    assert batch["pixel_values"].shape[0] == 1, "Batch size must be 1"
    vae_scale_factor = 2 ** (len(vae.config.block_out_channels) - 1)
    image_processor = VaeImageProcessor(vae_scale_factor=vae_scale_factor)
    with torch.no_grad(), autocast(device_type="cuda", dtype=torch.float32):
        actions = batch["input_ids"]

        conditioning_frames_latents = encode_conditioning_frames(
            vae,
            images=batch["pixel_values"],
            vae_scale_factor=vae_scale_factor,
            dtype=torch.float32,
        )
        new_frame = next_latent(
            unet=unet,
            vae=vae,
            noise_scheduler=noise_scheduler,
            action_embedding=action_embedding,
            context_latents=conditioning_frames_latents,
            device=device,
            actions=actions,
            skip_action_conditioning=skip_action_conditioning,
            num_inference_steps=num_inference_steps,
            do_classifier_free_guidance=do_classifier_free_guidance,
            guidance_scale=guidance_scale,
        )

        # only take the last frame
        image = decode_and_postprocess(
            vae=vae, image_processor=image_processor, latents=new_frame
        )
    return image


def main(model_folder: str) -> None:
    device = torch.device(
        "cuda"
        if torch.cuda.is_available()
        else "mps"
        if torch.backends.mps.is_available()
        else "cpu"
    )

    unet, vae, action_embedding, noise_scheduler, tokenizer, text_encoder = load_model(
        model_folder, device
    )

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
        skip_action_conditioning=False,
        do_classifier_free_guidance=False,
        guidance_scale=CFG_GUIDANCE_SCALE,
        num_inference_steps=DEFAULT_NUM_INFERENCE_STEPS,
    )
    img.save("validation_image.png")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Run inference with customizable parameters"
    )
    parser.add_argument(
        "--model_folder",
        type=str,
        help="Path to the folder containing the model weights",
    )
    args = parser.parse_args()

    main(
        model_folder=args.model_folder,
    )
