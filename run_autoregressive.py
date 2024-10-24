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
from config_sd import BUFFER_SIZE, HEIGHT, REPO_NAME, WIDTH, VALIDATION_PROMPT
from sd3.model import get_model, load_model
from functools import partial
from torch.amp import autocast
from safetensors.torch import load_file
import json
from data_augmentation import no_img_conditioning_augmentation
from config_sd import ZERO_OUT_ACTION_CONDITIONING_PROB


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


from PIL import Image
import random
import os

def generate_autoregressive_rollout(
    unet,
    vae,
    noise_scheduler,
    action_embedding,
    tokenizer,
    text_encoder,
    initial_batch,
    device,
    num_frames=30,
    output_dir="rollout_frames",
    skip_action_conditioning=False,
    do_classifier_free_guidance=False,
    guidance_scale=1.5,
    num_inference_steps=50,
):

    #TODO: This can be accelerated by caching the latents and then just appending the generated latent after the conditioning frames, dropping the first. 
    os.makedirs(output_dir, exist_ok=True)
    frames = []
    batch = initial_batch.copy()

    for i in range(num_frames):
        print(f"Generating frame {i+1}/{num_frames}")
        
        # Generate the next frame
        new_frame = run_inference_img_conditioning_with_params(
            unet,
            vae,
            noise_scheduler,
            action_embedding,
            tokenizer,
            text_encoder,
            batch,
            device=device,
            skip_action_conditioning=skip_action_conditioning,
            do_classifier_free_guidance=do_classifier_free_guidance,
            guidance_scale=guidance_scale,
            num_inference_steps=num_inference_steps,
        )
        
        frames.append(new_frame)
        new_frame.save(os.path.join(output_dir, f"frame_{i:03d}.png"))

        # Update the batch with the new frame
        new_frame_tensor = transforms.ToTensor()(new_frame).unsqueeze(0).unsqueeze(0)
        batch["pixel_values"] = torch.cat([batch["pixel_values"][:, 1:], new_frame_tensor], dim=1)

        # Generate a random action
        new_action = torch.tensor([[random.randint(0, 17)]])
        batch["input_ids"] = torch.cat([batch["input_ids"][:, 1:], new_action], dim=1)

    # Save as GIF
    frames[0].save(
        os.path.join(output_dir, "rollout.gif"),
        save_all=True,
        append_images=frames[1:],
        duration=100,
        loop=0,
    )

if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="Run inference with customizable parameters")
    parser.add_argument("--skip_image_conditioning", action="store_true", help="Skip image conditioning")
    parser.add_argument("--skip_action_conditioning", action="store_true", help="Skip action conditioning")
    parser.add_argument("--model_folder", type=str, help="Path to the folder containing the model weights")
    args = parser.parse_args()

    skip_image_conditioning = args.skip_image_conditioning
    skip_action_conditioning = args.skip_action_conditioning

    device = torch.device(
        "cuda"
        if torch.cuda.is_available()
        else "mps"
        if torch.backends.mps.is_available()
        else "cpu"
    )

    dataset = load_dataset("P-H-B-D-a16z/ViZDoom-Deathmatch-PPO")
    if not args.model_folder:
        unet, vae, action_embedding, noise_scheduler, tokenizer, text_encoder = get_model(17, skip_image_conditioning=skip_image_conditioning)
    else:
        
        unet, vae, action_embedding, noise_scheduler, tokenizer, text_encoder = get_model(
        17, skip_image_conditioning=args.skip_image_conditioning)
        unet.load_state_dict(load_file(os.path.join(args.model_folder, "unet", "diffusion_pytorch_model.safetensors")))
        vae.load_state_dict(load_file(os.path.join(args.model_folder, "vae", "diffusion_pytorch_model.safetensors")))
        
        # Load scheduler configuration
        with open(os.path.join(args.model_folder, "scheduler", "scheduler_config.json"), "r") as f:
            scheduler_config = json.load(f)
        noise_scheduler = DDIMScheduler.from_config(scheduler_config)
        
        action_embedding.load_state_dict(torch.load(os.path.join(args.model_folder, "action_embedding.pth")))

        # Load embedding info
        embedding_info = torch.load(os.path.join(args.model_folder, "embedding_info.pth"))
        action_embedding.num_embeddings = embedding_info["num_embeddings"]
        action_embedding.embedding_dim = embedding_info["embedding_dim"]

    unet = unet.to(device)
    vae = vae.to(device)
    action_embedding = action_embedding.to(device)
    text_encoder = text_encoder.to(device)

    train_transforms = transforms.Compose(
        [
            transforms.Resize((HEIGHT, WIDTH), interpolation=transforms.InterpolationMode.BILINEAR),
            # transforms.CenterCrop(HEIGHT),
            # transforms.RandomHorizontalFlip() if args.random_flip else transforms.Lambda(lambda x: x),
            transforms.ToTensor(),
            transforms.Normalize([0.5], [0.5]),
        ]
    )

    def preprocess_train(examples):
        images = []
        for image_list in examples["images"]:
            current_images = []
            image_list = [
                Image.open(io.BytesIO(base64.b64decode(img))).convert("RGB")
                for img in image_list
            ]
            for image in image_list:
                current_images.append(train_transforms(image))
            images.append(current_images)

        actions = torch.tensor(examples["actions"]) if isinstance(examples["actions"], list) else examples["actions"]
        return {"pixel_values": images, "input_ids": actions}

    train_dataset = dataset["train"].with_transform(preprocess_train)
    
    def collate_fn(examples):
        # Function to create a black screen tensor
        def create_black_screen(height, width):
            return torch.zeros(3, height, width, dtype=torch.float32)

        if not args.skip_image_conditioning:
            # Process each example
            processed_images = []
            for example in examples:

                # This means you have BUFFER_SIZE conditioning frames + 1 target frame
                processed_images.append(
                    torch.stack(example["pixel_values"][:BUFFER_SIZE + 1]))

            # Stack all examples
            # images has shape: (batch_size, frame_buffer, 3, height, width)
            images = torch.stack(processed_images)
            images = images.to(memory_format=torch.contiguous_format).float()

            # UGLY HACK
            images = no_img_conditioning_augmentation(images, prob=ZERO_OUT_ACTION_CONDITIONING_PROB)
        else:
            images = torch.stack(
                [example["pixel_values"][0] for example in examples])
            images = images.to(memory_format=torch.contiguous_format).float()
        return {
            "pixel_values": images,
            "input_ids": torch.stack([torch.tensor(example["input_ids"][:BUFFER_SIZE+1]) for example in examples]),
        }

    # collate_fn = partial(collate_fn, skip_image_conditioning)

    # DataLoaders creation:
    train_dataloader = torch.utils.data.DataLoader(
        train_dataset,
        shuffle=False,
        collate_fn=collate_fn,
        batch_size=1,
        num_workers=0,
    )

    batch = next(iter(train_dataloader))


    generate_autoregressive_rollout(
        unet,
        vae,
        noise_scheduler,
        action_embedding,
        tokenizer,
        text_encoder,
        batch,
        device=device,
        num_frames=30,  # Adjust this to change the number of frames in the rollout
        skip_action_conditioning=skip_action_conditioning,
        do_classifier_free_guidance=False,
        guidance_scale=1.5,
        num_inference_steps=50,
    )