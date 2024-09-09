from diffusers import AutoencoderKL, DDIMScheduler, UNet2DConditionModel
from huggingface_hub import hf_hub_download
from config_sd import REPO_NAME
from diffusers import (
    AutoencoderKL,
    DDIMScheduler,
    UNet2DConditionModel,
    DDPMScheduler,
    StableDiffusionPipeline,
)
from transformers import CLIPTextModel, CLIPTokenizer, CLIPImageProcessor
import torch
from typing import List
from diffusers.utils.torch_utils import randn_tensor
from tqdm import tqdm
from diffusers.image_processor import VaeImageProcessor
import numpy as np
import random
from safetensors import safe_open
import torch


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


def run_inference():
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
    prompt = ["Spiderman riding a horse"]
    batch_size = len(prompt)
    num_images_per_prompt = 1
    do_classifier_free_guidance = True
    negative_prompt = None
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
        # prompt_embeds, negative_prompt_embeds = encode_prompt(
        #     tokenizer,
        #     text_encoder,
        #     prompt,
        #     negative_prompt,
        #     batch_size,
        #     device,
        #     do_classifier_free_guidance,
        #     num_images_per_prompt,
        # )

        # if do_classifier_free_guidance:
        #     prompt_embeds = torch.cat([negative_prompt_embeds, prompt_embeds])

        # 4. Prepare timesteps
        noise_scheduler.set_timesteps(num_inference_steps, device=device)
        timesteps = noise_scheduler.timesteps

        # # 5. Prepare latent variables
        num_channels_latents = unet.config.in_channels
        latents = get_latents(
            noise_scheduler,
            batch_size,
            height,
            width,
            num_channels_latents,
            vae_scale_factor,
            device,
            dtype,
        )

        # 7. Denoising loop
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
            # TODO: this is still buggy
            latents = noise_scheduler.step(
                noise_pred, t, latents, generator=None, return_dict=False
            )[0]

    image = vae.decode(
        latents / vae.config.scaling_factor, return_dict=False, generator=generator
    )[0]

    image = image_processor.postprocess(
        image.detach(), output_type="pil", do_denormalize=[True] * image.shape[0]
    )
    image[0].save(f"inference_test/image_steps{t}.png")


if __name__ == "__main__":
    run_inference()
