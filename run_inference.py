from diffusers import AutoencoderKL, DDIMScheduler, UNet2DConditionModel
from huggingface_hub import hf_hub_download
from config_sd import REPO_NAME
from diffusers import AutoencoderKL, DDIMScheduler, UNet2DConditionModel, DDPMScheduler
from transformers import CLIPTextModel, CLIPTokenizer, CLIPImageProcessor
import torch
from typing import List, Optional
from diffusers.utils.torch_utils import randn_tensor
from tqdm import tqdm
from diffusers.image_processor import VaeImageProcessor
import numpy as np
import random
from safetensors import safe_open
import torch
from datasets import load_dataset
from config_sd import BUFFER_SIZE
from torchvision import transforms
from PIL import Image
from config_sd import HEIGHT, WIDTH
import argparse


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

from PIL import Image
import base64
import io

def encode_conditioning_frames(
    vae: AutoencoderKL, conditioning_frames: list[Image.Image], dtype: torch.dtype
):
    to_tensor = transforms.ToTensor()

    conditioning_frames = [Image.open(io.BytesIO(base64.b64decode(img))) for img in conditioning_frames]
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

from datasets import load_dataset, DatasetDict
def run_inference():
    # Load the dataset for the conditioning frames
    dataset = load_dataset("P-H-B-D-a16z/ViZDoom-Deathmatch-PPO")
    # Create a train-test split
    dataset = dataset["train"].train_test_split(test_size=0.2)

    # Create a train-test split
    train_test_dataset = dataset["train"].train_test_split(test_size=0.1, seed=42)

    # Create a new DatasetDict with train and test splits
    dataset = DatasetDict({
        "train": train_test_dataset["train"],
        "test": train_test_dataset["test"]
    })
    # TODO: should it be BUFFER_SIZE-1 or BUFFER_SIZE?
    conditioning_frames = dataset["test"][0]["images"][: BUFFER_SIZE - 1]

    device = torch.device(
        "cuda"
        if torch.cuda.is_available()
        else "mps"
        if torch.backends.mps.is_available()
        else "cpu"
    )

    # unet = (
    #     UNet2DConditionModel.from_pretrained(REPO_NAME, subfolder="unet")
    #     .eval()
    #     .to(device)
    # )
    # vae = AutoencoderKL.from_pretrained(REPO_NAME, subfolder="vae").eval().to(device)
    # noise_scheduler = DDIMScheduler.from_pretrained(
    #     REPO_NAME, subfolder="noise_scheduler"
    # )
    # Load the trained model components
    import os 

    unet = UNet2DConditionModel.from_pretrained(args.output_dir, subfolder="unet").eval().to(device)
    vae = AutoencoderKL.from_pretrained(args.output_dir, subfolder="vae").eval().to(device)
    noise_scheduler = DDIMScheduler.from_pretrained(args.output_dir, subfolder="scheduler")

    # Load embedding dimensions
    embedding_info = torch.load(os.path.join(args.output_dir, "embedding_info.pth"))
    num_embeddings = embedding_info["num_embeddings"]
    embedding_dim = embedding_info["embedding_dim"]

    # Load action embedding
    action_embedding = torch.nn.Embedding(num_embeddings, embedding_dim)
    action_embedding.load_state_dict(torch.load(os.path.join(args.output_dir, "action_embedding.pth")))
    action_embedding.eval().to(device)

    file_path = hf_hub_download(
        repo_id=REPO_NAME, filename="action_embedding_model.safetensors"
    )
    # action_embedding = (
    #     read_action_embedding_from_safetensors(file_path).eval().to(device)
    # )

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
    # height = unet.config.sample_size * vae_scale_factor
    # width = unet.config.sample_size * vae_scale_factor
    height = HEIGHT
    width = WIDTH
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

        conditioning_frames_latents = encode_conditioning_frames(
            vae, conditioning_frames, dtype
        )

        # TODO: careful that the last frame should be the noisy frame
        latents = torch.cat([conditioning_frames_latents, latents], dim=0)

        # 7. Denoising loop
        for _, t in tqdm(enumerate(timesteps), total=num_inference_steps):
            # expand the latents if we are doing classifier free guidance
            latents = latents.view(1, -1, 30, 40)
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
            # TODO: here, we gotta step the scheduler only on the last frame, and recover the conditioning frames afterwards
            reshaped_frames = latents.reshape(10, 4, 30, 40)
            last_frame = reshaped_frames[-1]
            denoised_last_frame = noise_scheduler.step(
                noise_pred, t, last_frame, generator=None, return_dict=False
            )[0]
            reshaped_frames[-1] = denoised_last_frame
            latents = reshaped_frames.reshape(1, -1, 30, 40)

    last_frame_latent = latents.reshape(10, 4, 30, 40)[-1].unsqueeze(0)

    image = vae.decode(
        last_frame_latent / vae.config.scaling_factor, return_dict=False, generator=generator
    )[0]

    image = image_processor.postprocess(
        image.detach(), output_type="pil", do_denormalize=[True] * image.shape[0]
    )
    image[0].save(f"./image_steps{t}.png")

def parse_args():
    parser = argparse.ArgumentParser(description="Run inference with trained model")
    parser.add_argument(
        "--output_dir",
        type=str,
        default="sd-model-finetuned",
        help="The directory where the trained model is saved",
    )
    return parser.parse_args()

def run_inference_with_params(unet, vae, noise_scheduler, action_embedding, batch, device, num_inference_steps=50):
    with torch.no_grad():
        # Prepare inputs
        images = batch["images"]
        actions = batch["actions"]
        
        # Reshape and encode conditioning frames
        batch_size, buffer_size, channels, height, width = images.shape
        conditioning_frames = images[:, :BUFFER_SIZE-1].reshape(-1, channels, height, width)
        conditioning_frames_latents = vae.encode(conditioning_frames.to(device)).latent_dist.sample()
        conditioning_frames_latents = conditioning_frames_latents * vae.config.scaling_factor
        
        # Reshape conditioning_frames_latents back to include batch and buffer dimensions
        _, latent_channels, latent_height, latent_width = conditioning_frames_latents.shape
        conditioning_frames_latents = conditioning_frames_latents.reshape(
            batch_size, BUFFER_SIZE-1, latent_channels, latent_height, latent_width
        )

        # Generate initial noise for the last frame
        latents = torch.randn((batch_size, latent_channels, latent_height, latent_width), device=device)
        latents = latents * noise_scheduler.init_noise_sigma

        # Concatenate conditioning frames with the noisy last frame
        latents = torch.cat([conditioning_frames_latents, latents.unsqueeze(1)], dim=1)

        # Prepare timesteps
        noise_scheduler.set_timesteps(num_inference_steps, device=device)
        timesteps = noise_scheduler.timesteps

        # Denoising loop
        for _, t in tqdm(enumerate(timesteps), total=num_inference_steps):
            latents = latents.view(batch_size, -1, latent_height, latent_width)
            latent_model_input = noise_scheduler.scale_model_input(latents, t)
            
            # Prepare action embeddings
            action_hidden_states = action_embedding(actions.to(device))

            # Predict noise
            noise_pred = unet(
                latent_model_input,
                t,
                encoder_hidden_states=action_hidden_states,
                return_dict=False,
            )[0]

            # Perform denoising step on the last frame only
            reshaped_frames = latents.reshape(batch_size, BUFFER_SIZE, latent_channels, latent_height, latent_width)
            last_frame = reshaped_frames[:, -1]
            denoised_last_frame = noise_scheduler.step(
                noise_pred, t, last_frame, return_dict=False
            )[0]
            reshaped_frames[:, -1] = denoised_last_frame
            latents = reshaped_frames.reshape(batch_size, -1, latent_height, latent_width)

        # Decode the last frame
        last_frame_latent = latents.reshape(batch_size, BUFFER_SIZE, latent_channels, latent_height, latent_width)[:, -1]
        image = vae.decode(last_frame_latent / vae.config.scaling_factor, return_dict=False)[0]

        # Post-process the image
        image = (image / 2 + 0.5).clamp(0, 1)
        image = image.cpu().permute(0, 2, 3, 1).numpy()
        image = (image * 255).round().astype("uint8")
        image = [Image.fromarray(img).convert("RGB") for img in image]

    return image[0]  # Return the first (and only) image
# if __name__ == "__main__":
#     args = parse_args()
#     run_inference()
