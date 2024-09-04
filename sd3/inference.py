from diffusers import AutoencoderKL, DDIMScheduler, UNet2DConditionModel, DDPMScheduler
from transformers import CLIPTextModel, CLIPTokenizer, CLIPImageProcessor
import torch
from typing import List
from diffusers.utils.torch_utils import randn_tensor
from tqdm import tqdm
from diffusers.image_processor import VaeImageProcessor


repo_name = 'CompVis/stable-diffusion-v1-4'

unet = UNet2DConditionModel.from_pretrained(repo_name, subfolder='unet')
vae = AutoencoderKL.from_pretrained(repo_name, subfolder='vae')
noise_scheduler = DDPMScheduler.from_pretrained(repo_name, subfolder="scheduler")
# noise_scheduler = DDIMScheduler()
tokenizer = CLIPTokenizer.from_pretrained(
    repo_name, subfolder="tokenizer"
)
text_encoder = CLIPTextModel.from_pretrained(
    repo_name, subfolder="text_encoder"
)


# Defining all config-related variables here
prompt = ['Spiderman riding a horse']
batch_size = len(prompt)
device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
num_images_per_prompt = 1
do_classifier_free_guidance = True
negative_prompt = None
num_inference_steps = 50
generator = torch.Generator(device=device)
dtype = torch.float32
vae_scale_factor = 2 ** (len(vae.config.block_out_channels) - 1)
height = unet.config.sample_size * vae_scale_factor
width = unet.config.sample_size * vae_scale_factor
guidance_scale = 1

image_processor = VaeImageProcessor(vae_scale_factor=vae_scale_factor)


def encode_prompt(prompts: List[str], negative_prompt: List[str], batch_size: int, device: str, do_classifier_free_guidance: bool):
    assert isinstance(prompts, list), f'Expected list but received: {type(prompts)}'
    text_inputs = tokenizer(
                prompts,
                padding="max_length",
                max_length=tokenizer.model_max_length,
                truncation=True,
                return_tensors="pt",
            )
    prompt_embeds = text_encoder(text_inputs.input_ids.to(device), attention_mask=text_inputs.attention_mask.to(device))
    prompt_embeds = prompt_embeds[0]

    if do_classifier_free_guidance:
        uncond_tokens = [""] * batch_size if not negative_prompt else negative_prompt
        uncond_input = tokenizer(
                uncond_tokens,
                padding="max_length",
                max_length=tokenizer.model_max_length,
                truncation=True,
                return_tensors="pt",
            )
        negative_prompt_embeds = text_encoder(uncond_input.input_ids.to(device), attention_mask = uncond_input.input_ids.to(device))
        negative_prompt_embeds = negative_prompt_embeds[0]

    return prompt_embeds, negative_prompt_embeds

def get_latents(generator: torch.Generator, batch_size: int, height: int, width: int, dtype = torch.float32):
    shape = (
        batch_size,
        num_channels_latents,
        int(height) // vae_scale_factor,
        int(width) // vae_scale_factor,
    )
    if isinstance(generator, list) and len(generator) != batch_size:
        raise ValueError(
            f"You have passed a list of generators of length {len(generator)}, but requested an effective batch"
            f" size of {batch_size}. Make sure the batch size matches the length of the generators."
        )

    latents = randn_tensor(shape, generator=generator, device=device, dtype=dtype)

    # scale the initial noise by the standard deviation required by the scheduler
    latents = latents * noise_scheduler.init_noise_sigma
    return latents

with torch.no_grad():
    prompt_embeds, negative_prompt_embeds = encode_prompt(
        prompt,
        negative_prompt,
        batch_size,
        device,
        do_classifier_free_guidance,
    )

    # For classifier free guidance, we need to do two forward passes.
    # Here we concatenate the unconditional and text embeddings into a single batch
    # to avoid doing two forward passes
    if do_classifier_free_guidance:
        prompt_embeds = torch.cat([negative_prompt_embeds, prompt_embeds])

    # 4. Prepare timesteps
    noise_scheduler.set_timesteps(num_inference_steps, device=device)
    timesteps = noise_scheduler.timesteps

    # # 5. Prepare latent variables
    num_channels_latents = unet.config.in_channels
    latents = get_latents(
        generator, batch_size, height, width, dtype
    )

    # # 6.2 Optionally get Guidance Scale Embedding
    # timestep_cond = None
    # if self.unet.config.time_cond_proj_dim is not None:
    #     guidance_scale_tensor = torch.tensor(self.guidance_scale - 1).repeat(batch_size * num_images_per_prompt)
    #     timestep_cond = self.get_guidance_scale_embedding(
    #         guidance_scale_tensor, embedding_dim=self.unet.config.time_cond_proj_dim
    #     ).to(device=device, dtype=latents.dtype)

    # 7. Denoising loop
    num_warmup_steps = len(timesteps) - num_inference_steps * noise_scheduler.order
    _num_timesteps = len(timesteps)

    for i, t in tqdm(enumerate(timesteps), total=_num_timesteps):
        # expand the latents if we are doing classifier free guidance
        latent_model_input = torch.cat([latents] * 2) if do_classifier_free_guidance else latents
        latent_model_input = noise_scheduler.scale_model_input(latent_model_input, t)

        # predict the noise residual
        noise_pred = unet(
            latent_model_input,
            t,
            encoder_hidden_states=prompt_embeds,
            timestep_cond=None,
            return_dict=False,
        )[0]

        # perform guidance
        if do_classifier_free_guidance:
            noise_pred_uncond, noise_pred_text = noise_pred.chunk(2)
            noise_pred = noise_pred_uncond + guidance_scale * (noise_pred_text - noise_pred_uncond)

        # TODO: not sure what this does
        # if do_classifier_free_guidance and self.guidance_rescale > 0.0:
        #     # Based on 3.4. in https://arxiv.org/pdf/2305.08891.pdf
        #     noise_pred = rescale_noise_cfg(noise_pred, noise_pred_text, guidance_rescale=self.guidance_rescale)

        # compute the previous noisy sample x_t -> x_t-1
        latents = noise_scheduler.step(noise_pred, t, latents, return_dict=False)[0]

        # if callback_on_step_end is not None:
        #     callback_kwargs = {}
        #     for k in callback_on_step_end_tensor_inputs:
        #         callback_kwargs[k] = locals()[k]
        #     callback_outputs = callback_on_step_end(self, i, t, callback_kwargs)

        #     latents = callback_outputs.pop("latents", latents)
        #     prompt_embeds = callback_outputs.pop("prompt_embeds", prompt_embeds)
        #     negative_prompt_embeds = callback_outputs.pop("negative_prompt_embeds", negative_prompt_embeds)

        # call the callback, if provided
        # if i == len(timesteps) - 1 or ((i + 1) > num_warmup_steps and (i + 1) % self.scheduler.order == 0):
        #     if callback is not None and i % callback_steps == 0:
        #         step_idx = i // getattr(self.scheduler, "order", 1)
        #         callback(step_idx, t, latents)

    # if not output_type == "latent":
    #     image = self.vae.decode(latents / self.vae.config.scaling_factor, return_dict=False, generator=generator)[
    #         0
    #     ]
    #     image, has_nsfw_concept = self.run_safety_checker(image, device, prompt_embeds.dtype)
    # else:
    #     image = latents
    #     has_nsfw_concept = None

    # if has_nsfw_concept is None:
    #     do_denormalize = [True] * image.shape[0]
    # else:
    #     do_denormalize = [not has_nsfw for has_nsfw in has_nsfw_concept]

        image = vae.decode(latents / vae.config.scaling_factor, return_dict=False, generator=generator)[
            0
        ]

        image = image_processor.postprocess(image.detach(), output_type='pil', do_denormalize=[True] * image.shape[0])
        image[0].save(f'inference_test/image_steps{t}.png')