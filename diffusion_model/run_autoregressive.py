import argparse
import random

import numpy as np
import torch
from diffusers.image_processor import VaeImageProcessor
from PIL import Image
from tqdm import tqdm

from config_sd import BUFFER_SIZE, CFG_GUIDANCE_SCALE, TRAINING_DATASET_DICT
from dataset import EpisodeDataset, collate_fn
from run_inference import (
    decode_and_postprocess,
    encode_conditioning_frames,
    next_latent,
)
from model import load_model

"""Action mapping for the Doom environment:
Built action space of size 18 from buttons [
    <Button.ATTACK: 0> 
    <Button.MOVE_FORWARD: 13> 
    <Button.MOVE_LEFT: 11>
    <Button.MOVE_RIGHT: 10>
    <Button.TURN_RIGHT: 14>
    <Button.TURN_LEFT: 15>
    ]
"""


torch.manual_seed(9052924)
np.random.seed(9052924)
random.seed(9052924)

EPISODE_LENGTH = 30


def generate_rollout(
    unet,
    vae,
    action_embedding,
    noise_scheduler,
    image_processor,
    actions: list[int],
    initial_frame_context: torch.Tensor,
    initial_action_context: torch.Tensor,
) -> list[Image]:
    device = unet.device
    all_latents = []
    current_actions = initial_action_context
    context_latents = initial_frame_context

    for i in tqdm(range(len(actions))):
        # Generate next frame latents
        target_latents = next_latent(
            unet=unet,
            vae=vae,
            noise_scheduler=noise_scheduler,
            action_embedding=action_embedding,
            context_latents=context_latents.unsqueeze(0),
            device=device,
            skip_action_conditioning=False,
            do_classifier_free_guidance=False,
            guidance_scale=CFG_GUIDANCE_SCALE,
            num_inference_steps=50,
            actions=current_actions.unsqueeze(0),
        )
        all_latents.append(target_latents)
        current_actions = torch.cat(
            [
                current_actions[(-BUFFER_SIZE + 1) :],
                torch.tensor([actions[i]]).to(device),
            ]
        )

        # Update context latents using sliding window
        # Always take exactly BUFFER_SIZE most recent frames
        context_latents = torch.cat(
            [context_latents[(-BUFFER_SIZE + 1) :], target_latents], dim=0
        )

    # Decode all latents to images
    all_images = []
    for latent in all_latents:  # Skip the initial context frames
        all_images.append(
            decode_and_postprocess(
                vae=vae, image_processor=image_processor, latents=latent
            )
        )
    return all_images


def main(model_folder: str) -> None:
    device = torch.device(
        "cuda"
        if torch.cuda.is_available()
        else "mps"
        if torch.backends.mps.is_available()
        else "cpu"
    )

    dataset = EpisodeDataset(TRAINING_DATASET_DICT["small"])
    start_indices = [
        random.randint(0, len(dataset) - EPISODE_LENGTH) for _ in range(20)
    ]

    for start_idx in start_indices:
        # Collate to ge the right tensor dims
        batch = collate_fn([dataset[start_idx]])
        actions = [
            dataset[i]["input_ids"][-1].item()
            for i in range(
                start_idx + BUFFER_SIZE, start_idx + BUFFER_SIZE + EPISODE_LENGTH
            )
        ]

        unet, vae, action_embedding, noise_scheduler, _, _ = load_model(
            model_folder, device=device
        )

        vae_scale_factor = 2 ** (len(vae.config.block_out_channels) - 1)
        image_processor = VaeImageProcessor(vae_scale_factor=vae_scale_factor)

        # Encode initial context frames
        context_latents = encode_conditioning_frames(
            vae,
            images=batch["pixel_values"],
            vae_scale_factor=vae_scale_factor,
            dtype=torch.float32,
        )

        # Store all generated latents - split context frames into individual tensors
        initial_frame_context = context_latents.squeeze(0)  # [BUFFER_SIZE, 4, 30, 40]
        initial_action_context = batch["input_ids"].squeeze(0)[:BUFFER_SIZE].to(device)

        all_images = generate_rollout(
            unet=unet,
            vae=vae,
            action_embedding=action_embedding,
            noise_scheduler=noise_scheduler,
            image_processor=image_processor,
            actions=actions,
            initial_frame_context=initial_frame_context,
            initial_action_context=initial_action_context,
        )

        all_images[0].save(
            f"rollouts/rollout_{start_idx}.gif",
            save_all=True,
            append_images=all_images[1:],
            duration=100,  # 100ms per frame
            loop=0,
        )


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

    main(model_folder=args.model_folder)
