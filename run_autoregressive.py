import argparse
import base64
import io
import random
from typing import List, Optional

import numpy as np
import torch
from datasets import load_dataset
from diffusers import DDIMScheduler
from torchvision import transforms
import os
from sd3.model import get_model
from functools import partial
from safetensors.torch import load_file
import json
import random
import os
from run_inference import run_inference_img_conditioning_with_params
from train_text_to_image import preprocess_train


torch.manual_seed(9052924)
np.random.seed(9052924)
random.seed(9052924)

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
            skip_action_conditioning=True, # Skip action conditioning should be true for autoregressive
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


def main():
    device = torch.device(
        "cuda"
        if torch.cuda.is_available()
        else "mps"
        if torch.backends.mps.is_available()
        else "cpu"
    )

    dataset = load_dataset("P-H-B-D-a16z/ViZDoom-Deathmatch-PPO")
    if not args.model_folder:
        unet, vae, action_embedding, noise_scheduler, tokenizer, text_encoder = get_model(17, skip_image_conditioning=True)
    else:
        
        unet, vae, action_embedding, noise_scheduler, tokenizer, text_encoder = get_model(
        17, skip_image_conditioning=True)
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

    train_dataset = dataset["train"].with_transform(preprocess_train)

    collate_fn = partial(collate_fn, True) # Skip image conditioning should be true for autoregressive

    dataloader = torch.utils.data.DataLoader(
        train_dataset,
        shuffle=False,
        collate_fn=collate_fn,
        batch_size=1,
        num_workers=0,
    )

    batch = next(iter(dataloader))

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
        do_classifier_free_guidance=False,
        guidance_scale=1.5,
        num_inference_steps=50,
    )

if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="Run inference with customizable parameters")
    parser.add_argument("--model_folder", type=str, help="Path to the folder containing the model weights")
    args = parser.parse_args()
    main(args.model_folder)