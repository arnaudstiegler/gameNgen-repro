import argparse
import torch
import torchvision.transforms as transforms
from PIL import Image
import random
import os
from run_inference import run_inference_img_conditioning_with_params
from sd3.model import load_model
import json
import io
import base64
import numpy as np
from datasets import load_dataset
from config_sd import HEIGHT, WIDTH, BUFFER_SIZE, ZERO_OUT_ACTION_CONDITIONING_PROB
from data_augmentation import no_img_conditioning_augmentation
from sd3.model import get_model



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
        unet, vae, action_embedding, noise_scheduler, tokenizer, text_encoder = load_model(args.model_folder, 17)

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