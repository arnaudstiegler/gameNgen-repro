import argparse
import torch
import random
from sd3.model import load_model
from config_sd import BUFFER_SIZE
from config_sd import CFG_GUIDANCE_SCALE, TRAINING_DATASET_DICT
from diffusers.image_processor import VaeImageProcessor
from dataset import get_single_batch
from run_inference import next_latent, encode_conditioning_frames
import numpy as np
# Action 0: TURN_LEFT
# Action 1: TURN_RIGHT
# Action 2: MOVE_RIGHT
# Action 3: MOVE_RIGHT + TURN_LEFT
# Action 4: MOVE_RIGHT + TURN_RIGHT
# Action 5: MOVE_LEFT
# Action 6: MOVE_LEFT + TURN_LEFT
# Action 7: MOVE_LEFT + TURN_RIGHT
# Action 8: MOVE_FORWARD
# Action 9: MOVE_FORWARD + TURN_LEFT
# Action 10: MOVE_FORWARD + TURN_RIGHT
# Action 11: MOVE_FORWARD + MOVE_RIGHT
# Action 12: MOVE_FORWARD + MOVE_RIGHT + TURN_LEFT
# Action 13: MOVE_FORWARD + MOVE_RIGHT + TURN_RIGHT
# Action 14: MOVE_FORWARD + MOVE_LEFT
# Action 15: MOVE_FORWARD + MOVE_LEFT + TURN_LEFT
# Action 16: MOVE_FORWARD + MOVE_LEFT + TURN_RIGHT
# Action 17: ATTACK

torch.manual_seed(9052924)
np.random.seed(9052924)
random.seed(9052924)


def main(model_folder: str, num_frames: int) -> None:
    device = torch.device(
        "cuda"
        if torch.cuda.is_available()
        else "mps"
        if torch.backends.mps.is_available()
        else "cpu"
    )

    unet, vae, action_embedding, noise_scheduler, _, _ = load_model(
        model_folder, device=device
    )

    batch = get_single_batch(TRAINING_DATASET_DICT["small"])

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
    initial_context = context_latents.squeeze(0)  # [BUFFER_SIZE, 4, 30, 40]
    all_latents = [
        initial_context[i : i + 1] for i in range(initial_context.shape[0])
    ]  # List of [1, 4, 30, 40] tensors
    current_actions = batch["input_ids"].squeeze(0)[:BUFFER_SIZE].to(device)

    # Autoregressive rollout
    for i in range(num_frames):
        print(f"Generating frame {i}")
        # Generate next frame latents
        target_latents = next_latent(
            unet=unet,
            vae=vae,
            noise_scheduler=noise_scheduler,
            action_embedding=action_embedding,
            context_latents=context_latents,
            device=device,
            skip_action_conditioning=False,
            do_classifier_free_guidance=False,
            guidance_scale=CFG_GUIDANCE_SCALE,
            num_inference_steps=50,
            actions=current_actions.unsqueeze(0),
        )

        # Append new latents
        all_latents.append(target_latents)

        # Generate random action for next step
        next_action = torch.tensor([random.randint(0, 17)], device=device)
        # next_action = torch.tensor([8]).to(device)
        current_actions = torch.cat([current_actions[1:], next_action])

        # Update context latents using sliding window
        # Always take exactly BUFFER_SIZE most recent frames
        latest_frames = torch.cat(
            all_latents[-BUFFER_SIZE:], dim=0
        )  # [BUFFER_SIZE, 4, 30, 40]
        context_latents = latest_frames.unsqueeze(
            0
        )  # Add batch dimension [1, BUFFER_SIZE, 4, 30, 40]

    # Decode all latents to images
    all_images = []
    for latent in all_latents[BUFFER_SIZE:]:  # Skip the initial context frames
        image = vae.decode(latent / vae.config.scaling_factor, return_dict=False)[0]

        image = image_processor.postprocess(
            image.detach(), output_type="pil", do_denormalize=[True] * image.shape[0]
        )[0]
        all_images.append(image)

    # Save as GIF
    all_images[0].save(
        "autoregressive_rollout.gif",
        save_all=True,
        append_images=all_images[1:],
        duration=100,  # 100ms per frame
        loop=0,
    )


if __name__ == "__main__":
    # TODO: extract all that to a main function
    parser = argparse.ArgumentParser(
        description="Run inference with customizable parameters"
    )
    parser.add_argument(
        "--model_folder",
        type=str,
        help="Path to the folder containing the model weights",
    )
    parser.add_argument(
        "--num_frames", type=int, help="Number of frames to generate", default=20
    )
    args = parser.parse_args()

    main(model_folder=args.model_folder, num_frames=args.num_frames)
