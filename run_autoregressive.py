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
from PIL import Image
from loguru import logger


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

'''
0: ?
1: right
2: move right
3: unclear
4: ?
5: move left
6: turn left
7: turn right?

'''

torch.manual_seed(9052924)
np.random.seed(9052924)
random.seed(9052924)


def generate_rollout(unet, vae, action_embedding, noise_scheduler, image_processor, actions: list[int], initial_frame_context: torch.Tensor, initial_action_context: torch.Tensor) -> list[Image]:
    device = unet.device
    all_latents = []
    current_actions = initial_action_context
    context_latents = initial_frame_context
    
    for i in range(len(actions)):
        print(f"Generating frame {i}")
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
        current_actions = torch.cat([current_actions[(-BUFFER_SIZE+1):], torch.tensor([actions[i]]).to(device)])

        # Update context latents using sliding window
        # Always take exactly BUFFER_SIZE most recent frames
        context_latents = torch.cat(
            [context_latents[(-BUFFER_SIZE+1):], target_latents], dim=0
        )

    # Decode all latents to images
    all_images = []
    for latent in all_latents[BUFFER_SIZE:]:  # Skip the initial context frames
        image = vae.decode(latent / vae.config.scaling_factor, return_dict=False)[0]

        image = image_processor.postprocess(
            image.detach(), output_type="pil", do_denormalize=[True] * image.shape[0]
        )[0]
        all_images.append(image)
    return all_images


def main(model_folder: str) -> None:
    device = torch.device(
        "cuda"
        if torch.cuda.is_available()
        else "mps"
        if torch.backends.mps.is_available()
        else "cpu"
    )
    
    scenarios = {

        # 'forward': [8]*15,
        # 'forward_right_attack': [8, 4, 8, 4, 8, 4, 8, 4, 8, 4, 8, 4, 8, 4, 8, 4, 8, 4, 8, 17],
        # 'forward_left_attack': [8, 12, 8, 12, 8, 12, 8, 12, 8, 12, 8, 12, 8, 12, 8, 12, 8, 12, 8, 17],
        'forward_attack_forward_attack': [8, 8, 8, 8, 8, 17, 17, 17, 17, 8, 8, 8, 8, 8, 8, 8, 8, 8, 4, 4, 4, 4, 8, 8, 8, 8, 8, 17, 17, 17, 17, 8, 8, 8, 8, 8, 8, 8, 8, 8, 4, 4, 4, 4, 8, 8, 8, 8, 8, 17, 17, 17, 17, 8, 8, 8, 8, 8, 8, 8, 8, 8, 4, 4, 4, 4, 8, 8, 8, 8, 8, 17, 17, 17, 17, 8, 8, 8, 8, 8, 8, 8, 8, 8, 4, 4, 4, 4, 8, 8, 8, 8, 8, 17, 17, 17, 17, 8],
    }

    batch = get_single_batch(TRAINING_DATASET_DICT["small"])
    for scenario_name, actions in scenarios.items():
        unet, vae, action_embedding, noise_scheduler, _, _ = load_model(
            model_folder, device=device
    )
        logger.info(f"Generating rollout forscenario {scenario_name}")

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
            f"rollout_{scenario_name}.gif",
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
    args = parser.parse_args()

    main(model_folder=args.model_folder)
