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
from sd3.model import get_model
from functools import partial
from torch.amp import autocast
from safetensors.torch import load_file
import json
from train_text_to_image import preprocess_train
from inference_utils import run_inference_with_params, run_inference_img_conditioning_with_params



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





def main(model_folder: str, skip_action_conditioning: bool, skip_image_conditioning: bool):
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

    collate_fn = partial(collate_fn, skip_image_conditioning)

    dataloader = torch.utils.data.DataLoader(
        train_dataset,
        shuffle=False,
        collate_fn=collate_fn,
        batch_size=1,
        num_workers=0,
    )

    batch = next(iter(dataloader))
    if skip_image_conditioning:
        run_inference_with_params(
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
        )
    else:
        run_inference_img_conditioning_with_params(
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
            guidance_scale=1.5,
            skip_action_conditioning=skip_action_conditioning,
        )

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run inference with customizable parameters")
    parser.add_argument("--model_folder", type=str, help="Path to the folder containing the model weights")
    args = parser.parse_args()
    main(args.model_folder)
