import os
import torch
import textwrap
from diffusers import AutoencoderKL, UNet2DConditionModel, DDIMScheduler
from transformers import CLIPTokenizer, CLIPTextModel
from huggingface_hub import hf_hub_download
from config_sd import BUFFER_SIZE, PRETRAINED_MODEL_NAME_OR_PATH
from utils import NUM_BUCKETS
from huggingface_hub import upload_folder
from diffusers.utils.hub_utils import load_or_create_model_card, populate_model_card
from safetensors.torch import save_file, load_file
import json



def get_ft_vae_decoder() -> AutoencoderKL:
    """
    Based on the original GameNGen code, the vae decoder is finetuned on images from the
    training set to improve the quality of the images.
    """ 
    return AutoencoderKL.from_pretrained("arnaudstiegler/game-n-gen-vae-finetuned")

def get_model(
    action_embedding_dim: int, skip_image_conditioning: bool = False
) -> tuple[
    UNet2DConditionModel,
    AutoencoderKL,
    torch.nn.Embedding,
    DDIMScheduler,
    CLIPTokenizer,
    CLIPTextModel,
]:
    """
    Args:
        action_embedding_dim: the dimension of the action embedding, i.e the number of possible actions + 1 (do nothing action)
        skip_image_conditioning: whether to skip image conditioning
    """

    # This will be used to encode the actions
    action_embedding = torch.nn.Embedding(
        num_embeddings=action_embedding_dim + 1, embedding_dim=768
    )
    torch.nn.init.normal_(action_embedding.weight, mean=0.0, std=0.02)

    # DDIM scheduler allows for v-prediction and less sampling steps
    noise_scheduler = DDIMScheduler.from_pretrained(
        PRETRAINED_MODEL_NAME_OR_PATH, subfolder="scheduler"
    )
    # This is what the paper uses
    noise_scheduler.register_to_config(prediction_type="v_prediction")

    vae = get_ft_vae_decoder()

    unet = UNet2DConditionModel.from_pretrained(
        PRETRAINED_MODEL_NAME_OR_PATH, subfolder="unet"
    )
    # There are 10 noise buckets total
    unet.register_to_config(num_class_embeds=NUM_BUCKETS)
    # We do not use .add_module() because the class_embedding is already initialized as None
    unet.class_embedding = torch.nn.Embedding(
        NUM_BUCKETS, unet.time_embedding.linear_2.out_features
    )

    tokenizer = CLIPTokenizer.from_pretrained(
        PRETRAINED_MODEL_NAME_OR_PATH, subfolder="tokenizer"
    )
    text_encoder = CLIPTextModel.from_pretrained(
        PRETRAINED_MODEL_NAME_OR_PATH, subfolder="text_encoder"
    )

    if not skip_image_conditioning:
        # This is to accomodate concatenating previous frames in the channels dimension
        new_in_channels = 4 * (BUFFER_SIZE + 1)
        new_conv_in = torch.nn.Conv2d(
            new_in_channels, 320, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)
        )
        torch.nn.init.xavier_uniform_(new_conv_in.weight)
        torch.nn.init.zeros_(new_conv_in.bias)

        # Replace the conv_in layer
        unet.conv_in = new_conv_in
        # Have to account for BUFFER SIZE conditioning frames + 1 for the noise
        unet.config["in_channels"] = new_in_channels

    unet.requires_grad_(True)
    vae.requires_grad_(False)
    text_encoder.requires_grad_(False)
    return unet, vae, action_embedding, noise_scheduler, tokenizer, text_encoder


def load_embedding_info_dict(model_folder: str) -> dict:
    if os.path.exists(model_folder):
        with open(os.path.join(model_folder, "embedding_info.json"), "r") as f:
            embedding_info = json.load(f)
    else:
        file_path = hf_hub_download(
            repo_id=model_folder, filename="embedding_info.json", repo_type="model"
        )
        with open(file_path, "r") as f:
            embedding_info = json.load(f)
    return embedding_info


def load_action_embedding(
    model_folder: str, action_num_embeddings: int
) -> torch.nn.Embedding:
    action_embedding = torch.nn.Embedding(
        num_embeddings=action_num_embeddings, embedding_dim=768
    )
    if os.path.exists(model_folder):
        action_embedding.load_state_dict(
            load_file(os.path.join(model_folder, "action_embedding_model.safetensors"))
        )
    else:
        file_path = hf_hub_download(
            repo_id=model_folder,
            filename="action_embedding_model.safetensors",
            repo_type="model",
        )
        action_embedding.load_state_dict(load_file(file_path))
    return action_embedding


def load_model(
    model_folder: str, device: torch.device | None = None
) -> tuple[
    UNet2DConditionModel,
    AutoencoderKL,
    torch.nn.Embedding,
    DDIMScheduler,
    CLIPTokenizer,
    CLIPTextModel,
]:
    """
    Load a model from the hub

    Args:
        model_folder: the folder to load the model from, can be a model id or a local folder
    """
    embedding_info = load_embedding_info_dict(model_folder)
    action_embedding = load_action_embedding(
        model_folder=model_folder,
        action_num_embeddings=embedding_info["num_embeddings"],
    )

    noise_scheduler = DDIMScheduler.from_pretrained(
        model_folder, subfolder="noise_scheduler"
    )

    vae = get_ft_vae_decoder()
    unet = UNet2DConditionModel.from_pretrained(model_folder, subfolder="unet")

    assert (
        noise_scheduler.config.prediction_type == "v_prediction"
    ), "Noise scheduler prediction type should be 'v_prediction'"
    assert (
        unet.config.num_class_embeds == NUM_BUCKETS
    ), f"UNet num_class_embeds should be {NUM_BUCKETS}"

    # Unaltered
    tokenizer = CLIPTokenizer.from_pretrained(
        PRETRAINED_MODEL_NAME_OR_PATH, subfolder="tokenizer"
    )
    text_encoder = CLIPTextModel.from_pretrained(
        PRETRAINED_MODEL_NAME_OR_PATH, subfolder="text_encoder"
    )
    
    if device:
        unet = unet.to(device)
        vae = vae.to(device)
        action_embedding = action_embedding.to(device)
        text_encoder = text_encoder.to(device)
    
    return unet, vae, action_embedding, noise_scheduler, tokenizer, text_encoder


def save_model(
    output_dir: str,
    unet: UNet2DConditionModel,
    vae: AutoencoderKL,
    noise_scheduler: DDIMScheduler,
    action_embedding: torch.nn.Embedding,
) -> None:
    unet.save_pretrained(os.path.join(output_dir, "unet"))
    vae.save_pretrained(os.path.join(output_dir, "vae"))
    noise_scheduler.save_pretrained(os.path.join(output_dir, "noise_scheduler"))
    save_file(
        action_embedding.state_dict(),
        os.path.join(output_dir, "action_embedding_model.safetensors"),
    )

    # Save embedding dimensions
    embedding_info = {
        "num_embeddings": action_embedding.num_embeddings,
        "embedding_dim": action_embedding.embedding_dim,
    }

    with open(os.path.join(output_dir, "embedding_info.json"), "w") as f:
        json.dump(embedding_info, f)


def save_and_maybe_upload_to_hub(
    repo_id: str,
    output_dir: str,
    unet: UNet2DConditionModel,
    vae: AutoencoderKL,
    noise_scheduler: DDIMScheduler,
    action_embedding: torch.nn.Embedding,
    should_upload_to_hub: bool = True,
    images: list = None,
    dataset_name: str = None,
) -> None:
    save_model(output_dir, unet, vae, noise_scheduler, action_embedding)

    if should_upload_to_hub:
        try:
            upload_folder(
                repo_id=repo_id,
            folder_path=output_dir,
            commit_message="End of training",
            ignore_patterns=["step_*", "epoch_*"],
            )
            save_model_card(
                repo_id=repo_id,
                images=images,
                base_model=PRETRAINED_MODEL_NAME_OR_PATH,
                dataset_name=dataset_name,
                repo_folder=output_dir,
            )
        except Exception as e:
            print(f"Error uploading to hub: {e}")


def save_model_card(
    repo_id: str,
    images: list = None,
    base_model: str = None,
    dataset_name: str = None,
    repo_folder: str = None,
):
    img_str = ""
    if images is not None:
        for i, image in enumerate(images):
            image.save(os.path.join(repo_folder, f"image_{i}.png"))
            img_str += f"![img_{i}](./image_{i}.png)\n"

    model_description = textwrap.dedent(f"""
        # GameNgen fine-tuning - {repo_id}
        Full finetune of {base_model}. The weights were fine-tuned on the {dataset_name} dataset. You can find some example images in the following. \n
        {img_str}
        """)

    model_card = load_or_create_model_card(
        repo_id_or_path=repo_id,
        from_training=True,
        license="creativeml-openrail-m",
        base_model=base_model,
        model_description=model_description,
        inference=True,
    )

    tags = [
        "stable-diffusion",
        "stable-diffusion-diffusers",
        "text-to-image",
        "diffusers",
        "diffusers-training",
    ]
    model_card = populate_model_card(model_card, tags=tags)

    model_card.save(os.path.join(repo_folder, "README.md"))
