import os
import torch
from diffusers import AutoencoderKL, UNet2DConditionModel, DDIMScheduler
from transformers import CLIPTokenizer, CLIPTextModel
from huggingface_hub import hf_hub_download
from config_sd import BUFFER_SIZE
from utils import NUM_BUCKETS
from huggingface_hub import upload_folder
from diffusers.utils.hub_utils import load_or_create_model_card, populate_model_card
from safetensors.torch import save_file


PRETRAINED_MODEL_NAME_OR_PATH = "CompVis/stable-diffusion-v1-4"


def get_ft_vae_decoder():
    '''
    Based on the original GameNGen code, the vae decoder is finetuned on images from the
    training set to improve the quality of the images.
    '''
    file_path = hf_hub_download(repo_id="P-H-B-D-a16z/GameNGenSDVaeDecoder", filename="trained_vae_decoder.pth")
    decoder_state_dict = torch.load(file_path)
    return decoder_state_dict


def get_model(action_dim: int, skip_image_conditioning: bool = False):
    # Max number of actions in the action space

    # This will be used to encode the actions
    action_embedding = torch.nn.Embedding(num_embeddings=action_dim + 1,
                                          embedding_dim=768)

    # DDIM scheduler allows for v-prediction and less sampling steps
    noise_scheduler = DDIMScheduler.from_pretrained(
        PRETRAINED_MODEL_NAME_OR_PATH, subfolder="scheduler")
    # This is what the paper uses
    noise_scheduler.register_to_config(prediction_type="v_prediction")

    vae = AutoencoderKL.from_pretrained(PRETRAINED_MODEL_NAME_OR_PATH,
                                        subfolder="vae")
    decoder_state_dict = get_ft_vae_decoder()
    vae.decoder.load_state_dict(decoder_state_dict)


    unet = UNet2DConditionModel.from_pretrained(PRETRAINED_MODEL_NAME_OR_PATH,
                                                subfolder="unet")
    # There are 10 noise buckets total
    unet.register_to_config(num_class_embeds=NUM_BUCKETS)
    # We do not use .add_module() because the class_embedding is already initialized as None
    unet.class_embedding = torch.nn.Embedding(NUM_BUCKETS, unet.time_embedding.linear_2.out_features)

    tokenizer = CLIPTokenizer.from_pretrained(PRETRAINED_MODEL_NAME_OR_PATH,
                                              subfolder="tokenizer")
    text_encoder = CLIPTextModel.from_pretrained(PRETRAINED_MODEL_NAME_OR_PATH,
                                                 subfolder="text_encoder")

    if not skip_image_conditioning:
        """
        This is to accomodate concatenating previous frames in the channels dimension
        """
        new_in_channels = 4 * (BUFFER_SIZE + 1)
        old_conv_in = unet.conv_in
        new_conv_in = torch.nn.Conv2d(new_in_channels,
                                      320,
                                      kernel_size=(3, 3),
                                      stride=(1, 1),
                                      padding=(1, 1))

        # Initialize the new conv layer with the weights from the old one
        with torch.no_grad():
            new_conv_in.weight[:, :4, :, :] = old_conv_in.weight
            # Initialize new channels to random values
            new_conv_in.weight[:, 4:, :, :] = torch.randn_like(
                new_conv_in.weight[:, 4:, :, :])

            new_conv_in.bias = old_conv_in.bias

        # Replace the conv_in layer
        unet.conv_in = new_conv_in
        # Have to account for BUFFER SIZE conditioning frames + 1 for the noise
        unet.config["in_channels"] = new_in_channels

    unet.requires_grad_(True)
    vae.requires_grad_(False)
    text_encoder.requires_grad_(False)
    return unet, vae, action_embedding, noise_scheduler, tokenizer, text_encoder

def load_model(model_folder: str, action_dim: int):
    noise_scheduler = DDIMScheduler.from_pretrained(
        model_folder, subfolder="scheduler"
    )

    vae = AutoencoderKL.from_pretrained(model_folder, subfolder="vae", map_location=torch.device('cpu'))
    decoder_state_dict = get_ft_vae_decoder()
    vae.decoder.load_state_dict(decoder_state_dict)
    import ipdb; ipdb.set_trace()
    unet = UNet2DConditionModel.from_pretrained(
        model_folder, subfolder="unet"
    )
    action_embedding = torch.nn.Embedding(
        num_embeddings=action_dim + 1, embedding_dim=768
    )
    action_embedding.load_state_dict(torch.load(os.path.join(model_folder, "action_embedding.pth")))

    assert noise_scheduler.config.prediction_type == "v_prediction", "Noise scheduler prediction type should be 'v_prediction'"
    assert unet.config.num_class_embeds == NUM_BUCKETS, f"UNet num_class_embeds should be {NUM_BUCKETS}"

    # Unaltered
    tokenizer = CLIPTokenizer.from_pretrained(PRETRAINED_MODEL_NAME_OR_PATH, subfolder="tokenizer")
    text_encoder = CLIPTextModel.from_pretrained(
        PRETRAINED_MODEL_NAME_OR_PATH, subfolder="text_encoder"
    )
    return unet, vae, action_embedding, noise_scheduler, tokenizer, text_encoder


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

    model_description = f"""
# GameNgen fine-tuning - {repo_id}
Full finetune of {base_model}. The weights were fine-tuned on the {dataset_name} dataset. You can find some example images in the following. \n
{img_str}
"""

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


def save_model(output_dir: str, unet, vae, noise_scheduler, action_embedding):
    unet.save_pretrained(os.path.join(output_dir, "unet"))
    vae.save_pretrained(os.path.join(output_dir, "vae"))
    noise_scheduler.save_pretrained(
        os.path.join(output_dir, "scheduler"))
    torch.save(action_embedding.state_dict(),
                os.path.join(output_dir, "action_embedding.pth"))

    # Save embedding dimensions
    embedding_info = {
        "num_embeddings": action_embedding.num_embeddings,
        "embedding_dim": action_embedding.embedding_dim
    }

    torch.save(embedding_info,
                os.path.join(output_dir, "embedding_info.pth"))
    unet = unet.to(torch.float32)

def save_to_hub(repo_id: str, output_dir: str, dataset_name: str, validation_images: list[str] | None, unet, vae, noise_scheduler, action_embedding):
    unet.save_pretrained(os.path.join(output_dir, "unet"))
    vae.save_pretrained(os.path.join(output_dir, "vae"))
    save_file(
        action_embedding.state_dict(),
        os.path.join(output_dir,
                        "action_embedding_model.safetensors"),
    )
    noise_scheduler.save_pretrained(
        os.path.join(output_dir, "noise_scheduler"))

    save_model_card(
        repo_id,
        images=validation_images if validation_images else [],
        base_model=PRETRAINED_MODEL_NAME_OR_PATH,
        dataset_name=dataset_name,
        repo_folder=output_dir,
    )
    upload_folder(
        repo_id=repo_id,
        folder_path=output_dir,
        commit_message="End of training",
        ignore_patterns=["step_*", "epoch_*"],
    )