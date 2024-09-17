import torch
from diffusers import AutoencoderKL, UNet2DConditionModel, DDPMScheduler
from config_sd import BUFFER_SIZE


PRETRAINED_MODEL_NAME_OR_PATH = "CompVis/stable-diffusion-v1-4"


def get_model(action_dim: int):
    # Max number of actions in the action space
    
    # This will be used to encode the actions
    action_embedding = torch.nn.Embedding(
        num_embeddings=action_dim + 1, embedding_dim=768
    )

    # Load scheduler, tokenizer and models.
    noise_scheduler = DDPMScheduler.from_pretrained(
        PRETRAINED_MODEL_NAME_OR_PATH, subfolder="scheduler"
    )

    vae = AutoencoderKL.from_pretrained(
        PRETRAINED_MODEL_NAME_OR_PATH,
        subfolder="vae"
    )
    unet = UNet2DConditionModel.from_pretrained(
        PRETRAINED_MODEL_NAME_OR_PATH,
        subfolder="unet"
    )

    """
    This is to accomodate concatenating previous frames in the channels dimension
    """
    old_conv_in = unet.conv_in
    new_conv_in = torch.nn.Conv2d(
        40, 320, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)
    )

    # Initialize the new conv layer with the weights from the old one
    with torch.no_grad():
        new_conv_in.weight[:, :4, :, :] = old_conv_in.weight
        new_conv_in.weight[:, 4:, :, :] = 0  # Initialize new channels to 0
        new_conv_in.bias = old_conv_in.bias

    # Replace the conv_in layer
    unet.conv_in = new_conv_in
    unet.config["in_channels"] = 4 * BUFFER_SIZE

    unet.requires_grad_(True)
    # TODO: unfreeze
    vae.requires_grad_(False)
    return unet, vae, action_embedding, noise_scheduler