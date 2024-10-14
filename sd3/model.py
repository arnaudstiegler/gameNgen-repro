import os
import torch
from diffusers import AutoencoderKL, UNet2DConditionModel, DDIMScheduler
from transformers import CLIPTokenizer, CLIPTextModel
from config_sd import BUFFER_SIZE
from utils import NUM_BUCKETS


PRETRAINED_MODEL_NAME_OR_PATH = "CompVis/stable-diffusion-v1-4"


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
    unet = UNet2DConditionModel.from_pretrained(PRETRAINED_MODEL_NAME_OR_PATH,
                                                subfolder="unet")
    # There are 10 noise buckets total
    unet.register_to_config(num_class_embeds=NUM_BUCKETS)
    # TODO: pretty unsure about the dimension here
    unet.class_embeddings = torch.nn.Embedding(NUM_BUCKETS, unet.time_embedding.linear_2.out_features)

    import ipdb; ipdb.set_trace()

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
    # TODO: unfreeze
    vae.requires_grad_(False)
    text_encoder.requires_grad_(False)
    return unet, vae, action_embedding, noise_scheduler, tokenizer, text_encoder

def load_model(model_folder: str, action_dim: int, skip_image_conditioning: bool = False):
    noise_scheduler = DDIMScheduler.from_pretrained(
        model_folder, subfolder="scheduler"
    )

    vae = AutoencoderKL.from_pretrained(model_folder, subfolder="vae")
    unet = UNet2DConditionModel.from_pretrained(
        model_folder, subfolder="unet"
    )
    tokenizer = CLIPTokenizer.from_pretrained(PRETRAINED_MODEL_NAME_OR_PATH, subfolder="tokenizer")
    text_encoder = CLIPTextModel.from_pretrained(
        PRETRAINED_MODEL_NAME_OR_PATH, subfolder="text_encoder"
    )
    action_embedding = torch.nn.Embedding(
        num_embeddings=action_dim + 1, embedding_dim=768
    )
    action_embedding.load_state_dict(torch.load(os.path.join(model_folder, "action_embedding.pth")))
    return unet, vae, action_embedding, noise_scheduler, tokenizer, text_encoder
