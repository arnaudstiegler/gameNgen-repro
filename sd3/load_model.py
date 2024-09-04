from diffusers import AutoencoderKL, DDIMScheduler, UNet2DConditionModel
from huggingface_hub import hf_hub_download
from config_sd import REPO_NAME

unet = UNet2DConditionModel.from_pretrained(REPO_NAME, subfolder='unet')
vae = AutoencoderKL.from_pretrained(REPO_NAME, subfolder='vae')
noise_scheduler = DDIMScheduler.from_pretrained(REPO_NAME, subfolder='noise_scheduler')

file_path = hf_hub_download(repo_id=REPO_NAME, filename="action_embedding_model.safetensors")