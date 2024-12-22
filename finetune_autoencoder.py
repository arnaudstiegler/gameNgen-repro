from datetime import datetime
import argparse

import torch
import torch.nn.functional as F
from datasets import load_dataset
from diffusers import AutoencoderKL
from torch import optim
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers import get_cosine_schedule_with_warmup
from config_sd import PRETRAINED_MODEL_NAME_OR_PATH

import wandb
from dataset import preprocess_train

# Fine-tuning parameters
NUM_EPOCHS = 2
NUM_WARMUP_STEPS = 500
BATCH_SIZE = 16
LEARNING_RATE = 1e-4
WEIGHT_DECAY = 1e-5
GRADIENT_CLIP_NORM = 1.0
EVAL_STEP = 1000


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Fine-tune VAE model")
    parser.add_argument(
        "--hf_model_folder",
        type=str,
        required=True,
        help="HuggingFace model folder to save the model to",
    )
    return parser.parse_args()


def make_decoder_trainable(model: AutoencoderKL):
    for param in model.encoder.parameters():
        param.requires_grad_(False)
    for param in model.decoder.parameters():
        param.requires_grad_(True)


def eval_model(model: AutoencoderKL, test_loader: DataLoader) -> float:
    model.eval()
    with torch.no_grad():
        test_loss = 0
        progress_bar = tqdm(test_loader, desc=f"Evaluating")

        for batch in progress_bar:
            data = batch["pixel_values"].to(model.device)
            reconstruction = model(data).sample
            loss = F.mse_loss(reconstruction, data, reduction="mean")
            test_loss += loss.item()

            recon = model.decode(model.encode(data).latent_dist.sample()).sample
            wandb.log(
                {
                    "original": [wandb.Image(img) for img in data],
                    "reconstructed": [wandb.Image(img) for img in recon],
                }
            )
        return test_loss / len(test_loader)


def main():
    args = parse_args()
    wandb.init(
        project="gamengen-vae-training",
        config={
            # Model parameters
            "model": PRETRAINED_MODEL_NAME_OR_PATH,
            # Training parameters
            "num_epochs": NUM_EPOCHS,
            "eval_step": EVAL_STEP,
            "batch_size": BATCH_SIZE,
            "learning_rate": LEARNING_RATE,
            "weight_decay": WEIGHT_DECAY,
            "warmup_epochs": NUM_WARMUP_STEPS,
            "gradient_clip_norm": GRADIENT_CLIP_NORM,
            "hf_model_folder": args.hf_model_folder,
        },
        name=f"vae-finetuning-{datetime.now().strftime('%Y-%m-%d-%H-%M-%S')}",
    )

    # Dataset Setup
    dataset = load_dataset("arnaudstiegler/vizdoom-500-episodes-skipframe-4-lvl5")
    split_dataset = dataset["train"].train_test_split(test_size=500, seed=42)
    train_dataset = split_dataset["train"].with_transform(preprocess_train)
    test_dataset = split_dataset["test"].with_transform(preprocess_train)
    train_loader = DataLoader(
        train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=8
    )
    test_loader = DataLoader(
        test_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=8
    )
    # Model Setup
    vae = AutoencoderKL.from_pretrained(PRETRAINED_MODEL_NAME_OR_PATH, subfolder="vae")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = vae.to(device)
    make_decoder_trainable(model)
    # Optimizer Setup
    optimizer = optim.AdamW(
        model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY
    )
    scheduler = get_cosine_schedule_with_warmup(
        optimizer,
        num_warmup_steps=NUM_WARMUP_STEPS,
        num_training_steps=NUM_EPOCHS * len(train_loader),
    )

    step = 0
    for epoch in range(NUM_EPOCHS):
        train_loss = 0
        progress_bar = tqdm(train_loader, desc=f"Epoch {epoch + 1}/{NUM_EPOCHS}")

        for batch in progress_bar:
            model.train()
            data = batch["pixel_values"].to(device)
            optimizer.zero_grad()

            reconstruction = model(data).sample
            loss = F.mse_loss(reconstruction, data, reduction="mean")

            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), GRADIENT_CLIP_NORM)
            optimizer.step()
            scheduler.step()

            train_loss += loss.item()
            current_lr = scheduler.get_last_lr()[0]

            progress_bar.set_postfix({"loss": loss.item(), "lr": current_lr})

            wandb.log(
                {
                    "train_loss": loss.item(),
                    "learning_rate": current_lr,
                }
            )

            step += 1
            if step % EVAL_STEP == 0:
                test_loss = eval_model(model, test_loader)
                # save model to hub
                model.save_pretrained(
                    "test",
                    repo_id=args.hf_model_folder,
                    push_to_hub=True,
                )
                wandb.log({"test_loss": test_loss})


if __name__ == "__main__":
    main()
