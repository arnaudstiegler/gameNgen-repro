import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from datasets import load_dataset
from diffusers import AutoencoderKL
from torchvision import transforms
from PIL import Image
import io
import base64
import wandb
from config_sd import HEIGHT, WIDTH
from tqdm import tqdm   

# Constants
BATCH_SIZE = 8
LEARNING_RATE = 1e-4
NUM_EPOCHS = 10
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load the dataset
dataset = load_dataset("P-H-B-D-a16z/ViZDoom-Deathmatch-PPO")
train_dataset = dataset["train"]

# Define image transformations
transform = transforms.Compose([
    transforms.Resize((HEIGHT, WIDTH),
                            interpolation=transforms.InterpolationMode.BILINEAR),
    transforms.ToTensor(),
    transforms.Normalize([0.5], [0.5]),
])


# Custom dataset class
class CustomDataset(torch.utils.data.Dataset):
    def __init__(self, dataset, transform):
        self.dataset = dataset
        self.transform = transform

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        image_list = self.dataset[idx]["images"]
        image = Image.open(io.BytesIO(base64.b64decode(image_list[0]))).convert("RGB")
        return self.transform(image)

# Create dataset and dataloader
custom_dataset = CustomDataset(train_dataset, transform)
dataloader = DataLoader(custom_dataset, batch_size=BATCH_SIZE, shuffle=True)

# Load the pre-trained VAE
PRETRAINED_MODEL_NAME_OR_PATH = "CompVis/stable-diffusion-v1-4"
vae = AutoencoderKL.from_pretrained(PRETRAINED_MODEL_NAME_OR_PATH,
                                        subfolder="vae").to(DEVICE)
# Load the saved encoder state dictionary
decoder_state_dict = torch.load("trained_vae_decoder.pth")

# Load the state dictionary into the model's encoder
vae.decoder.load_state_dict(decoder_state_dict)


# Freeze the encoder
for param in vae.encoder.parameters():
    param.requires_grad = False

# Define loss function and optimizer
criterion = nn.MSELoss()
optimizer = optim.Adam(vae.decoder.parameters(), lr=LEARNING_RATE)


# Initialize wandb
wandb.init(project="vae-training", name="vae-mse-loss")
try:
    for epoch in range(NUM_EPOCHS):
        vae.train()
        total_loss = 0
        print("starting epoch", epoch)
        
        # Wrap the dataloader with tqdm for progress bar
        progress_bar = tqdm(dataloader, desc=f"Epoch {epoch+1}/{NUM_EPOCHS}")
        for step, batch in enumerate(progress_bar):
            batch = batch.to(DEVICE)
            
            # Forward pass
            encoded = vae.encode(batch).latent_dist.sample()
            decoded = vae.decode(encoded).sample
            
            # Compute loss
            loss = criterion(decoded, batch)

            # Backward pass and optimize
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            
            wandb.log({"step": step + 1, "step_loss": loss.item()})

            # Update tqdm description with current step loss
            progress_bar.set_postfix({"step_loss": loss.item()})

            # Log an example image every 100 steps
            if (step) % 100 == 0:
                vae.eval()
                with torch.no_grad():
                    sample = batch[:4]  # Take a small batch for logging
                    recon = vae.decode(vae.encode(sample).latent_dist.sample()).sample
                    wandb.log({
                        "original": [wandb.Image(img) for img in sample],
                        "reconstructed": [wandb.Image(img) for img in recon]
                    })
                vae.train()

        avg_loss = total_loss / len(dataloader)
        print(f"Epoch [{epoch+1}/{NUM_EPOCHS}], Loss: {avg_loss:.4f}")

        # Log epoch loss
        wandb.log({"epoch": epoch+1, "epoch_loss": avg_loss})

except KeyboardInterrupt:
    print("Training interrupted. Saving model...")

finally:
    # Save the trained decoder
    torch.save(vae.decoder.state_dict(), "trained_vae_decoder.pth")
    print("Model saved.")

    # Finish wandb run
    wandb.finish()

print("Training completed!")