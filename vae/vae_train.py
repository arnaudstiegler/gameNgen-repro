import argparse
import time
from datasets import Dataset
from diffusers import AutoencoderKL
import torch
from torch import nn, optim
from torch.utils.data import DataLoader
from torchvision import transforms


def main(dataset_dir, save_dir, num_epochs):
    # Load dataset
    dataset = Dataset.load_from_disk(dataset_dir)
    dataset = dataset[0]["images"]

    class CustomDataset(torch.utils.data.Dataset):
        def __init__(self, hf_dataset, transform=None):
            self.dataset = hf_dataset
            self.transform = transform

        def __len__(self):
            return len(self.dataset)

        def __getitem__(self, idx):
            image = self.dataset[idx]
            if self.transform:
                image = self.transform(image)
            return image

    # Define transformations and dataloader
    transform = transforms.Compose(
        [transforms.Resize((256, 256)), transforms.ToTensor()]
    )
    torch_dataset = CustomDataset(dataset, transform=transform)
    dataloader = DataLoader(torch_dataset, batch_size=8, shuffle=True)

    # Load VAE and freeze encoder
    vae = AutoencoderKL.from_pretrained(
        "CompVis/stable-diffusion-v1-4", subfolder="vae"
    )
    for param in vae.encoder.parameters():
        param.requires_grad = False

    # Only the decoder parameters will be trainable
    decoder_params = [
        param for param in vae.decoder.parameters() if param.requires_grad
    ]
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    vae.to(device)

    # Define loss function and optimizer
    criterion = nn.MSELoss()
    optimizer = optim.Adam(decoder_params, lr=1e-4)

    # Training loop
    print("Starting Training")
    start_time = time.time()
    for epoch in range(num_epochs):
        print(f"Starting epoch {epoch+1}")
        vae.train()
        running_loss = 0.0
        for images in dataloader:
            images = images.to(device)

            optimizer.zero_grad()
            # Forward pass through the encoder and decoder
            latents = vae.encode(images).latent_dist.sample()
            recon_images = vae.decode(latents)
            loss = criterion(recon_images[0], images)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
            # break   # Comment this line to train on the entire dataset

        avg_loss = running_loss / len(dataloader)
        elapsed_time = time.time() - start_time
        print(
            f"Epoch [{epoch+1}/{num_epochs}], Loss: {avg_loss:.4f}, Time: {elapsed_time:.4f}"
        )

    print("Finished Training")
    vae.save_pretrained(save_dir)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train a VAE on a dataset.")
    parser.add_argument(
        "--dataset_dir", type=str, required=True, help="Directory of the dataset"
    )
    parser.add_argument(
        "--save_dir",
        type=str,
        required=True,
        help="Directory to save the finetuned VAE",
    )
    parser.add_argument(
        "--num_epochs", type=int, required=True, help="Number of epochs to train"
    )

    args = parser.parse_args()
    main(args.dataset_dir, args.save_dir, args.num_epochs)
