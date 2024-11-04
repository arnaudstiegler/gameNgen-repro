import torch
from PIL import Image
import io
import base64
from torchvision import transforms
from config_sd import HEIGHT, WIDTH, TRAINING_DATASET_DICT, BUFFER_SIZE, ZERO_OUT_ACTION_CONDITIONING_PROB
from datasets import load_dataset
from data_augmentation import no_img_conditioning_augmentation


IMG_TRANSFORMS = transforms.Compose(
        [
            transforms.Resize((HEIGHT, WIDTH), interpolation=transforms.InterpolationMode.BILINEAR),
            transforms.ToTensor(),
            transforms.Normalize([0.5], [0.5]),
        ]
    )

def collate_fn(examples):
    processed_images = []
    for example in examples:

        # This means you have BUFFER_SIZE conditioning frames + 1 target frame
        processed_images.append(
            torch.stack(example["pixel_values"][:BUFFER_SIZE + 1]))

    # Stack all examples
    # images has shape: (batch_size, frame_buffer, 3, height, width)
    images = torch.stack(processed_images)
    images = images.to(memory_format=torch.contiguous_format).float()

    # UGLY HACK
    images = no_img_conditioning_augmentation(images, prob=ZERO_OUT_ACTION_CONDITIONING_PROB)
    return {
        "pixel_values": images,
        "input_ids": torch.stack([example["input_ids"][:BUFFER_SIZE+1].clone().detach() for example in examples]),
    }


def preprocess_train(examples):
    images = []
    for image_list in examples["images"]:
        current_images = []
        image_list = [
            Image.open(io.BytesIO(base64.b64decode(img))).convert("RGB")
            for img in image_list
        ]
        for image in image_list:
            current_images.append(IMG_TRANSFORMS(image))
        images.append(current_images)

    actions = torch.tensor(examples["actions"]) if isinstance(examples["actions"], list) else examples["actions"]
    return {"pixel_values": images, "input_ids": actions}


def get_dataset(dataset_name: str):
    dataset = load_dataset(dataset_name)
    return dataset["train"].with_transform(preprocess_train)  # type: ignore


def get_dataloader(dataset_name: str, batch_size: int = 1, shuffle: bool = False):
    dataset = get_dataset(dataset_name)
    return torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, collate_fn=collate_fn)  # type: ignore

def get_single_batch(dataset_name: str) -> dict[str, torch.Tensor]:
    dataloader = get_dataloader(dataset_name, batch_size=1, shuffle=False)
    return next(iter(dataloader))

if __name__ == "__main__":
    dataset = get_dataset(TRAINING_DATASET_DICT['small'])
    print(dataset[0])
    dataloader = get_dataloader(TRAINING_DATASET_DICT['small'])
    print(next(iter(dataloader)))