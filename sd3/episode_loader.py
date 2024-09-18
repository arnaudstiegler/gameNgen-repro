from dataclasses import dataclass
from typing import List
from datasets import load_dataset
from PIL import Image as pil_image
import pickle


# TODO: probably not needed unless we need some utils functions on top of it
@dataclass
class Episode:
    sample_id: str
    health: List[int]
    actions: List[int]
    images: List[pil_image.Image]


if __name__ == "__main__":
    ds = load_dataset("P-H-B-D-a16z/ViZDoom-Deathmatch-PPO")
    episode = Episode(**ds["test"][0])
