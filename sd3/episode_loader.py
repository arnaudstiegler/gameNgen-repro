from dataclasses import dataclass
from typing import List
from datasets import load_dataset
from PIL import Image as pil_image
import pickle


@dataclass
class Episode:
    sample_id: str
    health: List[int]
    actions: List[int]
    images: List[pil_image.Image]


def load_episode(episode_path: str) -> Episode:
    with open(episode_path, "rb") as f:
        pickled_eps = pickle.load(f)
        return Episode(**pickled_eps)


if __name__ == "__main__":
    ds = load_dataset("P-H-B-D-a16z/ViZDoom-Deathmatch-PPO")
    episode = Episode(**ds["test"][0])
