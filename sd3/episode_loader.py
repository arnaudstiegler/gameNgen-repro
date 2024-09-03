from pydantic import BaseModel
from typing import List, Dict, Any
import pickle

class Episode(BaseModel):
    actions: List[List[int]]
    frames: List[Dict[str, Any]]

def load_episode(episode_path: str) -> Episode:
    with open(episode_path, "rb") as f:
        pickled_eps = pickle.load(f)
        return Episode(**pickled_eps)


if __name__ == "__main__":
    episode = load_episode("data/episode_0.pkl")
    print(episode)
