from typing import Dict, Any
from loguru import logger
from stable_baselines3 import PPO
import gymnasium
from vizdoom import gymnasium_wrapper
import numpy as np
from collections import deque
from PIL import Image as pil_image
import io
from datasets import Dataset, Features, Image, Value, Sequence
from config_sd import BUFFER_SIZE

"""
For DefendTheLine scenario:
0. No action (no-op)
1. Move right
2. Move left
3. Shoot
"""

EPISODE_LENGTH = 100


def generate_hf_parquet_dataset(entries: Dict[str, Any]):
    features = Features(
        {
            "images": Sequence(Image()),
            "sample_id": Value("int32"),
            "actions": Sequence(Value("int32")),
            "game_variables": Sequence(Sequence(Value("float32"))),
        }
    )
    dataset = Dataset.from_list(entries, features=features)

    # dataset.to_parquet("gameNgen_test_dataset.parquet")

    return dataset


logger.info("Start generating episodes")

vizdoom_scenarios = [
    "VizdoomHealthGatheringSupreme-v0",
    "VizdoomCorridor-v0",
    "VizdoomDefendCenter-v0",
    "VizdoomHealthGathering-v0",
    "VizdoomMyWayHome-v0",
    "VizdoomPredictPosition-v0",
    "VizdoomTakeCover-v0",
]


# TODO: there is a frame_skip option here
env = gymnasium.make("VizdoomCorridor-v0", render_mode="rgb_array")
env.get_wrapper_attr("game").set_render_hud(True)

model = PPO.load("trained_agents/ppo_trained_agent", env=env)
vec_env = model.get_env()
obs = vec_env.reset()


frame_buffer = deque(maxlen=BUFFER_SIZE)
action_buffer = deque(maxlen=BUFFER_SIZE)

entries = []
for i in range(EPISODE_LENGTH + BUFFER_SIZE):
    # Either use the model to predict the action or sample a random action
    # action, _state = model.predict(obs, deterministic=True)

    action = [env.action_space.sample()]
    # The action is a list of actions, but the buffer is a queue of actions
    action_buffer.append(action[0])
    obs, reward, done, info = vec_env.step(action)
    frame_buffer.append(obs)

    if i < BUFFER_SIZE:
        # So that we don't have to worry about padding for now
        continue

    actions = list(action_buffer)
    frames = list(frame_buffer)
    game_variables = []
    images = []
    # Dump frames locally with a unique id
    for j, frame in enumerate(frames):
        frame_path = f"test_dataset/data/train/{i}_{j}.png"
        game_variables.append(frame["gamevariables"][0].tolist())
        # Careful with the collision between HF Image and PIL Image
        images.append(
            pil_image.fromarray(np.transpose(frame["screen"].squeeze(0), (1, 2, 0)))
        )

    entries.append(
        {
            "sample_id": i,
            "actions": actions,
            "game_variables": game_variables,
            "images": images,
        }
    )

logger.info("Creating the HF dataset")
dataset = generate_hf_parquet_dataset(entries)
dataset.push_to_hub("arnaudstiegler/gameNgen_test_dataset")
