# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
#   /$$    /$$ /$$           /$$$$$$$                                          /$$$$$$$  /$$$$$$$   /$$$$$$     #
#  | $$   | $$|__/          | $$__  $$                                        | $$__  $$| $$__  $$ /$$__  $$    #
#  | $$   | $$ /$$ /$$$$$$$$| $$  \ $$  /$$$$$$   /$$$$$$  /$$$$$$/$$$$       | $$  \ $$| $$  \ $$| $$  \ $$    #
#  |  $$ / $$/| $$|____ /$$/| $$  | $$ /$$__  $$ /$$__  $$| $$_  $$_  $$      | $$$$$$$/| $$$$$$$/| $$  | $$    #
#   \  $$ $$/ | $$   /$$$$/ | $$  | $$| $$  \ $$| $$  \ $$| $$ \ $$ \ $$      | $$____/ | $$____/ | $$  | $$    #
#    \  $$$/  | $$  /$$__/  | $$  | $$| $$  | $$| $$  | $$| $$ | $$ | $$      | $$      | $$      | $$  | $$    #
#     \  $/   | $$ /$$$$$$$$| $$$$$$$/|  $$$$$$/|  $$$$$$/| $$ | $$ | $$      | $$      | $$      |  $$$$$$/    #
#      \_/    |__/|________/|_______/  \______/  \______/ |__/ |__/ |__/      |__/      |__/       \______/     #
#                                                                                                               #
# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #                                                                                                
                                                                                                         
# FORK OF LEANDRO KIELIGER'S DOOM PPO TUTORIAL: https://lkieliger.medium.com/deep-reinforcement-learning-in-practice-by-playing-doom-part-1-getting-started-618c99075c77                                                                                                       

# SCRIPT TO RUN PPO AGENT AND GENERATE DATASET FOR DOOM ENVIRONMENT. 

import imageio
from common import envs
import torch
from vizdoom.vizdoom import GameVariable


from tqdm import tqdm
import argparse
from huggingface_hub import HfApi
import pandas as pd

import io

import os

import pyarrow as pa
import pyarrow.parquet as pq
import numpy as np
from datasets import Dataset, DatasetDict
import glob
import pyarrow.parquet as pq
from train_ppo_parallel import DoomWithBotsCurriculum, game_instance
from stable_baselines3.common.vec_env import (
    VecTransposeImage,
    DummyVecEnv
)
from PIL import Image
from loguru import logger

# To replicate frame_skip in the environment
ACTION_REPEAT = 4
MODEL_PATH = "logs/models/deathmatch_simple/best_model.zip"


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
logger.info(f"Using device: {device}")


def dummy_vec_env_with_bots_curriculum(n_envs=1, **kwargs) -> VecTransposeImage:
    """Wraps a Doom game instance in a vectorized environment with shaped rewards and curriculum."""
    scenario = kwargs.pop("scenario")  # Remove 'scenario' from kwargs
    return VecTransposeImage(
        DummyVecEnv(
            [lambda: DoomWithBotsCurriculum(game_instance(scenario), **kwargs)] * n_envs
        )
    )


def make_gif(agent, file_path, eval_env_args, num_episodes=1):
    """Generate a GIF by running the agent in the environment.

    Args:
        agent: The trained PPO agent.
        file_path (str): Path to save the generated GIF.
        eval_env_args (dict): Arguments for the evaluation environment.
        num_episodes (int): Number of episodes to run.
    
    Returns:
        list: Collected health values for analysis.
    """
    # Set frame_skip to 1 to capture all frames
    eval_env_args['frame_skip'] = 1
    env = dummy_vec_env_with_bots_curriculum(1, **eval_env_args)

    images = []
    actions = []
    health_values = []
    current_action = None
    frame_counter = 0

    for episode in range(num_episodes):
        logger.info(f"Episode {episode + 1} of {num_episodes}")
        obs = env.reset()

        done = False
        while not done:
            if frame_counter % ACTION_REPEAT == 0:
                current_action, _ = agent.predict(obs)
            
            obs, _, done, _ = env.step(current_action)

            # Get the raw screen buffer from the Doom game instance
            screen = env.venv.envs[0].game.get_state().screen_buffer

            # Get the current health value
            health = env.venv.envs[0].game.get_game_variable(GameVariable.HEALTH)
            health_values.append(health)  # Store the health value

            actions.append(current_action)
            images.append(screen)

            frame_counter += 1

    print("Health values:", health_values)
    print("Number of health values:", len(health_values))
    print("Number of actions:", len(actions))
    print("Number of images:", len(images))

    # Save only the first 1000 frames to avoid large file size
    imageio.mimsave(file_path, images[:1000], fps=20)
    env.close()
    logger.info(f"GIF saved to {file_path}")
    
    return health_values


def make_pkls_dataset(agent, output_dir, eval_env_args, num_episodes=1):
    """Generate a dataset by running the agent in the environment and saving the data as Parquet files.

    Args:
        agent: The trained PPO agent.
        output_dir (str): Directory to save the Parquet files.
        eval_env_args (dict): Arguments for the evaluation environment.
        num_episodes (int): Number of episodes to run.
    """
    # Set frame_skip to 1 to capture all frames
    eval_env_args['frame_skip'] = 1
    env = dummy_vec_env_with_bots_curriculum(1, **eval_env_args)
    os.makedirs(output_dir, exist_ok=True)

    current_action = None
    frame_counter = 0

    for episode in tqdm(range(num_episodes), desc="Episodes"):
        obs = env.reset()
        done = False

        step_id = 0
        frames = []
        actions = []
        health_values = []
        step_ids = []

        while not done:
            if frame_counter % ACTION_REPEAT == 0:
                current_action, _ = agent.predict(obs)

            obs, _, done, _ = env.step(current_action)

            screen = env.venv.envs[0].game.get_state().screen_buffer
            health = env.venv.envs[0].game.get_game_variable(GameVariable.HEALTH)

            frames.append(screen)
            actions.append(int(current_action.item()))
            health_values.append(int(health))
            step_ids.append(step_id)

            step_id += 1
            frame_counter += 1

        episode_data = {
            'frames': [compress_image(frame) for frame in frames],
            'actions': actions,
            'health': health_values,
            'step_id': step_ids,
            'episode_id': [episode] * len(step_ids)
        }
        save_episodes_to_parquet(episode_data, output_dir)

    env.close()


def upload_to_hf(local_path: str, repo_id: str):
    api = HfApi()
    try:
        # Ensure the dataset repository exists
        api.create_repo(repo_id=repo_id, repo_type="dataset", exist_ok=True)
        
        # Upload the zip file
        api.upload_file(
            path_or_fileobj=local_path,
            path_in_repo=os.path.basename(local_path),
            repo_id=repo_id,
            repo_type="dataset",
        )
        print(f"Uploaded {local_path} to dataset {repo_id}")
    except Exception as e:
        print(f"Error uploading file: {e}")


def compress_image(image_array: np.ndarray, format='JPEG', quality=85):
    """Compress image using PIL with JPEG compression"""
    img = Image.fromarray(image_array)
    buffer = io.BytesIO()
    img.save(buffer, format=format, quality=quality)
    return buffer.getvalue()

def save_episodes_to_parquet(episode_data: dict, output_parquet: str):
    """Save batch with compressed binary images instead of base64"""
    processed_data = {
        'episode_id': episode_data['episode_id'],
        'frames': [frame for frame in episode_data['frames']],
        'actions': [int(a) for a in episode_data['actions']],
        'health': [int(h) for h in episode_data['health']],
        'step_ids': [int(s) for s in episode_data['step_id']]
    }

    df = pd.DataFrame.from_dict(processed_data)
    table = pa.Table.from_pandas(df)
    
    filename = f'{output_parquet}/episode_{episode_data["episode_id"][0]}.parquet'
    pq.write_table(table, filename, compression='zstd')


def create_hf_dataset_from_parquets(parquet_dir: str, repo_id: str) -> None:

    """Create a HF dataset from multiple parquet files"""
    # Get all parquet files
    parquet_files = glob.glob(f"{parquet_dir}/*.parquet")
    
    # Load and combine all parquet files
    dataset = Dataset.from_parquet(parquet_files)
    
    dataset_dict = DatasetDict({
        'train': dataset,
    })
    
    dataset_dict.push_to_hub(
            repo_id,
            private=False
        )
    logger.info(f"Dataset pushed to hub: {repo_id}")
    
    
def parse_arguments():
    parser = argparse.ArgumentParser(description="Generate GIF or Parquet file from pretrained PPO agent")
    parser.add_argument("--output", choices=["gif", "parquet"], required=True, help="Output format")
    parser.add_argument("--episodes", type=int, default=1, help="Number of episodes to run")
    parser.add_argument("--upload", action="store_true", help="Upload the output to Hugging Face Hub")
    parser.add_argument("--hf_token", help="Hugging Face API token (optional if HF_TOKEN env variable is set)")
    parser.add_argument("--hf_repo", help="Hugging Face repository name")
    return parser.parse_args()


def main():
    args = parse_arguments()
    scenario = "deathmatch_simple"

    env_args = {
        "scenario": scenario,
        "frame_skip": 1,
        "frame_processor": envs.default_frame_processor,
        "n_bots": 8,
        "shaping": True,
        "initial_level": 5,
        "max_level": 5,
        "rolling_mean_length": 10,
    }

    eval_env_args = dict(env_args)
    new_env = dummy_vec_env_with_bots_curriculum(1, **env_args)
    agent2 = envs.load_model(
        MODEL_PATH,
        new_env,
    )

    if args.output == "gif":
        output_file = "./output.gif"
        make_gif(agent2, output_file, eval_env_args, num_episodes=args.episodes)
    else:
        output_dir = "./dataset"
        make_pkls_dataset(agent2, output_dir, num_episodes=args.episodes, eval_env_args=eval_env_args)

    if args.upload:
        if not args.hf_repo:
            print("Error: --hf_repo is required for uploading to Hugging Face Hub")
        else:
            if args.output == "gif":
                upload_to_hf(output_file, args.hf_repo)
            else:
                # upload_to_hf(parquet_path, args.hf_repo, hf_token)
                create_hf_dataset_from_parquets(output_dir, repo_id=args.hf_repo)


if __name__ == "__main__":
    main()
