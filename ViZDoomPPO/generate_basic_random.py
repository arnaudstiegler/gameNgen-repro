import cv2
import numpy as np
from common import envs
import imageio
import os
import io
from PIL import Image
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
from tqdm import tqdm
from huggingface_hub import HfApi
from datasets import Dataset, DatasetDict
import glob
import argparse
from loguru import logger

def compress_image(image_array: np.ndarray, format='JPEG', quality=85):
    """Compress image using PIL with JPEG compression"""
    img = Image.fromarray(image_array)
    buffer = io.BytesIO()
    img.save(buffer, format=format, quality=quality)
    return buffer.getvalue()

def save_episodes_to_parquet(episode_data: dict, output_parquet: str):
    """Save batch with compressed binary images"""
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

def run_random_episodes(output_dir='./dataset', n_episodes=10):
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)

    # Environment parameters
    env_args = {
        'scenario': 'basic',
        'frame_skip': 1,
        'frame_processor': lambda frame: cv2.resize(
            frame, None, fx=.5, fy=.5, interpolation=cv2.INTER_AREA)
    }
    
    env = envs.create_env(**env_args)
    
    for episode in tqdm(range(n_episodes), desc="Episodes"):
        frames = []
        actions = []
        health_values = []
        step_ids = []
        step_id = 0
        
        obs = env.reset()
        done = False
        
        while not done:
            # Get random action
            action = env.action_space.sample()
            
            # Take step
            obs, reward, terminated, truncated = env.step(action)
            done = terminated or truncated
            
            # Get the raw screen buffer
            if isinstance(obs, np.ndarray):
                frame = obs
            else:
                frame = obs.transpose(1, 2, 0)  # CHW to HWC format
            
            # Compress and store data
            frames.append(compress_image(frame))
            actions.append(int(action))
            health_values.append(100)  # Basic scenario always has 100 health
            step_ids.append(step_id)
            
            step_id += 1

        # Save episode data
        episode_data = {
            'frames': frames,
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
    parser = argparse.ArgumentParser(description="Generate random action dataset for basic Doom scenario")
    parser.add_argument("--episodes", type=int, default=10, help="Number of episodes to run")
    parser.add_argument("--output_dir", type=str, default="./dataset", help="Output directory for parquet files")
    parser.add_argument("--upload", action="store_true", help="Upload the output to Hugging Face Hub")
    parser.add_argument("--hf_repo", help="Hugging Face repository name")
    return parser.parse_args()

if __name__ == '__main__':
    args = parse_arguments()
    
    # Run episodes and generate parquet files
    run_random_episodes(output_dir=args.output_dir, n_episodes=args.episodes)
    
    # Upload to HuggingFace if requested
    if args.upload:
        if not args.hf_repo:
            print("Error: --hf_repo is required for uploading to Hugging Face Hub")
        else:
            create_hf_dataset_from_parquets(args.output_dir, repo_id=args.hf_repo)