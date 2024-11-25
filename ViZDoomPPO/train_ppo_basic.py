import cv2
import multiprocessing
from stable_baselines3 import ppo
from stable_baselines3.common import callbacks
from stable_baselines3.common import policies
from stable_baselines3.common.vec_env import VecTransposeImage, SubprocVecEnv
import torch
from common import envs

if __name__ == '__main__':
    # Set up CUDA device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Environment parameters
    env_args = {
        'scenario': 'basic',
        'frame_skip': 4,  # Increased from 1 to 4 like in parallel example
        'frame_processor': lambda frame: cv2.resize(
            frame, None, fx=.5, fy=.5, interpolation=cv2.INTER_AREA)
    }

    # Use multiple environments for parallel training
    n_envs = multiprocessing.cpu_count() - 1  # Use all CPUs except one
    training_env = SubprocVecEnv([lambda: envs.create_env(**env_args) for _ in range(n_envs)])
    training_env = VecTransposeImage(training_env)  # Wrap for proper image handling
    
    # Single eval environment is fine
    eval_env = envs.create_vec_env(**env_args)

    # Create an agent with parallel training parameters
    agent = ppo.PPO(
        policy=policies.ActorCriticCnnPolicy,
        env=training_env,
        n_steps=4096,  # Increased buffer size for parallel collection
        batch_size=32,  # Added explicit batch size
        learning_rate=1e-4,
        device=device,
        tensorboard_log=None
    )

    # Add an evaluation callback that will save the best model when new records are reached.
    evaluation_callback = callbacks.EvalCallback(eval_env,
                                                 n_eval_episodes=10,
                                                 eval_freq=5000,
                                                 best_model_save_path='logs/models/basic')

    # Play!
    agent.learn(total_timesteps=1500, tb_log_name='ppo_basic', callback=evaluation_callback)

    training_env.close()
    eval_env.close()