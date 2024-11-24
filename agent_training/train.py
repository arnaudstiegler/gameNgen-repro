import multiprocessing
import gymnasium as gym
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import SubprocVecEnv, VecTransposeImage
from loguru import logger



def make_env(rank):
    """Create a wrapped, monitored VecEnv for ViZDoom.
    
    Args:
        rank (int): Process rank for seeding
    
    Returns:
        callable: A function that creates the environment
    """
    def _init():
        env = gym.make("VizdoomDeathmatchSimple-v0", render_mode="rgb_array")
        return env
    return _init

# Number of parallel environments
n_envs = multiprocessing.cpu_count() - 1

# Create vectorized environments
env = SubprocVecEnv([make_env(i) for i in range(n_envs)])
# Wrap to get images in correct format (channels first)
env = VecTransposeImage(env)

logger.info("Start training")
model = PPO("CnnPolicy", env,
            n_steps=4096 // n_envs,  # Adjust steps per environment
            batch_size=64,
            learning_rate=1e-4,
            ent_coef=0.1,
            gamma=0.99,
            verbose=1)

model.learn(total_timesteps=10_000_000, progress_bar=True)
model.save("trained_agent")

# Don't forget to close the environments
env.close()