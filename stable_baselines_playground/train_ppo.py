import gymnasium as gym

from stable_baselines3 import PPO
import gymnasium
from loguru import logger

from vizdoom import gymnasium_wrapper  # noqa

env = gymnasium.make("VizdoomCorridor-v0", render_mode="rgb_array")

logger.info("Start training")
model = PPO("MultiInputPolicy", env, verbose=1)
model.learn(total_timesteps=10, progress_bar=True)

model.save("trained_agents/ppo_trained_agent")

# VecEnv resets automatically
# if done:
#   obs = vec_env.reset()
