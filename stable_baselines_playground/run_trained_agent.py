from loguru import logger
from stable_baselines3 import PPO
import gymnasium
from vizdoom import gymnasium_wrapper

logger.info('Start predict')

env = gymnasium.make("VizdoomHealthGatheringSupreme-v0", render_mode="rgb_array")

model = PPO.load("ppo_trained_agent", env=env)
vec_env = model.get_env()
obs = vec_env.reset()
for i in range(1000):
    action, _state = model.predict(obs, deterministic=True)
    obs, reward, done, info = vec_env.step(action)
    vec_env.render("human")