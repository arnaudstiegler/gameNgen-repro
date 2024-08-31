from loguru import logger
from stable_baselines3 import PPO
import gymnasium
from vizdoom import gymnasium_wrapper
import numpy as np


logger.info('Start predict')

vizdoom_scenarios = [
    "VizdoomHealthGatheringSupreme-v0",
    "VizdoomCorridor-v0",
    "VizdoomDefendCenter-v0",
    "VizdoomHealthGathering-v0",
    "VizdoomMyWayHome-v0",
    "VizdoomPredictPosition-v0",
    "VizdoomTakeCover-v0",
]

env = gymnasium.make("VizdoomDefendLine-v0", render_mode="rgb_array")
env.get_wrapper_attr('game').set_render_hud(True)

model = PPO.load("trained_agents/ppo_trained_agent", env=env)
vec_env = model.get_env()
obs = vec_env.reset()
for i in range(1000):
    # Either use the model to predict the action or sample a random action
    # action, _state = model.predict(obs, deterministic=True)
    action = np.array(env.action_space.sample())
    import ipdb; ipdb.set_trace()
    obs, reward, done, info = vec_env.step(action)
    # vec_env.render("human")
