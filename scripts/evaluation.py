
import numpy as np
import gymnasium as gym
from pathlib import Path
import sys
ROOT_DIR = Path(__file__).resolve().parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.append(str(ROOT_DIR))
import snake_env 
from stable_baselines3 import PPO

# Load model
model_path = "models/PPO/best_model/best_model_optuna_stacked_50.zip"  # Adjust to your saved checkpoint
model = PPO.load(model_path)

# Create environment with rendering
env = gym.make("Snake-v0", render_mode="human", render_fps=50)

# Number of episodes to watch
NUM_EPISODES = 5

ep_apples = []

for ep in range(NUM_EPISODES):
    obs, _ = env.reset()
    done = False
    total_reward = 0

    while not done:
        action, _ = model.predict(obs, deterministic=False)
        obs, reward, terminated, truncated, info = env.step(action)
        done = terminated or truncated
        total_reward += reward
        env.render()  
    ep_apples.append(info["apples"])

mean_apples = np.mean(ep_apples)
sd_apples = np.std(ep_apples)

print("# Eval episodes:", NUM_EPISODES)
print("Mean:", mean_apples)
print("Std:", sd_apples)