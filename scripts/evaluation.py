import gymnasium as gym
from pathlib import Path
import sys
ROOT_DIR = Path(__file__).resolve().parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.append(str(ROOT_DIR))
import snake_env 
from stable_baselines3 import PPO

# Load model
model_path = "models/PPO/interrupted_model.zip"  # Adjust to your saved checkpoint
model = PPO.load(model_path)

# Create environment with rendering
env = gym.make("Snake-v0", render_mode="human", render_fps=50)

# Number of episodes to watch
NUM_EPISODES = 5

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
