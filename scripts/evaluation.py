import argparse
import gymnasium as gym
from pathlib import Path
import sys
ROOT_DIR = Path(__file__).resolve().parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.append(str(ROOT_DIR))
import snake_env 
from stable_baselines3 import PPO

def main():
    parser = argparse.ArgumentParser(description="Evalúa un checkpoint de Snake.")
    parser.add_argument("--episodes", type=int, default=1000)
    parser.add_argument("--render_mode", type=str, default=None)
    parser.add_argument("--render_fps", type=int, default=60)
    args = parser.parse_args()
    if args.episodes < 1:
        parser.error("--episodes debe ser mayor que cero")

    model_selection = input("Selecciona el modelo (ruta relativa a models/PPO): ")
    model = PPO.load(Path("models/PPO") / model_selection, device="cpu")
    total_apples = total_steps = step_limits = 0

    with gym.make("Snake-v0", render_mode = args.render_mode, render_fps = args.render_fps) as env:
        print(f"Evaluando {args.episodes} partidas de hasta {env.spec.max_episode_steps} pasos.", flush=True)
        for episode in range(1, args.episodes + 1):
            obs, _ = env.reset()
            terminated = truncated = False
            while not (terminated or truncated):
                action, _ = model.predict(obs, deterministic=True)
                obs, _, terminated, truncated, info = env.step(action)
                total_steps += 1
                env.render()


            total_apples += info["apples"]
            step_limits += truncated and not terminated
            if episode == 1 or episode % 100 == 0 or episode == args.episodes:
                print(
                    f"{episode}/{args.episodes} partidas · {total_apples / episode:.2f} manzanas de media · ",
                    flush=True,
                )


if __name__ == "__main__":
    main()
