
import os
import argparse

import snake_env
from .utils import (
    cfg,
    make_env,
    CustomEvalCallback
)

import gymnasium as gym
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import SubprocVecEnv
from stable_baselines3.common.evaluation import evaluate_policy

def parse_args():
    parser = argparse.ArgumentParser(
        description=""
    )

    parser.add_argument(
        "--n_envs",
        type=int,
        default=8,
        help="Number of parallel environments (default=8)"
    )
    parser.add_argument(
        "--n_iterations",
        type=int,
        default=10,
        help="Number of training iterations (default=10). Each iteration corresponds to 100_000 timesteps."
    )
    parser.add_argument(
        "--n_eval_episodes",
        type=int,
        default=10,
        help="Number of episodes to run for evaluation (default=10)."
    )
    parser.add_argument(
        "--n_evals",
        type=int,
        default=10,
        help="Number of evals during training (default=10)."
    )

    return parser.parse_args()


if __name__ == "__main__":

    # SETUP
    args = parse_args()
    if not os.path.exists(cfg.models_dir):
        os.makedirs(cfg.models_dir)

    if not os.path.exists(cfg.log_dir):
        os.makedirs(cfg.log_dir)


    # ENVS
    if args.n_envs == 1:
        envs = gym.make(cfg.env_id, render_mode="rgb_array")
    else:
        envs = SubprocVecEnv([
            make_env(cfg.env_id)
            for _ in range(args.n_envs)
        ])
    eval_env = gym.make(cfg.env_id, render_mode="rgb_array")

    eval_freq = max(
        (cfg.TIMESTEPS * args.n_iterations) // (args.n_envs * args.n_evals),
        1
    )
    custom_eval_callback = CustomEvalCallback(
        eval_env,
        best_model_save_path=f"{cfg.models_dir}/best_model",
        log_path="./logs/eval",
        eval_freq=eval_freq,
        n_eval_episodes=args.n_eval_episodes,
        deterministic=True,
    )


    # TRAINING
    model = PPO("MlpPolicy", envs, verbose=False, tensorboard_log=cfg.log_dir)

    try:

        for i in range(1, args.n_iterations + 1):
            print(f"Iteración {i}")
            model.learn(
                total_timesteps=cfg.TIMESTEPS,
                reset_num_timesteps=False,
                tb_log_name=cfg.ALGORITHM,
                callback=custom_eval_callback,
                progress_bar=True
            )
            model.save(f"{cfg.models_dir}/{cfg.TIMESTEPS * i}")

        # Evaluate the agent
        mean_reward, std_reward = evaluate_policy(model, eval_env, n_eval_episodes=args.n_eval_episodes)
        print(f"Mean reward: {mean_reward} ± {std_reward}")

    finally:
        envs.close()
