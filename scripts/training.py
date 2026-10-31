
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
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.callbacks import CheckpointCallback
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common.vec_env import SubprocVecEnv, DummyVecEnv

def parse_args():
    parser = argparse.ArgumentParser(
        description="Trains a RL algorithm on the Snake env."
    )

    parser.add_argument(
        "--n_envs",
        type=int,
        default=8,
        help="Number of parallel environments (default=8)"
    )
    parser.add_argument(
        "--total_timesteps",
        type=int,
        default=1e7,
        help="Number of training timesteps (default=10_000_000)."
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
    envs = make_vec_env(
        cfg.env_id,
        n_envs=args.n_envs,
        seed=args.seed,
        vec_env_cls=SubprocVecEnv if args.n_envs > 1 else DummyVecEnv,
    )
    eval_env = Monitor(gym.make(cfg.env_id, render_mode="rgb_array"))


    eval_freq = max(
        args.total_timesteps // (args.n_envs * args.n_evals),
        1
    )
    eval_cb = CustomEvalCallback(
        eval_env,
        best_model_save_path=f"{cfg.models_dir}/best_model",
        log_path=f"{cfg.log_dir}/eval",
        eval_freq=eval_freq,
        n_eval_episodes=args.n_eval_episodes,
        deterministic=True,
    )
    checkpt_cb = CheckpointCallback(
        save_freq=eval_freq,
        save_path=cfg.models_dir,
        name_prefix="ppo_snake",
    )


    # MODEL
    model = PPO("MlpPolicy", envs, verbose=False, tensorboard_log=cfg.log_dir)

    # TRAINING
    try:
        model.learn(
            total_timesteps=args.total_timesteps,
            reset_num_timesteps=False,
            tb_log_name=cfg.ALGORITHM,
            callback=eval_cb,
            progress_bar=True
        )
        model.save(f"{cfg.models_dir}/final_model")

        # final model eval
        mean, std = evaluate_policy(model, eval_env, n_eval_episodes=args.n_eval_episodes)
        print(f"Final model  - mean reward: {mean:.2f} ± {std:.2f}")

        # best model eval
        best_path = f"{cfg.models_dir}/best_model/best_model.zip"
        if os.path.exists(best_path):
            best = PPO.load(best_path, device=args.device)
            mean, std = evaluate_policy(best, eval_env, n_eval_episodes=args.n_eval_episodes)
            print(f"Best model - mean reward: {mean:.2f} ± {std:.2f}")

    except KeyboardInterrupt:
        print("Interrupted, saving model...")
        model.save(f"{cfg.models_dir}/interrupted_model")

    finally:
        envs.close()
        eval_env.close()
