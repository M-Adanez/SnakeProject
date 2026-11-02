import optuna
import gymnasium as gym
from .utils import cfg,calc_score,OptunaCallback
import numpy as np

from snake_env.env import SnakeEnv
import argparse

from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.vec_env import SubprocVecEnv, DummyVecEnv
from stable_baselines3 import PPO


        
def parse_args():

    parser = argparse.ArgumentParser(
        description="Optuma Hyperpatameter Optimiztion for Snake"
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
        default=int(1e7),
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
    parser.add_argument(
        "--seed",
        type=int,
        default=123,
        help="Random seed to use (default=123)."
    )

    parser.add_argument(
        "--n_trials",
        type=int,
        default=600,
        help="N_trials fot optuna (default=600)."
    )

    return parser.parse_args()

args = parse_args()


def objective(trial: optuna.Trial) -> float:
    train_env=make_vec_env(
        SnakeEnv,
        n_envs=args.n_envs,
        seed=args.seed,
        vec_env_cls=SubprocVecEnv if args.n_envs > 1 else DummyVecEnv,
    )

    eval_env=gym.make(cfg.env_id, render_mode="rgb_array")
    params = {
        "learning_rate": trial.suggest_float("learning_rate", 1e-6, 1e-3,log=True),
        "n_steps": trial.suggest_int("n_steps", 512, 4096),
        "batch_size": trial.suggest_int("batch_size", 32, 256),
        "n_epochs": trial.suggest_int("epochs", 3, 15),
        "gamma": trial.suggest_float("gamma", 0.8,0.999),
        "gae_lambda": trial.suggest_float("gae_lambda", 0.9, 0.99),
        "clip_range":trial.suggest_float("clip_range",0.05,0.3),
        "ent_coef":trial.suggest_float("ent_coef", 0.01, 0.02)
    }
    try:
        model=PPO(
            "MlpPolicy",
            train_env,
            verbose=0,
            seed=args.seed,
            **params
        )
        
        eval_freq = max(1, args.total_timesteps // args.n_evals)      
        callback=OptunaCallback(trial,eval_env,args.n_eval_episodes,eval_freq=eval_freq)

        model.learn(total_timesteps=args.total_timesteps,callback=callback)

        mean_reward = calc_score(model,eval_env, n_episodes=args.n_eval_episodes)

    finally:
        train_env.close()
        eval_env.close()

    
    return mean_reward

if __name__ == "__main__":
    # Suppress verbose logs if needed: optuna.logging.set_verbosity(optuna.logging.WARNING)

    # 4. Create study (direction: "maximize" for accuracy/F1, "minimize" for loss/MSE)
    study = optuna.create_study(
        study_name="snake_optim",
        direction="maximize",
        sampler=optuna.samplers.TPESampler(seed=args.seed, n_startup_trials=20),
        pruner=optuna.pruners.MedianPruner(
            n_startup_trials=10,
            n_warmup_steps=20_000,
            interval_steps=10_000
        ),
        storage="sqlite:///logs/optuna_logs.db",
        load_if_exists=True
    )

    # 5. Execute optimization
    study.optimize(objective, n_trials=args.n_trials,show_progress_bar=True)

    # 6. Print best results
    print(f"Number of finished trials: {len(study.trials)}")
    print(f"Best trial value: {study.best_value:.4f}")
    print("Best hyperparameters:")
    for key, value in study.best_params.items():
        print(f"    {key}: {value}")

