from pathlib import Path
import sys

ROOT_DIR = Path(__file__).resolve().parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.append(str(ROOT_DIR))
import snake_env

import gymnasium as gym
from stable_baselines3.common.callbacks import EvalCallback
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv

from utils import cfg, parse_args, make_ppo_normalized, store_normalized_model


def main():
    args = parse_args()
    env_id = "Snake-v0"
    model_name = "PPO_baseline"
    models_dir = Path(cfg.models_dir)
    models_dir.mkdir(parents=True, exist_ok=True)

    eval_limit = gym.spec(env_id).max_episode_steps or cfg.eval_max_steps
    with Monitor(gym.make(env_id, max_episode_steps=eval_limit)) as eval_env:
        envs = make_vec_env(
            env_id,
            n_envs=args.n_envs,
            seed=args.seed,
            vec_env_cls=SubprocVecEnv if args.n_envs > 1 else DummyVecEnv,
        )
        try:
            snake = eval_env.unwrapped
            model = make_ppo_normalized(envs, snake.tableSize - 10 * snake.action_size, args)
            eval_callback = EvalCallback(
                eval_env,
                eval_freq=max(args.total_timesteps // (args.n_envs * args.n_evals), 1),
                n_eval_episodes=args.n_eval_episodes,
                deterministic=True,
                verbose=0,
            )
            model.learn(
                total_timesteps=args.total_timesteps,
                callback=eval_callback,
                tb_log_name=model_name,
                progress_bar=True,
            )
            store_normalized_model(model, models_dir / model_name, eval_env=eval_env)
        finally:
            envs.close()


if __name__ == "__main__":
    main()
