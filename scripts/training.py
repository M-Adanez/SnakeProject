
from functools import partial
from pathlib import Path
import sys

ROOT_DIR = Path(__file__).resolve().parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.append(str(ROOT_DIR))
import snake_env

from utils import (
    cfg,
    CustomEvalCallback,
    parse_args,
    make_ppo_normalized,
    store_normalized_model
)

import gymnasium as gym
from stable_baselines3 import PPO
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common.vec_env import SubprocVecEnv, DummyVecEnv

from stable_baselines3.common.utils import LinearSchedule



def main():
    args = parse_args()
    if args.total_timesteps < 3:
        raise ValueError("Se necesitan al menos tres pasos para repartir entre las fases")
    small_steps = max(1, args.total_timesteps // 5)
    medium_steps = max(1, args.total_timesteps * 3 // 10)
    large_steps = args.total_timesteps - small_steps - medium_steps
    stages = (
        ("Snake_curriculum_reward_decay-v0", small_steps),
        ("Snake_curriculum_reward_decay-v1", medium_steps),
        ("Snake_curriculum_reward_decay-v2", large_steps),
    )

    models_dir = Path(cfg.models_dir)
    models_dir.mkdir(parents=True, exist_ok=True)


    eval_max_steps = gym.spec("Snake-v0").max_episode_steps or cfg.eval_max_steps
    eval_env = Monitor(gym.make(
        "Snake-v0", render_mode=None, max_episode_steps=eval_max_steps
    ))
    exportar = partial(store_normalized_model, eval_env=eval_env)
    model = None

    try:
        eval_cb = CustomEvalCallback(
            eval_env,
            best_model_save_path=models_dir / "best_model",
            log_path=Path(cfg.log_dir) / "eval",
            eval_freq=max(large_steps // (args.n_envs * args.n_evals), 1),
            n_eval_episodes=args.n_eval_episodes,
            apple_threshold=cfg.apple_threshold,
            deterministic=True,
            save_model=exportar,
        )

        for stage, (env_id, timesteps) in enumerate(stages, start=1):
            envs = make_vec_env(
                env_id,
                n_envs=args.n_envs,
                seed=args.seed,
                vec_env_cls=SubprocVecEnv if args.n_envs > 1 else DummyVecEnv,
            )
            try:
                size = envs.get_attr("tableSize", indices=0)[0]
                step = 10 * envs.get_attr("action_size", indices=0)[0]
                divisor = size - step
                next_model = make_ppo_normalized(envs, divisor, args, learning_rate = 0.00062, gamma = 0.93, gae_lambda = 0.91, clip_range = 0.13)
                if model is not None:
                    next_model.policy.load_state_dict(model.policy.state_dict())
                model = next_model

                print(f"Fase {stage}/{len(stages)} · {size}×{size} · {timesteps:,} pasos")
                model.learn(
                    total_timesteps=timesteps,
                    tb_log_name=f"{cfg.ALGORITHM}_{size}",
                    callback=eval_cb if stage == len(stages) else None,
                    progress_bar=True,
                )
            finally:
                envs.close()

        exportar(model, models_dir / "final_model_V2")
        mean, std = evaluate_policy(model, eval_env, n_eval_episodes=args.n_eval_episodes)
        print(f"Final model - mean reward: {mean:.2f} ± {std:.2f}")

        if eval_cb.best_mean_apples >= 0:
            best = PPO.load(eval_cb.best_apples_model_path.with_suffix(".zip"), device=args.device)
            mean, std = evaluate_policy(best, eval_env, n_eval_episodes=args.n_eval_episodes)
            print(f"Best model - mean reward: {mean:.2f} ± {std:.2f}")

    except KeyboardInterrupt:
        if model is not None:
            print("Interrupted, saving model...")
            exportar(model, models_dir / "interrupted_model")
    finally:
        eval_env.close()


if __name__ == "__main__":
    main()
