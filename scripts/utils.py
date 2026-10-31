
from dataclasses import dataclass
from collections.abc import Callable

import gymnasium as gym
from stable_baselines3.common.callbacks import EvalCallback


@dataclass(frozen=True)
class Config:
    ALGORITHM: str = "PPO"
    TIMESTEPS: int = 100_000
    env_id: str = "Snake-v0"

    models_dir: str = f"models/{ALGORITHM}"
    log_dir: str= "logs"


cfg = Config()

def make_env(env_id: str) -> Callable:
    def _init() -> gym.Env:
        env = gym.make(env_id)
        return env
    return _init

class CustomEvalCallback(EvalCallback):
    def __init__(
        self,
        eval_env,
        best_model_save_path,
        log_path,
        eval_freq,
        n_eval_episodes,
        deterministic=True,
        verbose=0
    ):
        super().__init__(
            eval_env,
            best_model_save_path=best_model_save_path,
            log_path=log_path,
            eval_freq=eval_freq,
            n_eval_episodes=n_eval_episodes,
            deterministic=deterministic,
            verbose=verbose
        )


    def _on_step(self) -> bool:
        super()._on_step()
        infos = self.locals.get("infos", [])
        for info in infos:
            if "apples" in info:
                self.logger.record("custom/apples", info["apples"])
            if "snake_length" in info:
                self.logger.record("custom/snake_length", info["snake_length"])

        return True  # Devuelve True para continuar el entrenamiento
