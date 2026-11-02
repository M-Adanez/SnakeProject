
from dataclasses import dataclass
from collections.abc import Callable
import optuna

import gymnasium as gym
from stable_baselines3.common.callbacks import EvalCallback,BaseCallback



@dataclass(frozen=True)
class Config:
    ALGORITHM: str = "PPO"
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


def calc_score(model,env,n_episodes=10):
    scores=[]

    for _ in range(n_episodes):
        obs,_ = env.reset()
        done=False

        while not done:
            action,_ =model.predict(obs, deterministic=True)

            obs, reward, terminated, truncated, info =env.step(action)

            done = terminated or truncated

        scores.append(info["apples"])

    return np.mean(scores)


class OptunaCallback(BaseCallback):
    def __init__(self, trial,eval_env,n_eval_episodes,eval_freq):
        super().__init__()
        self.trial=trial
        self.eval_env=eval_env
        self.best_score = 0
        self.n_eval_episodes=n_eval_episodes
        self.eval_freq=eval_freq

    def _on_step(self):
        if self.n_calls % self.eval_freq != 0:
            return True
        score = calc_score(self.model, self.eval_env, n_episodes=self.n_eval_episodes,deterministic=True)

        self.trial.report(score, self.num_timesteps)

        if score >self.best_score:
            self.best_score=score
        
        if self.trial.should_prine():
            raise optuna.TrialPruned()
        
        return True