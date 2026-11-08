
import numpy as np
from dataclasses import dataclass
from collections.abc import Callable
from pathlib import Path
import optuna

import argparse

import torch

import gymnasium as gym
from stable_baselines3.common.callbacks import EvalCallback,BaseCallback
from stable_baselines3.common.torch_layers import FlattenExtractor
from stable_baselines3 import PPO



@dataclass(frozen=True)
class Config:
    ALGORITHM: str = "PPO"
    env_id: str = "Snake_curriculum_reward_decay-v0"
    models_dir: str = f"models/{ALGORITHM}"
    log_dir: str= "logs"
    apple_threshold: int = 100
    eval_max_steps: int = 5_000


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
        apple_threshold,
        deterministic=True,
        verbose=0,
        *,
        save_model=None,
    ):
        super().__init__(
            eval_env,
            log_path=log_path,
            eval_freq=eval_freq,
            n_eval_episodes=n_eval_episodes,
            deterministic=deterministic,
            verbose=verbose
        )
        self.apple_threshold = apple_threshold
        self.best_apples_model_path = Path(best_model_save_path) / "best_model"
        self.best_mean_apples = float("-inf")
        self._eval_apples: list[int] = []
        self._save_model = save_model

    def _log_success_callback(self, locals_, globals_):
        super()._log_success_callback(locals_, globals_)

        if locals_["done"]:
            apples = locals_["info"]["apples"]
            self._eval_apples.append(apples)
            self.logger.record_mean("eval/mean_apples", apples)

    def _on_step(self) -> bool:
        self._eval_apples.clear()
        continue_training = super()._on_step()
        infos = self.locals.get("infos", [])
        for info in infos:
            if "apples" in info:
                self.logger.record("custom/apples", info["apples"])
            if "snake_length" in info:
                self.logger.record("custom/snake_length", info["snake_length"])

        if self._eval_apples:
            mean_apples = sum(self._eval_apples) / len(self._eval_apples)
            if mean_apples > self.best_mean_apples:
                self.best_apples_model_path.parent.mkdir(parents=True, exist_ok=True)
                if self._save_model is None:
                    self.model.save(self.best_apples_model_path)
                else:
                    self._save_model(self.model, self.best_apples_model_path)
                self.best_mean_apples = mean_apples

            if mean_apples >= self.apple_threshold:
                print(f"Objetivo alcanzado con {mean_apples:.2f} manzanas por partida de media.")
                return False

        return continue_training


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
        score = calc_score(self.model, self.eval_env, n_episodes=self.n_eval_episodes)

        self.trial.report(score, self.num_timesteps)

        if score >self.best_score:
            self.best_score=score
        
        if self.trial.should_prune():
            raise optuna.TrialPruned()
        
        return True




class NormalizedPositions(FlattenExtractor):
    def __init__(self, observation_space, divisor):
        super().__init__(observation_space)
        escala = torch.ones(self.features_dim)
        escala[:4] = divisor
        self.register_buffer("escala", escala, persistent=False)

    def forward(self, observations):
        return super().forward(observations) / self.escala


def make_ppo_normalized(env, divisor, args, learning_rate=0.0006217264243836279):
    return PPO(
        "MlpPolicy",
        env,
        n_steps=4096,
        batch_size=64,
        n_epochs=13,
        learning_rate=learning_rate,
        gamma=0.9315601793271204,
        gae_lambda=0.91276989459545,
        clip_range=0.1312962772610166,
        ent_coef=0.0017946640158490546,
        device=args.device,
        tensorboard_log=cfg.log_dir,
        policy_kwargs={
            "features_extractor_class": NormalizedPositions,
            "features_extractor_kwargs": {"divisor": divisor},
        },
    )

def store_normalized_model(modelo, ruta, *, eval_env):
    with torch.random.fork_rng(devices=[]):
        model = PPO("MlpPolicy", eval_env, device="cpu", n_steps=modelo.n_steps)
    model.policy.load_state_dict(modelo.policy.state_dict())
    net = model.policy.mlp_extractor
    divisor = eval_env.unwrapped.tableSize - 10 * eval_env.unwrapped.action_size
    with torch.no_grad():
        net.policy_net[0].weight[:, :4].div_(divisor)
        net.value_net[0].weight[:, :4].div_(divisor)
    model.save(ruta)



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
            "--n_steps",
            type=int,
            default=1024,
            help="Number of steps before updating policy"
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
        "--device",
        type=str,
        default="cuda",
        help="Device ['cpu', 'cuda', 'auto' (default)]."
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=123,
        help="Random seed to use (default=123)."
    )

    return parser.parse_args()
