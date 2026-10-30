import gymnasium as gym
import snake_env
from stable_baselines3 import PPO
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common.callbacks import EvalCallback
import os

class CustomEvalCallback(EvalCallback):
    def __init__(self, eval_env, best_model_save_path, log_path, eval_freq, deterministic,verbose=0):
        super().__init__(
            eval_env,
            best_model_save_path=best_model_save_path,
            log_path=log_path,
            eval_freq=eval_freq,
            deterministic=deterministic,
            verbose=verbose
        )


    def _on_step(self) -> bool:
        super()._on_step()
        infos = self.locals.get("infos", [])
        for info in infos:
            # Ejemplo: Si tu env.step() devuelve info = {"score": score, "snake_length": length}
            if "apples" in info:
                self.logger.record("custom/apples", info["apples"])
            if "snake_length" in info:
                self.logger.record("custom/snake_length", info["snake_length"])

        return True  # Devuelve True para continuar el entrenamiento

ALGORITHM = "PPO"
models_dir = f"models/{ALGORITHM}"
log_dir = "logs"

if not os.path.exists(models_dir):
    os.makedirs(models_dir)

if not os.path.exists(log_dir):
    os.makedirs(log_dir)

# Create the environment
env = gym.make("Snake-v0", render_mode="rgb_array")
eval_env = gym.make("Snake-v0", render_mode="rgb_array")

# eval_callback = EvalCallback(
#     eval_env,
#     best_model_save_path=f"{models_dir}/best_model",
#     log_path="./logs/eval",
#     eval_freq=10000,
#     deterministic=True,
# )
custom_eval_callback = CustomEvalCallback(
    eval_env,
    best_model_save_path=f"{models_dir}/best_model",
    log_path="./logs/eval",
    eval_freq=10000,
    deterministic=True,
)

# Instantiate the agent
model = PPO("MlpPolicy", env, verbose=False, tensorboard_log=log_dir)

TIMESTEPS = 100003
NUM_ITERATIONS = 10  # Adjust according to your needs

for i in range(1, NUM_ITERATIONS + 1):
    print(f"Iteración {i}")
    model.learn(
        total_timesteps=TIMESTEPS,
        reset_num_timesteps=True, # usado para seguir o no el training con el num de timesteps global
        tb_log_name=ALGORITHM,
        callback=custom_eval_callback,
        progress_bar=True
    )
    model.save(f"{models_dir}/{TIMESTEPS * i}")

# Evaluate the agent
mean_reward, std_reward = evaluate_policy(model, env, n_eval_episodes=10)
print(f"Mean reward: {mean_reward} ± {std_reward}")

env.close()
