from gymnasium.envs.registration import register

register(
    id="Snake-v0",
    entry_point="snake_env.env:SnakeEnv",
    max_episode_steps=None,
)

register(
    id="Snake-v0-step5",
    entry_point="snake_env.env:SnakeEnv",
    max_episode_steps=500,
    kwargs={"action_size":2},
)

register(
    id="Snake-v1",
    entry_point="snake_env.env:SnakeEnv",
    max_episode_steps=100,
)

register(
    id="Snake-v2",
    entry_point="snake_env.env:SnakeEnv",
    max_episode_steps=500,
    reward_threshold=30,
)

# new ENVs
register(
    id = "Snake-curriculum",
    entry_point = "snake_env.env:SnakeEnv",
    reward_threshold = 50,
    max_episode_steps = 1500,
    kwards = {"apple_curriculum": True}
)

register(
    id = "Snake-reward_decay",
    entry_point = "snake_env.env:SnakeEnv",
    reward_threshold = 50,
    max_episode_steps = 1500,
    kwards = {"reward_decay": True}
)


register(
    id = "Snake-curriculum+reward_decay",
    entry_point = "snake_env.env:SnakeEnv",
    reward_threshold = 50,
    max_episode_steps = 1500,
    kwards = {"reward_decay": True, "apple_curriculum": True}
)