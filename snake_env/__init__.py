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
    id = "Snake_curriculum-v0",
    entry_point = "snake_env.env:SnakeEnv",
    reward_threshold = 50,
    max_episode_steps = 1500,
    kwargs = {"render_mode":"human", "apple_curriculum": True, "apples_rule": 10000}
)

register(
    id = "Snake_reward_decay-v0",
    entry_point = "snake_env.env:SnakeEnv",
    reward_threshold = 50,
    max_episode_steps = 1500,
    kwargs = {"reward_decay": True}
)


register(
    id = "Snake_curriculum_reward_decay-v0",
    entry_point = "snake_env.env:SnakeEnv",
    reward_threshold = 50,
    max_episode_steps = 1500,
    kwargs = {"reward_decay": True, "apple_curriculum": True}
)