from gymnasium.envs.registration import register

register(
    id="Snake-v0",
    entry_point="snake_env.env:SnakeEnv",
    max_episode_steps=1000,
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
    reward_threshold = 100,
    max_episode_steps = 1500,
    kwargs = { "apple_curriculum": True, "apples_rule": 300, "random_spawn_prob": 0.2}
)

register(
    id = "Snake_reward_decay-v0",
    entry_point = "snake_env.env:SnakeEnv",
    reward_threshold = 100,
    max_episode_steps = 1000,
    kwargs = {"reward_decay": True}
)


register(
    id = "Snake_curriculum_reward_decay-v0",
    entry_point = "snake_env.env:SnakeEnv",
    reward_threshold = 1000,
    max_episode_steps = 700,
    kwargs = {"reward_decay": True,
                "apple_curriculum": True, 
               "decay_rule": 3, 
               "apples_rule": 150, 
               "table_size": 160,
               "random_spawn_prob": 0.4}
)

register(
    id = "Snake_curriculum_reward_decay-v1",
    entry_point = "snake_env.env:SnakeEnv",
    reward_threshold = 1000,
    max_episode_steps = 700,
    kwargs = {"reward_decay": True,
                "apple_curriculum": True, 
               "decay_rule": 5, 
               "apples_rule": 300, 
               "table_size": 300, 
               "random_spawn_prob": 0.3}
)

register(
    id = "Snake_curriculum_reward_decay-v2",
    entry_point = "snake_env.env:SnakeEnv",
    reward_threshold = 1000,
    max_episode_steps = 700,
    kwargs = {"reward_decay": True,
            "apple_curriculum": False, 
               "decay_rule": 7, 
               "apples_rule": 500, 
               "table_size": 500, 
               "random_spawn_prob": 0.2}
)
