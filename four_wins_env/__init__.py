from gym.envs.registration import register

register(
    id='FourWins-v0',
    entry_point='four_wins_env.four_wins:FourWinsEnv',
)