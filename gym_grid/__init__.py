from gym.envs.registration import register

register(
    id='grid-v0',
    entry_point='gym_grid.envs:GridEnv'
)