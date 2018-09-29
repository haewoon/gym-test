from gym.envs.registration import register

register(
    id='haewoon-maze1d-v0',
    entry_point='gym_test.envs:Maze1dEnv',
)


register(
    id='haewoon-maze2d-v0',
    entry_point='gym_test.envs:Maze2dEnv',
)