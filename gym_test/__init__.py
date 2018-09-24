from gym.envs.registration import register

register(
    id='haewoon-test-v0',
    entry_point='gym_test.envs:TestEnv',
)