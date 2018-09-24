import gym
import gym_test
import time

# from gym import envs
# env_ids = [env_spec.id for env_spec in envs.registry.all()]
# print(env_ids)

env = gym.make('haewoon-test-v0')
env.reset()
env.render()

while True:
    nextstate, reward, is_terminal, debug_info = env.step(
                env.action_space.sample())
    print (nextstate)
    env.render()
    if is_terminal:
        break    
    time.sleep(1)