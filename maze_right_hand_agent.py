import gym
import gym_test
import time

# Solving a maze by using right hand rule.

env = gym.make('haewoon-maze2d-v0')
current_state = env.reset()
env.render()

orientation = 1

while True:
    print ("Orientation: {}".format(["Left","Down","Right","Up"][orientation]))
    next_state, reward, is_terminal, debug_info = env.step(orientation)
    if current_state == next_state:
        orientation = (orientation+1) % env.action_space.n
    else:
        orientation = (orientation-1) % env.action_space.n
    current_state = next_state
    env.render()
    if is_terminal:
        break    
    time.sleep(1)