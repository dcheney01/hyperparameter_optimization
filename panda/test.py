import numpy as np
import robosuite as suite
from simple_env import SimpleEnv


env = SimpleEnv()
# reset the environment
env.reset()

steps = 1000

for i in range(steps):
    action = np.random.randn(env.robots[0].dof) # sample random action
    
    obs, reward, done, info = env.step(action)  # take action in the environment
    # print(f'action: {action}')
    # print(f'obs: {obs}')
    # print(f'info: {info}')


    env.render()  # render on display