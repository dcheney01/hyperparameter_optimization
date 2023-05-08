import numpy as np
import robosuite as suite
from simple_env import SimpleEnv
import time
np.set_printoptions(precision=4)
datapoints = 1000

env = SimpleEnv()
# reset the environment
env.reset()

trajectory_len = 1000


for i in range(datapoints):
    num_joints = np.random.randint(1,7)
    joints_to_control = np.random.randint(0,7, num_joints) # 7 joints
    print(f'joints_to_control: {joints_to_control}')
    torques = np.random.randn(num_joints) # sample random action
    action = np.zeros((8,))
    action[joints_to_control] = torques

    env.reset()
    # env.robots[0].reset(deterministic=False)
    robot.set_robot_joint_positions(joint_positions)

    for j in range(trajectory_len):
        obs, reward, done, info = env.step(action)  # take action in the environment
        print(f'action: {action}')
        # print(f'obs: {obs}')
        # print(f'info: {info}')
        if done:
            print('reached a done episode')
            break

        env.render()  # render on display
    time.sleep(1)