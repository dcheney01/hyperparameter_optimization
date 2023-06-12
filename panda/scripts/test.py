import numpy as np
import robosuite as suite
from robosuite import load_controller_config
from hyperparam_optimization.panda.scripts.simple_env import SimpleEnv
import time
np.set_printoptions(precision=4)
datapoints = 300


controller_config = load_controller_config(default_controller='JOINT_POSITION')
env = SimpleEnv(controller_configs=controller_config)
# 
trajectory_len = 32

# for i in range(datapoints):
#     num_joints = np.random.randint(1,7)
#     joints_to_control = np.random.randint(0, 7, num_joints) # 7 joints
#     print(f'joints_to_control: {joints_to_control}')
#     torques = np.random.randn(num_joints) # sample random action
#     action = np.zeros((8,))
#     action[joints_to_control] = torques

#     env.reset()
#     # env.robots[0].reset(deterministic=False)
#     env.robots[0].set_robot_joint_positions(np.zeros((7,)))
for i in range(10):
    random_position = np.random.randn(7)
    env.reset(init_qpos=random_position)
    # env.robots[0].init_qpos = random_position
    # env.robots[0].set_robot_joint_positions(random_position)
    # env.robots[0].set_robot_joint_velocities(np.zeros((7,)))
    action = np.zeros((8,))
    action[:7] = random_position

    for j in range(datapoints):
        obs, reward, done, info = env.step(action)  # take action in the environment
        # print(f'action: {action}')
        # print(f'obs: {obs}')
        # print(f'info: {info}')
        if done:
            print('reached a done episode')
            break

        env.render()  # render on display
        # time.sleep(1)