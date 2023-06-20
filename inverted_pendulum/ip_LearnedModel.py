"""
This script defines an inverted pendulum class, where the dynamics are modelled using a neural network.

State Vector x: [thetadot, theta].T

Inputs u: [torque_origin]
"""

import numpy as np
import torch
import matplotlib.pyplot as plt
import json

from ip_optimize import InvertedPendulumLightningModule
from InvertedPendulum import InvertedPendulum

class IP_LearnedModel(InvertedPendulum):
    def __init__(self, model_path=None, config=None, use_gpu=True):
        super().__init__()
        """
        Defined in InvertedPendulum.py:
            self.numStates = 2
            self.numInputs = 1
            self.uMax = 1.0
            self.uMin = -self.uMax
            self.l = 1.0

            self.xMax = np.array([[10.0*np.pi],[2.0*np.pi]])
            self.xMin = np.array([[-10.0*np.pi],[-2.0*np.pi]])
            self.wrapAngle = np.array([False,False])
            self.wrapRange = (-np.pi,np.pi)
        """
        # self.num_outputs = 2
        # self.num_inputs = 3
        
        self.use_gpu = use_gpu
        self.config = config
        
        if config['normalized_data']:
            self.xScale = self.xMax
            self.uScale = self.uMax
        else:
            self.xScale = np.ones(self.xMax.shape)
            self.uScale = np.ones(self.numInputs)

        if model_path is None:
            self.model = InvertedPendulumLightningModule(self.config)
        else:
            self.model = InvertedPendulumLightningModule.load_from_checkpoint(model_path, config=self.config)

        if self.use_gpu:
            self.model = self.model.cuda()

        self.model.eval()

    def calc_state_derivs(self,x,u):
        xNorm = x/self.xScale
        uNorm = u/self.uScale
        inputs = np.vstack([xNorm,uNorm]).T
        inputs_tensor = torch.tensor(inputs).float()
        if self.use_gpu==True:
            inputs_tensor = inputs_tensor.cuda()

        xdot = self.model(inputs_tensor).cpu().detach().numpy().T
        xdot *= self.xScale # will be unchanged for non-normalized data

        return xdot # [theta_doubledot, theta_dot]
    
    def forward_simulate_dt(self, x, u, dt):
        """
        Returns x at the next time step.
        Integration options: euler, RK4.

        Can also just return the output of the model if the model learned the next state instead of change in state

        This is the basic function to be used for simulation.

        If angle wrapping is desired, wrapAngle can be set to boolean numpy array with True values in the positions of angles to be wrapped. wrapRange controls the interval at which to wrap.
        """
        xcols = np.shape(x)[1]

        x = np.clip(x, self.xMin, self.xMax)
        u = np.clip(u, self.uMin, self.uMax)

        if self.config['learn_mode'] == 'x_dot':
            x_dot = self.calc_state_derivs(x, u)
            x = x + x_dot*dt
        elif self.config['learn_mode'] == 'x':
            x = self.calc_state_derivs(x, u)
        else:
            raise ValueError(f"Invalid learn_mode. Must be x or x_dot. Got {self.config['learn_mode']}")

        if self.wrapAngle.any() == True:
            low = self.wrapRange[0]
            high = self.wrapRange[1]
            cycle = high-low
            x[self.wrapAngle] = (x[self.wrapAngle]+cycle/2) % cycle + low

        # make sure output is expected size
        assert np.shape(x) == (self.numStates, xcols),\
            "Something went wrong, output vector is not correct size:"\
            "\nExpected size:({},n)\nActual size:{}"\
            .format(self.numStates, np.shape(x))

        return x
        

if __name__=='__main__':
    trial_dir = '/home/daniel/research/catkin_ws/src/hyperparam_optimization/inverted_pendulum/run_logs/fnn_optimization/train_21f01584_85_accuracy_tolerance=0.0100,act_fn=relu,b_size=16,calculates_xdot=False,cpu_num=3,dataset_size=60000,dt=0.0100,gen_2023-05-03_20-58-32'

    # get the checkpoint
    import glob
    checkpoint_path = glob.glob(trial_dir + '/lightning_logs/version_0/checkpoints/*.ckpt')[0]

    # get the config file
    with open(trial_dir + '/params.json', 'r') as f:
        config = json.load(f)

    print(json.dumps(config, indent=4))
    
    learned_system = IP_LearnedModel(model_path=checkpoint_path, config=config)  
    x_learned = np.zeros([2,1])
    x_learned[1] = .001
    u_learned = np.zeros([1,1])    

    analytical_system = InvertedPendulum()
    x_analytical = np.zeros([2,1])
    x_analytical[1] = .001
    u_analytical = np.zeros([1,1]) 

    dt = config['dt']
    sim_time = 10
    horizon = int(sim_time/dt)

    x_learned_list = []
    x_analytical_list = []
    xdot_learned_list = []
    xdot_analytical_list = []

    plt.ion()

    for i in range(0, horizon):
        x_learned = learned_system.forward_simulate_dt(x_learned, u_learned, dt)
        x_analytical = analytical_system.forward_simulate_dt(x_analytical, u_analytical, dt)

        x_learned_list.append(x_learned[0])
        x_analytical_list.append(x_analytical[0])
        xdot_learned_list.append(x_learned[1])
        xdot_analytical_list.append(x_analytical[1])

        if i%5==0:
            plt.figure(1)
            learned_system.visualize(x_learned, u_learned)

            plt.figure(2)
            analytical_system.visualize(x_analytical, u_analytical)    

    time_steps = [i for i in range(horizon)]

    plt.ioff()
    plt.figure()
    plt.plot(time_steps, x_learned_list, "-b", label='x_learned')
    plt.plot(time_steps, x_analytical_list, "-r", label='x_analytical')
    plt.legend(loc="upper right")

    plt.figure()
    plt.plot(time_steps, xdot_learned_list, "-b", label='xdot_learned')
    plt.plot(time_steps, xdot_analytical_list, "-r", label='xdot_analytical')
    plt.legend(loc="upper right")

    plt.show()