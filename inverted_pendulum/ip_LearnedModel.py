"""
This script defines an inverted pendulum class, where the dynamics are modelled using a neural network.

State Vector x: [thetadot, theta].T

Inputs u: [torque_origin]
"""

import numpy as np
import torch
import matplotlib.pyplot as plt
import json, sys

sys.path.append('/home/daniel/research/catkin_ws/src/')
from ip_optimize import InvertedPendulumLightningModule
from InvertedPendulum import InvertedPendulum
from hyperparam_optimization.BASE.NN_Architectures import *

class IP_LearnedModel(InvertedPendulum):
    def __init__(self, model_path=None, config=None, use_gpu=True):
        super().__init__
        self.num_outputs = 2
        self.num_inputs = 3
        self.numStates = self.num_outputs
        self.numInputs = 1
        self.use_gpu = use_gpu
        self.uMax = np.array([1.0])
        self.uMin = -self.uMax
        self.l = 1.0
        self.wrapAngle = np.array([False,True])
        self.wrapRange = (-np.pi,np.pi)
        self.config = config
        
        self.xMax = np.array([[10.0*np.pi],[2.0*np.pi]])
        self.xMin = np.array([[-10.0*np.pi],[-2.0*np.pi]])

        if config['normalized_data']:
            self.xScale = self.xMax
            self.uScale = self.uMax
        else:
            self.xScale = np.ones(self.xMax.shape)
            self.uScale = np.ones(self.uMax.shape)

        if model_path is None:
            self.model = InvertedPendulumLightningModule(self.config)
        else:
            self.model = InvertedPendulumLightningModule.load_from_checkpoint(model_path, config=self.config)

        if self.use_gpu==True:
            self.model = self.model.cuda()

        self.model.eval()

    def calc_state_derivs(self,x,u):
        # Find theta_dot and theta_doubledot
        xNorm = x/self.xScale
        uNorm = u/self.uScale
        inputs = np.vstack([xNorm,uNorm]).T
        inputs_tensor = torch.tensor(inputs).float()
        if self.use_gpu==True:
            inputs_tensor = inputs_tensor.cuda()

        xdot = self.model(inputs_tensor).cpu().detach().numpy().T

        if self.config['normalized_data']:
            xdot *= self.xScale

        return xdot
    
    def forward_simulate_dt(self, x, u, dt, method='RK4'):
        """
        Returns x at the next time step.
        Integration options: euler, RK4.

        Can also just return the output of the model if the model learned the next state instead of change in state

        This is the basic function to be used for simulation.

        If angle wrapping is desired, wrapAngle can be set to boolean numpy array with True values in the positions of angles to be wrapped. wrapRange controls the interval at which to wrap.
        """
        # make sure inputs are the correct sizes
        # u = u.reshape([self.numInputs,u.shape[1]])
        # x = x.reshape([self.numStates, x.shape[1]])
        xcols = np.shape(x)[1]

        # enforce state constraints
        x = np.clip(x, self.xMin, self.xMax)

        # # enforce input constraints
        u = np.clip(u,self.uMin,self.uMax)

        if self.config['learn_mode'] == 'x_dot':
            if method == 'euler':
                x_dot = self.calc_state_derivs(x, u)
                x = x + x_dot*dt
            elif method == 'RK4':
                F1 = self.calc_state_derivs(x, u)
                F2 = self.calc_state_derivs(x + dt/2 * F1, u)
                F3 = self.calc_state_derivs(x + dt/2 * F2, u)
                F4 = self.calc_state_derivs(x + dt * F3, u)
                x += dt / 6 * (F1 + 2 * F2 + 2 * F3 + F4)
            else:
                s = (method + ' is unimplemented. Using 1st Order Euler instead. "' + method +
                    '" sounds like a fun method, though! You should implement it! Consider looking at https://docs.scipy.org/doc/scipy/reference/generated/scipy.integrate.ode.html')
                print(s)
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
    # params_path = '/home/daniel/research/catkin_ws/src/hyperparam_optimization/inverted_pendulum/run_logs/fnn_optimization/train_7ad1aaeb_156_accuracy_tolerance=0.0100,act_fn=relu,b_size=32,calculates_xdot=False,cpu_num=3,dataset_size=60000,dt=0.0100,ge_2023-05-04_03-41-09/params.json'
    # checkpoint_path = '/home/daniel/research/catkin_ws/src/hyperparam_optimization/inverted_pendulum/run_logs/fnn_optimization/train_7ad1aaeb_156_accuracy_tolerance=0.0100,act_fn=relu,b_size=32,calculates_xdot=False,cpu_num=3,dataset_size=60000,dt=0.0100,ge_2023-05-04_03-41-09/lightning_logs/version_0/checkpoints/epoch=498-step=748500.ckpt'

    # params_path = '/home/daniel/research/catkin_ws/src/hyperparam_optimization/inverted_pendulum/run_logs/fnn_optimization/train_21f01584_85_accuracy_tolerance=0.0100,act_fn=relu,b_size=16,calculates_xdot=False,cpu_num=3,dataset_size=60000,dt=0.0100,gen_2023-05-03_20-58-32/params.json'
    # checkpoint_path = '/home/daniel/research/catkin_ws/src/hyperparam_optimization/inverted_pendulum/run_logs/fnn_optimization/train_21f01584_85_accuracy_tolerance=0.0100,act_fn=relu,b_size=16,calculates_xdot=False,cpu_num=3,dataset_size=60000,dt=0.0100,gen_2023-05-03_20-58-32/lightning_logs/version_0/checkpoints/epoch=498-step=1497000.ckpt'
    
    # params_path = '/home/daniel/research/catkin_ws/src/hyperparam_optimization/inverted_pendulum/run_logs/fnn_optimization/train_2528e9d6_184_accuracy_tolerance=0.0100,act_fn=relu,b_size=16,calculates_xdot=False,cpu_num=3,dataset_size=60000,dt=0.0100,ge_2023-05-04_04-34-57/params.json'
    # checkpoint_path = '/home/daniel/research/catkin_ws/src/hyperparam_optimization/inverted_pendulum/run_logs/fnn_optimization/train_2528e9d6_184_accuracy_tolerance=0.0100,act_fn=relu,b_size=16,calculates_xdot=False,cpu_num=3,dataset_size=60000,dt=0.0100,ge_2023-05-04_04-34-57/lightning_logs/version_0/checkpoints/epoch=498-step=1497000.ckpt'

    params_path = '/home/daniel/research/catkin_ws/src/hyperparam_optimization/inverted_pendulum/run_logs/fnn_optimization/train_716827a7_60_accuracy_tolerance=0.0100,act_fn=relu,b_size=16,calculates_xdot=False,cpu_num=3,dataset_size=60000,dt=0.0100,gen_2023-05-03_16-45-47/params.json'
    checkpoint_path = '/home/daniel/research/catkin_ws/src/hyperparam_optimization/inverted_pendulum/run_logs/fnn_optimization/train_716827a7_60_accuracy_tolerance=0.0100,act_fn=relu,b_size=16,calculates_xdot=False,cpu_num=3,dataset_size=60000,dt=0.0100,gen_2023-05-03_16-45-47/lightning_logs/version_0/checkpoints/epoch=498-step=1497000.ckpt'

    # params_path = '/home/daniel/research/catkin_ws/src/hyperparam_optimization/inverted_pendulum/run_logs/fnn_optimization/train_e49cea1a_125_accuracy_tolerance=0.0100,act_fn=relu,b_size=32,calculates_xdot=False,cpu_num=3,dataset_size=60000,dt=0.0100,ge_2023-05-04_01-56-18/params.json'
    # checkpoint_path = '/home/daniel/research/catkin_ws/src/hyperparam_optimization/inverted_pendulum/run_logs/fnn_optimization/train_e49cea1a_125_accuracy_tolerance=0.0100,act_fn=relu,b_size=32,calculates_xdot=False,cpu_num=3,dataset_size=60000,dt=0.0100,ge_2023-05-04_01-56-18/lightning_logs/version_0/checkpoints/epoch=498-step=748500.ckpt'

    with open(params_path, 'r') as f:
        config = json.load(f)

    print(json.dumps(config))
    
    learned_system = IP_LearnedModel(model_path=checkpoint_path, config=config)  
    x_learned = np.zeros([2,1])
    x_learned[1] = .001
    u_learned = np.zeros([1,1])    

    analytical_system = InvertedPendulum()
    x_analytical = np.zeros([2,1])
    x_analytical[1] = .001
    u_analytical = np.zeros([1,1]) 

    dt = config['dt']
    plt.ion()
    horizon = 1200
    x_learned_list = []
    x_analytical_list = []
    xdot_learned_list = []
    xdot_analytical_list = []

    x_error = []
    xdot_error = []

    for i in range(0, horizon):
        x_learned = learned_system.forward_simulate_dt(x_learned, u_learned, dt)
        x_analytical = analytical_system.forward_simulate_dt(x_analytical, u_analytical, dt)


        x_learned_list.append(x_learned[0])
        x_analytical_list.append(x_analytical[0])
        xdot_learned_list.append(x_learned[1])
        xdot_analytical_list.append(x_analytical[1])

        x_error.append((x_analytical[0] - x_learned[0])**2)
        xdot_error.append((x_analytical[1] - x_learned[1])**2)

        # if i%5==0:
        #     plt.figure(1)
        #     learned_system.visualize(x_learned, u_learned)

        #     plt.figure(2)
        #     analytical_system.visualize(x_analytical, u_analytical)    

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

    plt.figure()
    plt.plot(time_steps, x_error, "-b", label='x_error')
    plt.plot(time_steps, xdot_error, "-r", label='xdot_error')
    plt.legend(loc="upper right")

    print(f'total x_error**2: {sum(x_error)}')
    print(f'total xdot_error**2: {sum(xdot_error)}')

    plt.show()