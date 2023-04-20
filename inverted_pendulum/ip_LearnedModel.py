"""
This script defines an inverted pendulum class, where the dynamics are modelled using a neural network.

State Vector x: [thetadot, theta].T

Inputs u: [torque_origin]
"""

import numpy as np
import torch
from copy import deepcopy
import matplotlib.pyplot as plt
import json,sys

sys.path.append('/home/daniel/research/catkin_ws/src/')
from rad_models.InvertedPendulum import InvertedPendulum
from hyperparam_optimization.inverted_pendulum.InvertedPendulumLightningModule import InvertedPendulumLightningModule
from hyperparam_optimization.NN_Architectures import *

class IP_LearnedModel(InvertedPendulum):
    def __init__(self, model_path=None, config=None, use_gpu=True, normalized=False):
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

        if normalized == True:
            self.xScale = self.xMax
            self.uScale = self.uMax
        else:
            self.xScale = np.ones(self.xMax.shape)
            self.uScale = np.ones(self.uMax.shape)

        if model_path is None:
            self.model =  InvertedPendulumLightning(self.config)
        else:
            self.model = InvertedPendulumLightning.load_from_checkpoint(model_path)

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

        xdot = self.model(self.input_var).cpu().detach().numpy().T

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

        if self.config['calculates_xdot']:
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
        else:
            xNorm = x/self.xScale
            uNorm = u/self.uScale
            inputs = np.vstack([xNorm,uNorm]).T
            inputs_tensor = torch.tensor(inputs).float()

            if self.use_gpu==True:
                inputs_tensor = inputs_tensor.cuda()

            x = self.model(inputs_tensor).cpu().detach().numpy().T

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
    params_path = '/home/daniel/research/catkin_ws/src/hyperparam_optimization/inverted_pendulum/run_logs/train_ip_2023-04-04_10-43-04/train_ip_1fe95f8c_13_accuracy_tolerance=0.0100,act_fn=relu,b_size=16,hdim=256,loss_fn=mse,lr=0.0003,max_epochs=500,n_hlay=0,nn_arc_2023-04-04_11-38-55/params.json'
    checkpoint_path = '/home/daniel/research/catkin_ws/src/hyperparam_optimization/inverted_pendulum/run_logs/train_ip_2023-04-04_10-43-04/train_ip_1fe95f8c_13_accuracy_tolerance=0.0100,act_fn=relu,b_size=16,hdim=256,loss_fn=mse,lr=0.0003,max_epochs=500,n_hlay=0,nn_arc_2023-04-04_11-38-55/data-points-run/fancy-totem-203/hyperparam_opt_ip/ju0gk1hp/checkpoints/epoch=498-step=1871250.ckpt'
    
    with open(params_path, 'r') as f:
        config = json.load(f)

    # trained a few without this in the config so need to include it here
    config['calculates_xdot'] = False
    json.dumps(config)

    learned_system = IP_LearnedModel(model_path=checkpoint_path, config=config)  
    x_learned = np.zeros([2,1])
    x_learned[1] = .001
    u_learned = np.zeros([1,1])    

    analytical_system = InvertedPendulum()
    x_analytical = np.zeros([2,1])
    x_analytical[1] = .001
    u_analytical = np.zeros([1,1]) 

    dt = .01
    plt.ion()
    horizon = 1000
    for i in range(0, horizon):
        x_learned = learned_system.forward_simulate_dt(x_learned, u_learned, dt)
        x_analytical = analytical_system.forward_simulate_dt(x_analytical, u_analytical, dt)
        print(x_analytical)
        if i%5==0:
            plt.figure(1)
            learned_system.visualize(x_learned, u_learned)

            plt.figure(2)
            analytical_system.visualize(x_analytical, u_analytical)    
