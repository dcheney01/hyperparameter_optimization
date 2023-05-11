"""
This script defines an inverted pendulum class.

State Vector x: [thetadot, theta].T

Inputs u: [torque_origin]
"""
import sys
sys.path.append('/home/daniel/research/catkin_ws/src/')
from hyperparam_optimization.BASE.Model import Model

import matplotlib.pyplot as plt
import numpy as np
from copy import deepcopy

class InvertedPendulum(Model):

    def __init__(self, length=1.0, mass=0.2, damping=0.1, gravity=9.81, uMax=1.0):
        super().__init__()
        self.name = "Inverted_Pendulum"
        self.numStates = 2
        self.numInputs = 1

        self.uMax = uMax  # .reshape([self.numInputs,1])
        self.uMin = -uMax  # .reshape([self.numInputs,1])

        self.xMax = np.array([10.0*np.pi, 2.0*np.pi]
                             ).reshape([self.numStates, 1])
        self.xMin = -self.xMax.reshape([self.numStates, 1])
        self.wrapAngle = np.array([False, True])
        self.wrapRange = (-np.pi, np.pi)
        self.y_index = [1]
        self.ydot_index = [0]
        self.ydot_prev = 0.
        self.y_prev = 0.

        self.m = mass
        self.l = length
        self.b = damping
        self.g = gravity
        self.I = self.m*self.l**2.0

    def calc_state_derivs(self, x, u):
        x = deepcopy(x)
        u = deepcopy(u)
        x = x.reshape([self.numStates, -1])
        xdot = np.zeros(x.shape)
        xdot[0, :] = (-self.b*x[0, :] + self.m *
                      self.g*np.sin(x[1, :]) + u)/self.I
        xdot[1, :] = x[0, :]
        return xdot

    def calc_continuous_A_B_w(self, x, u, dt=.01):
        x = x.reshape(self.numStates, -1)
        A = np.matrix([[-self.b/self.I, 0],
                       [1.0, 0]])
        B = np.matrix([[1.0/self.I],
                       [0.0]])
        w = np.matrix([[self.m*self.g*np.sin(x[1, 0])/self.I],
                       [0.0]])
        return A, B, w

    def calc_discrete_A_B_w(self, x, u, dt=.01):
        x = deepcopy(x)
        u = deepcopy(u)
        x = x.reshape(self.numStates, -1)
        A = np.matrix([[-self.b/self.I, 0],
                       [1.0, 0]])
        B = np.matrix([[1.0/self.I],
                       [0.0]])
        w = np.matrix([[self.m*self.g*np.sin(x[1, 0])/self.I],
                       [0.0]])

        [Ad, Bd] = self.discretize_A_and_B(A, B, dt)
        wd = w*dt

        return Ad, Bd, wd

    def visualize(self, x, u=np.zeros(1)):
        x = x.reshape(-1, 1)
        u = u.reshape(-1, 1)
        CoM = [-0.5*np.sin(x[1, 0]), 0.5*np.cos(x[1, 0])]
        theta = x[1, 0]

        x = [CoM[0] + self.l/2.0 *
             np.sin(theta), CoM[0] - self.l/2.0*np.sin(theta)]
        y = [CoM[1] - self.l/2.0 *
             np.cos(theta), CoM[1] + self.l/2.0*np.cos(theta)]

        massX = CoM[0] - self.l/2.0*np.sin(theta)
        massY = CoM[1] + self.l/2.0*np.cos(theta)

        plt.clf()
        ax = plt.gca()
        ax.set_aspect('equal')
        plt.plot(x, y)
        plt.scatter(massX, massY, 50, 'r')
        plt.axis([-1.5, 1.5, -1.5, 1.5])
        plt.ion()
        plt.show()
        plt.pause(.0000001)

    def estimate_x(self, x, dt):

        y = self.measure(x) # estimated position
        ydot = self.differentiate_y(y, .2, dt) # estimated qdot/velocity

        self.y_prev = y
        self.ydot_prev = ydot  

        x_hat = np.array([ydot, y]).reshape([self.numStates, 1]) # estiamted state vector
        return x_hat


if __name__ == '__main__':

    sys = InvertedPendulum()
    x = np.zeros([sys.numStates, 1])
    x[1] = .001
    u = np.zeros([sys.numInputs, 1])

    dt = .01

    fig = plt.gca()
    plt.ion()

    horizon = 2000
    for i in range(0, horizon):

        x = sys.forward_simulate_dt(x, u, dt)
        if i % 5 == 0:
            sys.visualize(x, u)
