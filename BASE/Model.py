import numpy as np
from copy import deepcopy
from scipy.linalg import expm

import sys
sys.path.append('/home/daniel/research/catkin_ws/src/')


class Model():
    def __init__(self):
        self.name = None
        self.numStates = None
        self.numInputs = None
        self.uMax = None
        self.uMin = None
        self.xMax = None
        self.xMin = None
        self.y_prev = None
        self.ydot_prev = None
        self.wrapAngle = np.array(False)
        self.wrapRange = (-np.pi, np.pi)

    def forward_simulate_dt(self, x, u, dt, method='RK4'):
        """
        Returns x at the next time step.
        Integration options: euler, RK4.

        This is the basic function to be used for simulation.

        If angle wrapping is desired, wrapAngle can be set to boolean numpy array with True values in the positions of angles to be wrapped. wrapRange controls the interval at which to wrap.
        """
        # make sure inputs are the correct sizes
        x = deepcopy(x)
        xcols = np.shape(x)[1]

        u = deepcopy(u)

        # # enforce state constraints
        # too_high = np.where(x > self.xMax)
        # x[too_high] = self.xMax[too_high]
        # too_low = np.where(x < self.xMin)
        # x[too_low] = self.xMin[too_low]

        x = np.clip(x, self.xMin, self.xMax)
        # # enforce input constraints
        u = np.clip(u,self.uMin,self.uMax)


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

        # make sure output is expected size
        assert np.shape(x) == (self.numStates, xcols),\
            "Something went wrong, output vector is not correct size:"\
            "\nExpected size:({},n)\nActual size:{}"\
            .format(self.numStates, np.shape(x))

        return x

    def calc_state_derivs(self, x, u):
        print("calc_state_derivs function not yet implemented. Please add one!")
        raise NotImplementedError

    def visualize(x):
        print("visualize function not yet implemented. Please add one!")
        raise NotImplementedError

    def calc_A_B_w(self, x, u, delta=1.0e-6, method='central'):
        """Calculates a numerical A, B, and w for the system defined by 'calc_state_derivs'. Default method is a central difference approximation. Can be overwritten if an analytical solution is desired. This is necessary for some implementations of MPC, but otherwise not necessary for simulation."""
        x = deepcopy(x)
        u = deepcopy(u)
        A = np.zeros([self.numStates, self.numStates])
        B = np.zeros([self.numStates, self.numInputs])

        xdot0 = self.calc_state_derivs(x, u)
        xup = deepcopy(x)
        uup = deepcopy(u)
        xdown = deepcopy(x)
        udown = deepcopy(u)

        for i in range(0, self.numStates):
            xup[i] = x[i] + delta
            xdown[i] = x[i] - delta
            xdotup = self.calc_state_derivs(xup, u).flatten()
            xdotdown = self.calc_state_derivs(xdown, u).flatten()
            A[:, i] = (xdotup-xdotdown)/(2.0*delta)
            xup[i] = x[i]
            xdown[i] = x[i]

        for i in range(0, self.numInputs):
            uup[i] = u[i] + delta
            udown[i] = u[i] - delta
            xdotup = self.calc_state_derivs(x, uup).flatten()
            xdotdown = self.calc_state_derivs(x, udown).flatten()
            B[:, i] = (xdotup-xdotdown)/(2.0*delta)
            uup[i] = u[i]
            udown[i] = u[i]

        w = xdot0.reshape(-self.numStates,1) - A.dot(x).reshape(-self.numStates,1) - B.dot(u).reshape(-self.numStates,1)

        return A, B, w

    def calc_A_B(self, x, u, dt, delta=1.0e-6):
        """Old method - use "calc_A_B_w" instead"""
        A, B, _ = self.calc_A_B_w(x, u, delta=delta)
        return A, B

    def discretize_A_and_B(self, A, B, dt):
        """Discretizes a given A and B matrix. Attempts matrix exponentiation, but will perform Euler integration if exponentiation fails."""
        try:
            Ad = expm(A*dt)
            Bd = np.matmul(np.linalg.inv(A), np.matmul(
                Ad-np.eye(Ad.shape[0]), B))
        except:
            # print("Failed to do matrix exponentiation for discretization...")
            Ad = np.eye(A.shape[0]) + A*dt
            Bd = B*dt
        return Ad, Bd

    def calc_discrete_A_B_w(self, x, u, dt):
        """Calculates a dicrete A, B and w for a given state and input."""
        x = deepcopy(x)
        u = deepcopy(u)
        x = x.reshape(self.numStates, -1)
        A, B, w = self.calc_A_B_w(x, u)

        [Ad, Bd] = self.discretize_A_and_B(A, B, dt)
        wd = w*dt

        return Ad, Bd, wd

    def measure(self, x):
        """Calculates outputs from system as if there were sensors

        Args:
           self arg1
           x array (numStates, 1), state vector

        Returns:
            Measured states, y = Cx

        Example:
        self.y_index = np.array([2,3])
        x = np.array([[1,2,3,4]]).T

        y = measure(x) = array([[3,4]]).T

        """

        x = x.reshape(self.numStates, -1)

        y = x[self.y_index]

        assert y.shape[0] == len(self.y_index)

        return y

    def differentiate_y(self, y, sigma, dt):
        """Numerically differentiates measured output y

        Args:
           self arg1
           y array (numOutputs,1), measured output vector
           sigma float, dirty derivative gain
           dt float, time step

        Returns:
            None, sets self.ydot

        Notes:
        Dirty derivative filters out noise w/ freq > 1/sigma.

        """
        beta = (2*sigma - dt)/(2*sigma + dt)

        ydot = beta*self.ydot_prev \
            + (1-beta)/dt * (y - self.y_prev)

        return ydot