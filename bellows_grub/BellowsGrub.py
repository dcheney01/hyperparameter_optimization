import sys
sys.path.append('/home/daniel/research/catkin_ws/src/')
from hyperparam_optimization.BASE.Model import Model

import bellows_grub_dynamics as dyn

import numpy as np
from copy import deepcopy
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

np.set_printoptions(precision=2)


class BellowsGrub(Model):

    def __init__(self, mass=2.0, stiffness=36.0, damping=2.25, pressure_resp_coeff=20.0):
        super().__init__()

        self.name = 'grub'
        self.numStates = 8
        self.numInputs = 4

        self.stiffness = stiffness
        self.damping = damping
        self.alpha = pressure_resp_coeff

        self.K_passive_damper = np.diag([damping, damping])
        self.K_passive_spring = np.diag([stiffness, stiffness])
        self.prs_to_torque = np.array([[.15, -.15, 0, 0],
                                       [0, 0, .15, -.15]])*1.0
        # reference pressure limits in KPa
        self.uMax = 400.0
        self.uMin = 8.0

        # state limits, KPa, rad/s, and rad
        self.xMax = np.array([[500.0,
                               500.0,
                               500.0,
                               500.0,
                               100.0,
                               100.0,
                               np.pi/2,
                               np.pi/2]]).T

        self.xMin = np.array([[.0,
                               .0,
                               .0,
                               .0,
                               -100.0,
                               -100.0,
                               -np.pi/2,
                               -np.pi/2]]).T

        self.h = .21
        self.m = mass  # 1.923 measured
        self.r = .16

        # remember this is for pdot = alphas*(pref-p)
        self.alphas = np.eye(4)*pressure_resp_coeff

        self.y_index = np.array([0, 1, 2, 3, 6, 7])
        self.y_prev = np.zeros((len(self.y_index), 1))
        self.ydot_prev = np.zeros(np.shape(self.y_prev))

    def calc_grub_regressor(self, q, qd, qdr, qddr):
        q[np.abs(q) < .001] = np.sign(q[abs(q) < .001])*.001
        q[q == 0] = .001
        regressor = dyn.calc_regressor(
            q, qd, qdr, qddr, self.h, self.m, self.r, -9.81).reshape(-1, 2).T
        return regressor

    def calc_continuous_A_B_w(self, x):
        p = x[0:4]
        qd = x[4:6]
        q = x[6:8]
        # small numbers to .001 with same sign, i.e. -.0001 to -.001
        q[np.abs(q) < .001] = np.sign(q[abs(q) < .001])*.001
        # take care of any remaining zeros
        q[q == 0] = .001
        M = dyn.calc_M(q, self.h, self.m, self.r).reshape(2, 2)
        if np.linalg.cond(M) > 10000:
            print("Near singular Mass matrix...")
            print("q: ", q)
            print("M: ", M)
        M_inv = np.linalg.inv(M)

        C = dyn.calc_C(q, qd, self.h, self.m, self.r).reshape(2, 2)

        A11 = -self.alphas
        A12 = np.zeros([4, 2])
        A13 = np.zeros([4, 2])
        A21 = np.matmul(M_inv, self.prs_to_torque)
        # defined positively above
        A22 = np.matmul(M_inv, -C-self.K_passive_damper)
        A23 = np.matmul(M_inv, -self.K_passive_spring)
        A31 = np.zeros([2, 4])
        A32 = np.eye(2)
        A33 = np.zeros([2, 2])

        A = np.vstack([np.hstack([A11, A12, A13]), np.hstack(
            [A21, A22, A23]), np.hstack([A31, A32, A33])])

        B11 = self.alphas
        B21 = np.zeros([2, 4])
        B31 = np.zeros([2, 4])
        B = np.vstack([B11, B21, B31])

        gravity_torque = dyn.calc_grav(q, self.h, self.m, 9.81)

        w = np.matmul(M_inv, -gravity_torque)
        w = np.vstack(
            [np.zeros((4, 1)), np.reshape(w, (2, 1)), np.zeros((2, 1))])

        return A, B, w

    def calc_discrete_A_B_w(self, x, dt=.01):

        [A, B, w] = self.calc_continuous_A_B_w(x)

        # Matrix exponentiation
        [Ad, Bd] = self.discretize_A_and_B(A, B, dt)

        wd = np.linalg.inv(A).dot(Ad-np.eye(Ad.shape[0])).dot(w)
        # wd = np.matmul(Ad,w)*dt

        return Ad, Bd, wd

    def calc_state_derivs(self, x, u):
        """ Takes in x (shape [numStates, batch]) and u (shape [numInputs, batch])
        Returns x_dot in the same shape"""

        xDot = dyn.calc_state_derivs(x,
                                     u,
                                     self.m,
                                     self.stiffness,
                                     self.damping,
                                     self.alpha)

        return xDot

    def test_wrapper(self, t, x, *args):
        u = np.array(args)
        xdot = self.calc_state_derivs(x, u)
        return xdot.flatten()

    def visualize(self, x, ax):
        ax.clear()
        q = x[6:8]
        num_segments = 20
        last_pos = np.zeros([3, 1])
        for i in range(0, num_segments):
            scalar = float(i+1)/num_segments
            this_pos = np.asarray(dyn.fkEnd(q*scalar, self.h*scalar,
                                 self.h*scalar)).reshape(4, 4)[0:3, 3]
            self.plot_line(ax, last_pos, this_pos, 'orange', self.r*50)
            last_pos = this_pos

        self.set_axes_equal(ax)
        plt.ion()
        plt.show()
        plt.pause(.000001)

    def plot_line(self, ax, a, b, color, linewidth_custom, alpha=1.0):
        # Ax must be projection 3d
        a = a.flatten()
        ax.plot([a[0], b[0]], [a[1], b[1]], [a[2], b[2]],
                color=color, linewidth=linewidth_custom, alpha=alpha)

    def set_axes_equal(self, ax):

        ax.set_xlim3d([-self.h, self.h])
        ax.set_ylim3d([-self.h, self.h])
        ax.set_zlim3d([0, self.h])


if __name__ == '__main__':
    sys = BellowsGrub()

    dt = .02
    sim_time = 1.0 # seconds

    horizon = int(sim_time/dt)
    tspan = np.arange(0, sim_time, dt)
    x_hist = np.zeros((sys.numStates, horizon))

    q = np.ones((2, 1))*0 # u, v
    qd = np.zeros((2, 1))   # u_dot, v_dot
    p = np.array([[200],
                  [200],
                  [200],
                  [200]]) # current pressures

    x = np.vstack([p, qd, q]) # state
    x0 = deepcopy(x)

    y = x0[sys.y_index]
    ydot = np.zeros(np.shape(y))

    u = np.zeros((4,horizon)) # inputs
   
    fig = plt.figure()
    ax = fig.add_subplot(projection='3d')
    plt.ion()
    sys.visualize(x, ax)
    p_ref = np.ones((4,)) * 200
    step_dt = 0.001

    # p_ref = np.random.uniform(low=sys.uMin, high=sys.uMax, size=(4,))


    # SIMULATION BLOCK ============================================================
    for i in range(0, horizon):
        x_hist[:, i] = deepcopy(x.flatten())

        # if i % 20 == 0: # change p_ref 
        #     print(f' For p_ref={p_ref}')
        #     print(f'u: {x[6]}, v: {x[7]}')
        #     p_ref = np.random.uniform(low=sys.uMin, high=sys.uMax, size=(4,))
        #     # p_ref = u[:, i % 10]
        #     input("Press Enter to forward_simulate_dt 20 timesteps")

        # simulate 1 step
        u[:, i] = p_ref
        for j in range(0, int(dt/step_dt)):
            x = sys.forward_simulate_dt(x, p_ref, step_dt, method='RK4')

        print(f'u: {x[6]}, v: {x[7]}')
        sys.visualize(x, ax)
    # END SIMULATION BLOCK =========================================================
    plt.ioff()
    fig, axs = plt.subplots(2, 2)
    fig.suptitle("Simulation Results with dt={}".format(dt))
    axs[0, 1].plot(x_hist[:4, :].T)
    axs[0, 1].set_xlabel('Time (s)')
    axs[0, 1].set_ylabel('Pressure (KPa)')
    axs[0, 1].set_title('Pressure')
    axs[0, 1].legend(['p0', 'p1', 'p2', 'p3'])
    axs[0, 1].grid('True', alpha=.3)

    axs[1, 0].plot(tspan, x_hist[4:6, :].T)
    axs[1, 0].set_xlabel('Time (s)')
    axs[1, 0].set_ylabel('Velocity (rad/s)')
    axs[1, 0].set_title('Velocity')
    axs[1, 0].legend(['udot', 'vdot'])
    axs[1, 0].grid('True', alpha=.3)

    axs[1, 1].plot(tspan, x_hist[-2:, :].T)
    axs[1, 1].set_xlabel('Time (s)')
    axs[1, 1].set_ylabel('Theta (rad)')
    axs[1, 1].set_title('Position')
    axs[1, 1].legend(['u', 'v'])
    axs[1, 1].grid('True', alpha=.3)

    axs[0, 0].plot(tspan, u.T)
    axs[0, 0].set_xlabel('Time (s)')
    axs[0, 0].set_ylabel('Commanded Pressure (KPa)')
    axs[0, 0].set_title('Input')
    axs[0, 0].legend(['u0', 'u1', 'u2', 'u3'])
    axs[0, 0].grid('True', alpha=.3)

    plt.show()