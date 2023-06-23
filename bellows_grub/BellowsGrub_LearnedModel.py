from grub_optimize import GrubLightningModule
from hyperparam_optimization.bellows_grub.BellowsGrub import BellowsGrub
import bellows_grub_dynamics as dyn # this is the fixed version

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import json
import torch

np.set_printoptions(precision=2)

class BellowsGrub_LearnedModel(BellowsGrub):
    def __init__(self, model_path=None, config=None, use_gpu=True):
        super().__init__()
        """
        Defined in BellowsGrubIntegral.py:
            self.numStates = 8
            self.numInputs = 4
            self.uMax = 400.0
            self.uMin = 8.0

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
        """

        # in terms of the network
        # self.num_model_outputs = 8
        # self.num_model_inputs = 12

        self.use_gpu = use_gpu
        self.config = config

        if config['normalized_data']:
            self.xScale = self.xMax
            self.uScale = self.uMax
        else:
            self.xScale = np.ones(self.xMax.shape)
            self.uScale = np.ones(self.numInputs)

        if model_path is None:
            self.model = GrubLightningModule(self.config)
        else:
            self.model = GrubLightningModule.load_from_checkpoint(model_path, config=self.config)

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

        return xdot
    
    def forward_simulate_dt(self, x, u, dt):
        """
        Returns x at the next time step.
        Integration options: euler, RK4.

        Can also just return the output of the model if the model learned the next state instead of change in state

        This is the basic function to be used for simulation.

        If angle wrapping is desired, wrapAngle can be set to boolean numpy array with True values in the positions of angles to be wrapped. wrapRange controls the interval at which to wrap.
        """
        xcols = np.shape(x)[1]

        x = np.clip(x.cpu(), self.xMin, self.xMax)
        u = np.clip(u.cpu(), self.uMin, self.uMax)

        if self.config['learn_mode'] == 'x_dot':
            x_dot = self.calc_state_derivs(x, u)
            x = x + x_dot*dt
        elif self.config['learn_mode'] == 'x':
            x = self.calc_state_derivs(x, u)
        else:
            raise ValueError(f"Invalid learn_mode. Must be x or x_dot. Got {self.config['learn_mode']}")

        # make sure output is expected size
        assert np.shape(x) == (self.numStates, xcols),\
            "Something went wrong, output vector is not correct size:"\
            "\nExpected size:({},n)\nActual size:{}"\
            .format(self.numStates, np.shape(x))

        return x
    

if __name__=='__main__':
    trial_dir = '/home/daniel/research/catkin_ws/src/hyperparam_optimization/bellows_grub/run_logs/fnn_try4_moredata/train_bf2ff742_8_accuracy_tolerance=0.0500,act_fn=tanh,b_size=32,cpu_num=7,dataset_size=120000,dt=0.0100,generate_new_data=False,g_2023-06-20_22-49-18'

    # get the checkpoint
    import glob
    checkpoint_path = glob.glob(trial_dir + '/lightning_logs/version_0/checkpoints/*.ckpt')[0]

    # get the config file
    with open(trial_dir + '/params.json', 'r') as f:
        config = json.load(f)

    print(json.dumps(config, indent=4))

    learned_system = BellowsGrub_LearnedModel(model_path=checkpoint_path, config=config)  
    q_learned = np.ones((2,1))*.01
    qd_learned = np.zeros((2,1))
    p_learned = np.array([[200],
                  [200],
                  [200],
                  [200]])
    x_learned = np.vstack([p_learned,qd_learned,q_learned])


    analytical_system = BellowsGrub()
    q_analytical = np.ones((2,1))*.01
    qd_analytical = np.zeros((2,1))
    p_analytical = np.array([[200],
                  [200],
                  [200],
                  [200]])
    x_analytical = np.vstack([p_analytical,qd_analytical,q_analytical])

    p_ref = np.random.uniform(low=analytical_system.uMin, high=analytical_system.uMax, size=(4, 1))

    dt = config['dt']
    sim_time = 5
    horizon = int(sim_time/dt)

    u_learned_list = []
    u_analytical_list = []
    v_learned_list = []
    v_analytical_list = []
    
    fig_learned = plt.figure()
    fig_analytical = plt.figure()

    ax_learned = fig_learned.add_subplot(projection='3d')
    ax_analytical = fig_analytical.add_subplot(projection='3d')

    plt.ion()
    
    for i in range(0, horizon):

        # if I want to do random commands every n timesteps
        # if i%10 == 0:
        #     p_ref = np.random.uniform(low=sys.uMin, high=sys.uMax, size=(4,1))

        x_learned = learned_system.forward_simulate_dt(x_learned, p_ref, dt)
        x_analytical = analytical_system.forward_simulate_dt(x_analytical, p_ref, dt)

        u_learned_list.append(x_learned[6])
        u_analytical_list.append(x_analytical[6])
        v_learned_list.append(x_learned[7])
        v_analytical_list.append(x_analytical[7])

        if i%5==0:
            learned_system.visualize(x_learned, ax_learned)

            analytical_system.visualize(x_analytical, ax_analytical)    

    time_steps = [i for i in range(horizon)]

    plt.ioff()
    plt.figure()
    plt.plot(time_steps, u_learned_list, "-b", label='u_learned')
    plt.plot(time_steps, u_analytical_list, "-r", label='u_analytical')
    plt.legend(loc="upper right")

    plt.figure()
    plt.plot(time_steps, v_learned_list, "-b", label='v_learned')
    plt.plot(time_steps, v_analytical_list, "-r", label='v_analytical')
    plt.legend(loc="upper right")

    plt.show()
  
