import numpy as np
import matplotlib.pyplot as plt
from tqdm.auto import tqdm 
import json
from PIL import Image
import time
import torch
from mpl_toolkits.mplot3d import Axes3D
import glob


import sys
sys.path.append('/home/daniel/research/catkin_ws/src/')
from nempc.src.nonlinear_empc.NonlinearEMPC import NonlinearEMPC
from BellowsGrub import BellowsGrub
from BellowsGrub_LearnedModel import BellowsGrub_LearnedModel
from hyperparam_optimization.BASE.profile import profile

"""
This file takes a learned model of the bellows grub and controls it with NEMPC
"""
@profile(output_file='/home/daniel/research/catkin_ws/src/hyperparam_optimization/bellows_grub/grub_nempc.prof', sort_by='cumulative', lines_to_print=100, strip_dirs=True)
def run_grub_nempc(controller, ground_truth, x0, u0, xgoal, sim_length, horizon, 
                   numStates=8, numInputs=4, ctrl_dt=0.01, sim_dt=0.001, ugoal=None, 
                   analytical_controller=None,
                   visualize_horizon=False, visualize_system=False, print_stats=False, plot=False):
    x_learned = x0
    u_learned = u0
    x_hist_learned = np.zeros([numStates, sim_length])
    u_hist_learned = np.zeros([numInputs, sim_length])
    solve_time_hist_learned = np.zeros([sim_length,])
    error_hist_learned = np.zeros([4, sim_length])

    if analytical_controller is not None:
        use_analytical = True
        x_ana = x0
        u_ana = u0
        x_hist_ana = np.zeros([numStates, sim_length])
        u_hist_ana = np.zeros([numInputs, sim_length])
        solve_time_hist_ana = np.zeros([sim_length,])
        error_hist_ana = np.zeros([4, sim_length])

    if visualize_system:
        fig = plt.figure("Learned Model Visualization")
        ax = fig.add_subplot(projection='3d')
        plt.ion()

    if visualize_horizon:
        plt.figure("Learned Model Horizon")
        if use_analytical:
            plt.figure("Analytical Model Horizon")
        plt.ion()
        index = list(range(sim_length+horizon))

    # Forward Simulate with Controller 
    for i in range(sim_length): 
        start = time.time()
        u_learned, path_learned = controller.solve_for_next_u(x_learned, 
                                        xgoal, 
                                        ulast=u_learned.flatten(), 
                                        ugoal=u_learned, 
                                        mutation_noise=0.63)
        end = time.time()

        error_hist_learned[:, i] = np.abs((x_learned[4:] - xgoal[4:]).flatten())
        solve_time_hist_learned[i] = end - start
        x_hist_learned[:, i] = x_learned.flatten()
        u_hist_learned[:, i] = u_learned.flatten()

        if use_analytical:
            start = time.time()
            u_ana, path_ana = analytical_controller.solve_for_next_u(x_ana, 
                                            xgoal, 
                                            ulast=u_ana.flatten(), 
                                            ugoal=u_ana, 
                                            mutation_noise=0.63)
            end = time.time()

            error_hist_ana[:, i] = np.abs((x_ana[4:] - xgoal[4:]).flatten())
            solve_time_hist_ana[i] = end - start
            x_hist_ana[:, i] = x_ana.flatten()
            u_hist_ana[:, i] = u_ana.flatten()

        # Forward simulate the system
        for j in range(int(ctrl_dt/sim_dt)):
            x_learned = ground_truth.forward_simulate_dt(x_learned, u_learned, sim_dt)
            if use_analytical:
                x_ana = ground_truth.forward_simulate_dt(x_ana, u_ana, sim_dt)

        if visualize_horizon:
            plt.figure("Learned Model Horizon")
            plt.clf()
            plt.axhline(y=xgoal[6], color='y', linestyle='-')
            plt.axhline(y=xgoal[7], color='k', linestyle='-')
            plt.plot(index[i:i+horizon], path_learned[:,6],':r', index[i:i+horizon], path_learned[:,7],':b')
            plt.plot(index[:i], x_hist_learned[6,:i],'-m', index[:i], x_hist_learned[7,:i],'-c')
            plt.legend(['Goal U', 'Goal V', 'Planned U', 'Planned V', 'Sim U','Sim V'])
            plt.grid(True)

            if use_analytical:
                plt.figure("Analytical Model Horizon")
                plt.clf()
                plt.axhline(y=xgoal[6], color='y', linestyle='-')
                plt.axhline(y=xgoal[7], color='k', linestyle='-')
                plt.plot(index[i:i+horizon], path_ana[:,6],':r', index[i:i+horizon], path_ana[:,7],':b')
                plt.plot(index[:i], x_hist_ana[6,:i],'-m', index[:i], x_hist_ana[7,:i],'-c')
                plt.legend(['Goal U', 'Goal V', 'Planned U', 'Planned V', 'Sim U','Sim V'])
                plt.grid(True)

            plt.show()
            plt.pause(0.05)

        if visualize_system:
            ground_truth.visualize(x_learned,ax)

    iae_metric_learned = np.sum(error_hist_learned)
    avg_resting_pos_learned = np.mean(x_hist_learned[6:, -10:], 1)
    control_successful_learned = np.all(np.abs(avg_resting_pos_learned - xgoal[6:].T) < 0.1)

    if use_analytical:
        iae_metric_ana = np.sum(error_hist_ana)
        avg_resting_pos_ana = np.mean(x_hist_ana[6:, -10:], 1)
        control_successful_ana = np.all(np.abs(avg_resting_pos_ana - xgoal[6:].T) < 0.1)

    if print_stats:
        print("==============================  Learned Model Stats ===================================================")
        print(f'Average solve time (not including first 3): {np.mean(solve_time_hist_learned)}')
        print(f'q_goal is u={xgoal[6]} and v={xgoal[7]} \
            \n  q_final is u={x_learned[6]} and v={x_learned[7]}  \
            \nControl {"was Successful" if control_successful_learned else "Failed"} with IAE={iae_metric_learned}\n')
        
        if analytical_controller is not None:
            print("==============================  Analytical Model Stats ===================================================")
            print(f'Average solve time (not including first 3): {np.mean(solve_time_hist_ana)}')
            print(f'q_goal is u={xgoal[6]} and v={xgoal[7]} \
                \n  q_final is u={x_ana[6]} and v={x_ana[7]}  \
                \nControl {"was Successful" if control_successful_ana else "Failed"} with IAE={iae_metric_ana}\n')
            
    if plot:
        ender = -1
        plt.figure("Learned Model Joint Velocities")
        plt.plot(x_hist_learned[4,:ender],':r', x_hist_learned[5,:ender],':b')
        plt.xlabel("Timestep")
        plt.ylabel("Joint Vel (rad/s)")
        plt.legend(['Sim U','Sim V'])
        plt.grid(True)

        plt.figure("Learned Model Joint Positions")
        plt.axhline(y=xgoal[6], color='g', linestyle='-')
        plt.axhline(y=xgoal[7], color='k', linestyle='-')
        plt.plot(x_hist_learned[6,:ender],':r', x_hist_learned[7,:ender],':b')
        plt.xlabel("Timestep")
        plt.ylabel("Joint position (rad)")
        plt.legend(['Goal U', 'Goal V', 'Sim U','Sim V'])
        plt.grid(True)

        plt.figure("Learned Model Joint Pressures")
        plt.plot(x_hist_learned[0,:ender],'r--', x_hist_learned[1,:ender],'b--', x_hist_learned[2,:ender],'g--', x_hist_learned[3,:ender],'k--')
        plt.xlabel("Timestep")
        plt.ylabel("Joint Pressures (KPa)")
        plt.legend(['p0','p1','p2','p3'])
        plt.grid(True)

        plt.figure("Learned Model Solve Times")
        plt.plot(solve_time_hist_learned[3:].T)        
        plt.ylabel("Timestep")
        plt.grid(True)

        plt.figure("Learned Model Error")
        plt.plot(error_hist_learned.T)  
        plt.xlabel("Timestep")
        plt.ylabel("Error (rad)")
        plt.legend(['Error Udot', 'Error Vdot', 'Error U','Error V'])
        plt.grid(True)

        if use_analytical:
            plt.figure("Analytical Model Joint Velocities")
            plt.plot(x_hist_ana[4,:ender],':r', x_hist_ana[5,:ender],':b')
            plt.xlabel("Timestep")
            plt.ylabel("Joint Vel (rad/s)")
            plt.legend(['Sim U','Sim V'])
            plt.grid(True)

            plt.figure("Analytical Model Joint Positions")
            plt.axhline(y=xgoal[6], color='g', linestyle='-')
            plt.axhline(y=xgoal[7], color='k', linestyle='-')
            plt.plot(x_hist_ana[6,:ender],':r', x_hist_ana[7,:ender],':b')
            plt.xlabel("Timestep")
            plt.ylabel("Joint position (rad)")
            plt.legend(['Goal U', 'Goal V', 'Sim U','Sim V'])
            plt.grid(True)

            plt.figure("Analytical Model Joint Pressures")
            plt.plot(x_hist_ana[0,:ender],'r--', x_hist_ana[1,:ender],'b--', x_hist_ana[2,:ender],'g--', x_hist_ana[3,:ender],'k--')
            plt.xlabel("Timestep")
            plt.ylabel("Joint Pressures (KPa)")
            plt.legend(['p0','p1','p2','p3'])
            plt.grid(True)

            plt.figure("Analytical Model Solve Times")
            plt.plot(solve_time_hist_ana[3:].T)        
            plt.ylabel("Timestep")
            plt.grid(True)

            plt.figure("Analytical Model Error")
            plt.plot(error_hist_ana.T)  
            plt.xlabel("Timestep")
            plt.ylabel("Error (rad)")
            plt.legend(['Error Udot', 'Error Vdot', 'Error U','Error V'])
            plt.grid(True)

        plt.ioff()
        plt.show()
    elif visualize_horizon:
        plt.ioff()
        plt.show()

    return x_learned, iae_metric_learned, control_successful_learned

def nempc_setup(horizon, sim_seconds, ctrl_dt, trial_dir=None, random_ic=True):
    # Initial Conditions
    p0 = np.array([[200],
                    [200],
                    [200],
                    [200]])
    qd0 = np.zeros((2,1))
    if random_ic:
        q0 = np.random.uniform(low=-np.pi/2, high=np.pi/2, size=(2, 1))
    else:
        q0 = np.array([[0.1], [0.1]])
    x0 = np.vstack([p0,qd0,q0])
    u0 = 200*np.ones([4, 1])

    xgoal = np.zeros((8,1))
    q_goal = np.random.uniform(low=-np.pi/2, high=np.pi/2, size=(2,1)) # random joint angle goal
    xgoal[6:] = q_goal
    ugoal = np.zeros([4, 1]) # right now in the cost function this is weighted at 0

    Q = 1.0*np.diag([0,0,0,0,
                0.21, 0.21,
                50.0, 50.0])  

    sim_length = int(sim_seconds/ctrl_dt)

    def CostFunc(x, u, xgoal, ugoal, prev_u=None, final_timestep=False):
        if use_gpu:
            Qx = torch.mm(Q,(x-xgoal)**2.0)
            cost = torch.sum(Qx,axis=0)
        else:
            Qx = Q.dot(x-xgoal)**2.0
            cost = np.sum(Qx, axis=0)
        return cost
    
    # NEMPC Controller Parameters
    if trial_dir is None: # Use the analytical model
        system = BellowsGrub()
        use_gpu = False 
        numSims = 100
        numParents = 30
        numStrangers = 10
    else: # use the learned model
        # The learned model is used to predict the best inputs and the analytical model is used as ground truth to apply the predicted inputs
        checkpoint_path = glob.glob(trial_dir + '/lightning_logs/version_0/checkpoints/*.ckpt')[0]
        # get the config file
        with open(trial_dir + '/params.json', 'r') as f:
            config = json.load(f)
        use_gpu = True
        system = BellowsGrub_LearnedModel(model_path=checkpoint_path, config=config, use_gpu=use_gpu)
        numSims = 1000
        numParents = 300
        numStrangers = 100
        Q = torch.from_numpy(Q).float().cuda()

    forward = system.forward_simulate_dt 
    numStates = system.numStates
    numInputs = system.numInputs
    umin = np.ones((numInputs, 1))*system.uMin
    umax = np.ones((numInputs, 1))*system.uMax
    ground_truth = BellowsGrub()

    controller = NonlinearEMPC(
                                forward,
                                CostFunc, 
                                numStates=numStates, 
                                numInputs=numInputs, 
                                umin=umin, 
                                umax=umax, 
                                horizon=int(horizon/ctrl_dt), 
                                dt=ctrl_dt, 
                                numSims=numSims, 
                                numKnotPoints=4, 
                                useGPU=use_gpu, 
                                numParents=numParents, 
                                numStrangers=numStrangers,
                                display=False)
    
    return controller, ground_truth, x0, u0, xgoal, sim_length

if __name__=="__main__":
    trial_dir = '/home/daniel/research/catkin_ws/src/hyperparam_optimization/bellows_grub/run_logs/fnn_random_masdata/train_11940ae4_4_accuracy_tolerance=0.0100,act_fn=sigmoid,b_size=128,cpu_num=7,dataset_size=120000,dt=0.0100,generate_new_data=Fal_2023-07-07_22-42-54'
    
    ctrl_dt = 0.01
    horizon = 0.5
    horizon_steps = int(horizon / ctrl_dt)
    sim_seconds = 1.0
    
    controller, ground_truth, x0, u0, xgoal, sim_length = nempc_setup(horizon, sim_seconds, ctrl_dt, trial_dir=trial_dir)
    analytical_controller, _, _, _, _, _ = nempc_setup(horizon, sim_seconds, ctrl_dt, trial_dir=None)

    x, iae_metric, control_successful = run_grub_nempc(controller, ground_truth, x0, u0, xgoal, sim_length, horizon_steps, ctrl_dt=ctrl_dt,
                                                       analytical_controller=analytical_controller,
                                                       visualize_system=False, visualize_horizon=True, print_stats=True, plot=False)