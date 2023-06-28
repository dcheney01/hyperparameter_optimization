import numpy as np
import matplotlib.pyplot as plt
from tqdm.auto import tqdm 
import json
from PIL import Image
import time
import torch
from mpl_toolkits.mplot3d import Axes3D

import sys
sys.path.append('/home/daniel/research/catkin_ws/src/')
from nempc.src.nonlinear_empc.NonlinearEMPC import NonlinearEMPC
from BellowsGrub import BellowsGrub
from BellowsGrub_LearnedModel import BellowsGrub_LearnedModel
from hyperparam_optimization.BASE.profile import profile
# from rad_models.BellowsGrubIntegral import BellowsGrub
"""
This file takes a learned model of the bellows grub and controls it with NEMPC
"""
@profile(output_file='/home/daniel/research/catkin_ws/src/hyperparam_optimization/bellows_grub/grub_nempc.prof', sort_by='cumulative', lines_to_print=100, strip_dirs=True)
def grub_nempc(checkpoint_path, config, x0, u0, xgoal, ugoal, visualize=False, plot=False, makeGif=False):
    Q = 1.0*np.diag([0,0,0,0,
                0.21, 0.21,
                50.0, 50.0])      
    # NEMPC Controller Parameters
    if checkpoint_path is None: # Use the analytical model
        system = BellowsGrub()
        dt = 0.01
        use_gpu = False 
    else: # use the learned model
        # The learned model is used to predict the best inputs and the analytical model is used as ground truth to apply the predicted inputs
        use_gpu = True
        system = BellowsGrub_LearnedModel(model_path=checkpoint_path, config=config, use_gpu=use_gpu)
        dt = config['dt']
        Q = torch.from_numpy(Q).float().cuda()

    forward = system.forward_simulate_dt 
    numStates = system.numStates
    numInputs = system.numInputs
    umin = np.ones((numInputs, 1))*system.uMin
    umax = np.ones((numInputs, 1))*system.uMax
    ground_truth = BellowsGrub()

    x = x0
    u = u0

    horizon = 50
    numSims = 100
    numParents = 30
    numStrangers = 10
    sim_seconds = 1.5
    sim_length = int(sim_seconds/dt)

    def CostFunc(x, u, xgoal, ugoal, prev_u=None, final_timestep=False):
        if use_gpu:
            Qx = torch.mm(Q,(x-xgoal)**2.0)
            cost = torch.sum(Qx,axis=0)
        else:
            Qx = Q.dot(x-xgoal)**2.0
            cost = np.sum(Qx, axis=0)
        return cost
    
    controller = NonlinearEMPC(
                                forward,
                                CostFunc, 
                                numStates=numStates, 
                                numInputs=numInputs, 
                                umin=umin, 
                                umax=umax, 
                                horizon=horizon, 
                                dt=dt, 
                                numSims=numSims, 
                                numKnotPoints=4, 
                                useGPU=use_gpu, 
                                numParents=numParents, 
                                numStrangers=numStrangers,
                                display=plot)

    x_hist = np.zeros([numStates, sim_length])
    u_hist = np.zeros([numInputs, sim_length])
    solve_time_hist = np.zeros([sim_length,])
    error_hist = np.zeros([4, sim_length])

    if visualize:
        fig = plt.figure()
        ax = fig.add_subplot(projection='3d')
        plt.ion()
    if makeGif:
        gifFrames = []

    # Forward Simulate with Controller 
    for i in range(0, sim_length): 

        start = time.time()
        u = controller.solve_for_next_u(x, 
                                        xgoal, 
                                        ulast=u.flatten(), 
                                        ugoal=u, 
                                        mutation_noise=0.63)
        end = time.time()

        x = ground_truth.forward_simulate_dt(x, u, dt) # take the input and apply it to the analytical system
        
        error_hist[:, i] = np.abs((x[4:] - xgoal[4:]).flatten()) # error for theta, position
        solve_time_hist[i] = end - start
        x_hist[:, i] = x.flatten()
        u_hist[:, i] = u.flatten()
        if visualize:
            ground_truth.visualize(x,ax)
        if makeGif:
            imgData = np.frombuffer(plt.gcf().canvas.tostring_rgb(), dtype=np.uint8)
            w, h = plt.gcf().canvas.get_width_height()
            mod = np.sqrt(imgData.shape[0]/(3*w*h)) # multi-sampling of pixels on high-res displays does weird things.
            im = imgData.reshape((int(h*mod), int(w*mod), -1))
            gifFrames.append(Image.fromarray(im))

    iae_metric = np.sum(error_hist)
    avg_resting_pos = np.mean(x_hist[6:, -10:], 1)
    control_successful = np.all(np.abs(avg_resting_pos - xgoal[6:].T) < 0.1)

    if plot:
        print(f'x final is u={x[6]} and v={x[7]}')
        print(f'Average solve time (not including first 3): {np.mean(solve_time_hist)}')
        print(f'IAE measure of control performance: {iae_metric}') # https://www.online-courses.vissim.us/Strathclyde/measures_of_controlled_system_pe.htm
        print(f'Control Successful: {control_successful}')

        ender = -1
        # plt.figure(2)
        # plt.plot(x_hist[4,:ender],':r', x_hist[5,:ender],':b')
        # plt.xlabel("Timestep")
        # plt.ylabel("Joint Vel (rad/s)")
        # plt.legend(['Sim U','Sim V'])
        # plt.grid(True)

        plt.figure(3)
        plt.axhline(y=xgoal[6], color='g', linestyle='-')
        plt.axhline(y=xgoal[7], color='k', linestyle='-')
        plt.plot(x_hist[6,:ender],':r', x_hist[7,:ender],':b')
        plt.xlabel("Timestep")
        plt.ylabel("Joint position (rad)")
        plt.legend(['Goal U', 'Goal V', 'Sim U','Sim V'])
        plt.grid(True)

        # plt.figure(4)
        # plt.plot(x_hist[0,:ender],'r--', x_hist[1,:ender],'b--', x_hist[2,:ender],'g--', x_hist[3,:ender],'k--')
        # plt.xlabel("Timestep")
        # plt.ylabel("Joint Pressures (KPa)")
        # plt.legend(['p0','p1','p2','p3'])
        # plt.grid(True)

        # plt.figure(5)
        # plt.plot(solve_time_hist[3:].T)        
        # plt.ylabel("Timestep")
        # plt.title("Solve time at each time step")
        # plt.grid(True)

        # plt.figure(6)
        # plt.plot(error_hist.T)  
        # plt.xlabel("Timestep")
        # plt.ylabel("Error (rad)")
        # plt.legend(['Error Udot', 'Error Vdot', 'Error U','Error V'])
        # plt.grid(True)

        plt.ioff()
        plt.show()
    if makeGif:
        gifFrames[0].save('/home/daniel/research/catkin_ws/src/hyperparam_optimization/inverted_pendulum/animations/untrainedModel_NEMPC.gif', format='GIF',
                        append_images=gifFrames[1:],
                        save_all=True,
                        duration=realTimeHorizon*dt*.75, loop=1)

    return x, iae_metric, control_successful


if __name__=="__main__":
    # Known working model
    trial_dir = '/home/daniel/research/catkin_ws/src/hyperparam_optimization/bellows_grub/run_logs/fnn_testrun/train_e8d1e215_4_accuracy_tolerance=0.0100,act_fn=tanh,b_size=256,cpu_num=7,dataset_size=120000,dt=0.0100,generate_new_data=False,_2023-06-23_20-31-24'
    
    # get the checkpoint
    import glob
    checkpoint_path = glob.glob(trial_dir + '/lightning_logs/version_0/checkpoints/*.ckpt')[0]

    # get the config file
    with open(trial_dir + '/params.json', 'r') as f:
        config = json.load(f)

    # print(json.dumps(config, indent=4))

    # for test in range(1):
    # Initial Conditions
    x0 = np.array([200,200,200,200,
                    0,0,
                    0.1,0.1]).reshape(8,1) # p, qd, q
    u0 = 200*np.ones([4, 1])

    p_goal = np.array([[0],
                    [0],
                    [0],
                    [0]])
    qd_goal = np.zeros((2,1))
    q_goal = np.random.uniform(low=-np.pi/2, high=np.pi/2, size=(2, 1)) # random joint angle goal

    xgoal = np.vstack([p_goal,qd_goal,q_goal])
    ugoal = np.zeros([4, 1]) # right now in the cost function this is weighted at 0

    # print(f'q_goal is u={q_goal[0]} and v={q_goal[1]}')
    # checkpoint_path = None
    x, iae_metric, control_successful = grub_nempc(checkpoint_path, config, x0, u0, xgoal, ugoal, visualize=False, plot=True)
    print(f'q_goal is u={q_goal[0]} and v={q_goal[1]} \
            \n  q_final is u={x[6]} and v={x[7]}  \
            \nControl {"was Successful" if control_successful else "Failed"} with IAE={iae_metric}\n')