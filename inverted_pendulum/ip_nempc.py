import numpy as np
import matplotlib.pyplot as plt
from tqdm.auto import tqdm 
import json
from PIL import Image
import time
import torch

import sys
sys.path.append('/home/daniel/research/catkin_ws/src/')
from nempc.src.nonlinear_empc.NonlinearEMPC import NonlinearEMPC
from InvertedPendulum import InvertedPendulum
from ip_LearnedModel import IP_LearnedModel

"""
This file takes a learned model of the inverted pendulum and controls it with NEMPC
"""

def CostFunc(x, u, xgoal, ugoal, use_gpu=True, prev_u=None, final_timestep=False):
    """
    Calculates the cost of an input and state.

    Arguments
    ============
    x: The state at the current time step.
    u: The input to the system at the current time step.

    Returns
    ============
    cost: The cost at the current time step.
    """
    # angle wrapping (only wrap theta dot)
    wrapAngle = 1
    wrapRange = (-np.pi, np.pi)

    low = wrapRange[0]
    high = wrapRange[1]
    cycle = high-low
    x[wrapAngle] = (x[wrapAngle]+cycle/2) % cycle + low

    # Cost
    # start = time.time()
    if use_gpu:
        Q = 1.0*torch.diag(torch.tensor([0.0, 1.0])).cuda()
        Qf = 5.0*torch.diag(torch.tensor([0.0, 1.0])).cuda()
        R = .0*torch.diag(torch.tensor([1.0])).cuda()
        if final_timestep:
            Qx = torch.abs(torch.mm(Qf, x-xgoal)**2.0)
            cost = torch.sum(Qx, axis=0)
        else:
            Qx = torch.abs(torch.mm(Q, x-xgoal)**2.0)
            Ru = torch.abs(torch.mm(R, u-ugoal))
            cost = torch.sum(Qx, axis=0) + torch.sum(Ru, axis=0)
        # end = time.time()
        # print(f'cost time: {end-start}')
    else:
        Q = 1.0*np.diag([0, 1.0])
        Qf = 5.0*np.diag([0, 1.0])
        R = .0*np.diag([1.0])
        if final_timestep:
            Qx = np.abs(Qf.dot(x-xgoal)**2.0)
            cost = np.sum(Qx, axis=0)
        else:
            Qx = np.abs(Q.dot(x-xgoal)**2.0)
            Ru = np.abs(R.dot(u-ugoal))
            cost = np.sum(Qx,axis=0) + np.sum(Ru,axis=0)
        # end = time.time()
        # print(f'cost time: {end-start}')
    return cost


def ip_nempc(checkpoint_path, config, visualize=False, plot=False):
    # NEMPC Controller Parameters
    horizon = 50
    numSims = 500
    numParents = 100
    numStrangers = 200
    mutation_probability = 0.6
    mutation_noise = .63
    selection_method = 'elitism'
    tournament_size = 0
    crossover_method = 'knot_point'

    dt = config['dt']
    useGPU = False
    use_model_gpu = False

    def cost_wrapper(x, u, xgoal, ugoal, prev_u=None, final_timestep=False):
        use_gpu = useGPU
        return CostFunc(x, u, xgoal, ugoal, use_gpu=use_gpu, prev_u=prev_u, final_timestep=final_timestep)

    # Other parameters
    realTimeHorizon = 600
    makeGif = False
    if makeGif:
        gifFrames = []

    # Objects for the actual control
    # The learned model is used to predict the best inputs and the analytical model is used as ground truth to apply the predicted inputs
    learned_sys = IP_LearnedModel(model_path=checkpoint_path, config=config, use_gpu=use_model_gpu)
    learned_forward = learned_sys.forward_simulate_dt

    analytical_sys = InvertedPendulum()
    analytical_forward = analytical_sys.forward_simulate_dt

    controller = NonlinearEMPC(learned_forward, 
                                cost_wrapper, 
                                learned_sys.numStates, 
                                learned_sys.numInputs, 
                                learned_sys.uMin, 
                                learned_sys.uMax, 
                                horizon, 
                                dt, 
                                numSims, 
                                numKnotPoints=4, 
                                useGPU=useGPU, 
                                mutation_probability=mutation_probability, 
                                mutation_noise=mutation_noise, 
                                selection_method=selection_method, 
                                tournament_size=tournament_size, 
                                numParents=numParents, 
                                numStrangers=numStrangers, 
                                crossover_method=crossover_method,
                                seed=True,
                                display=plot)

    # Initial Conditions
    x = np.array([0, -np.pi]).reshape(learned_sys.numStates, 1) # initial state (thetadot, theta)
    u = np.zeros([learned_sys.numInputs, 1]) # initial input

    xgoal = np.zeros([learned_sys.numStates, 1])
    ugoal = np.zeros([learned_sys.numInputs, 1])

    # Plotting variables
    x_hist = np.zeros([learned_sys.numStates, realTimeHorizon])
    u_hist = np.zeros([learned_sys.numInputs, realTimeHorizon])
    error_hist = np.zeros([realTimeHorizon,])
    solve_time_hist = np.zeros([realTimeHorizon,])

    if visualize:
        plt.gcf().canvas.draw()

    # Forward Simulate with Controller
    for i in tqdm(range(0, realTimeHorizon), position=0, leave=True):            
        x_hist[:, i] = x.flatten()
        u_hist[:, i] = u.flatten()

        # Run one time step of NEMPC
        start = time.time()
        u = controller.solve_for_next_u(x, xgoal, ulast=u, ugoal=ugoal, mutation_noise=mutation_noise) # get optimal u from NEMPC
        end = time.time()
        # print("solve time: ", time.time()-start) # how long it takes to calculate u
        solve_time_hist[i] = (end - start)

        x = analytical_forward(x, u, dt) # take the input and apply it to the analytical system
        if x[1] < -np.pi:
            theta = -np.pi + np.abs(x[1] + np.pi)
        else:
            theta = x[1]

        error_hist[i] = np.abs(theta - xgoal[1]) # error for theta, position

        if visualize:
            analytical_sys.visualize(x)

        if makeGif:
            imgData = np.frombuffer(plt.gcf().canvas.tostring_rgb(), dtype=np.uint8)
            w, h = plt.gcf().canvas.get_width_height()
            mod = np.sqrt(imgData.shape[0]/(3*w*h)) # multi-sampling of pixels on high-res displays does weird things.
            im = imgData.reshape((int(h*mod), int(w*mod), -1))
            gifFrames.append(Image.fromarray(im))


    if makeGif:
        gifFrames[0].save('/home/daniel/research/catkin_ws/src/hyperparam_optimization/inverted_pendulum/animations/untrainedModel_NEMPC.gif', format='GIF',
                        append_images=gifFrames[1:],
                        save_all=True,
                        duration=realTimeHorizon*dt*.75, loop=1)
        
    iae_metric = np.sum(error_hist)
    avg_resting_pos = np.mean(x_hist[1, -20:])
    if avg_resting_pos < -1.8*np.pi: # meaning the angle is flipped (it settled at 2pi instead of 0, which is the same)
        control_successful = avg_resting_pos > -2.1*np.pi and avg_resting_pos < -1.9*np.pi
    else:
        control_successful = avg_resting_pos < xgoal[0] + 0.1 and avg_resting_pos > xgoal[0] - 0.1

    if plot:
        print(f'Average solve time (not including first 3): {np.mean(solve_time_hist)}')
        print(f'IAE measure of control performance: {iae_metric}') # https://www.online-courses.vissim.us/Strathclyde/measures_of_controlled_system_pe.htm

        # Plotting                    
        plt.figure("Position")
        plt.subplot(5, 1, 1)
        plt.plot(x_hist[1, :].T)
        plt.ylabel("Theta (rad)")
        plt.title("Nonlinear EMPC Position")
        plt.grid(True)

        plt.subplot(5, 1, 2)
        plt.plot(x_hist[0, :].T)        
        plt.ylabel("Vel (rad/s)")
        plt.title("Nonlinear EMPC Velocity")
        plt.grid(True)

        plt.subplot(5, 1, 3)
        plt.plot(u_hist[0, :].T)
        plt.ylabel("Torque (N")
        plt.title("Nonlinear EMPC Inputs")
        plt.grid(True)

        plt.subplot(5, 1, 4)
        plt.plot(solve_time_hist[3:].T)        
        plt.ylabel("Time (s)")
        plt.title("Solve time at each time step")
        plt.grid(True)

        plt.subplot(5, 1, 5)
        plt.plot(error_hist.T)        
        plt.ylabel("IAE")
        plt.title("Control error at each time step")
        plt.grid(True)

        plt.ioff()
        plt.show()

    return iae_metric, control_successful


if __name__=="__main__":
    # Known working model
    trial_dir = '/home/daniel/research/catkin_ws/src/hyperparam_optimization/inverted_pendulum/run_logs/fnn_optimization/train_21f01584_85_accuracy_tolerance=0.0100,act_fn=relu,b_size=16,calculates_xdot=False,cpu_num=3,dataset_size=60000,dt=0.0100,gen_2023-05-03_20-58-32'
    
    # get the checkpoint
    import glob
    checkpoint_path = glob.glob(trial_dir + '/lightning_logs/version_0/checkpoints/*.ckpt')[0]

    # get the config file
    with open(trial_dir + '/params.json', 'r') as f:
        config = json.load(f)

    print(json.dumps(config, indent=4))

    ip_nempc(checkpoint_path=checkpoint_path, config=config, visualize=True, plot=True)