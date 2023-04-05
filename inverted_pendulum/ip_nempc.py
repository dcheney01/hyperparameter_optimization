import numpy as np
import pandas as pd
import time
import matplotlib.pyplot as plt
from nonlinear_empc.NonlinearEMPC import NonlinearEMPC
from rad_models.InvertedPendulum import InvertedPendulum

from tqdm import tqdm 

import json

# this is a nonlinear pytorch model right now
# from rad_models.LearnedInvertedPendulum import *
# just importing model for simulation, wouldn't normally do this for real system
from ip_LearnedModel import IP_LearnedModel

def CostFunc(x, u, xgoal, ugoal, final_timestep=False):

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
    Q = 1.0*np.diag([0, 1.0])
    Qf = 5.0*np.diag([0, 1.0])
    R = .0*np.diag([1.0])

    # angle wrapping
    wrapAngle = 1
    wrapRange = (-np.pi, np.pi)

    low = wrapRange[0]
    high = wrapRange[1]
    cycle = high-low
    x[wrapAngle] = (x[wrapAngle]+cycle/2) % cycle + low

    # Cost
    cost = np.zeros((u.shape))
    if final_timestep:
        Qx = np.abs(Qf.dot(x-xgoal)**2.0)
        cost = np.sum(Qx,axis=0)
    else:
        # from IPython import embed; embed()
        Qx = np.abs(Q.dot(x-xgoal)**2.0)
        Ru = np.abs(R.dot(u-ugoal))
        cost = np.sum(Qx,axis=0) + np.sum(Ru,axis=0)
    return cost

if __name__=="__main__":
    params_path = '/home/daniel/research/catkin_ws/src/hyperparam_optimization/inverted_pendulum/run_logs/train_ip_2023-04-04_10-43-04/train_ip_1fe95f8c_13_accuracy_tolerance=0.0100,act_fn=relu,b_size=16,hdim=256,loss_fn=mse,lr=0.0003,max_epochs=500,n_hlay=0,nn_arc_2023-04-04_11-38-55/params.json'
    checkpoint_path = '/home/daniel/research/catkin_ws/src/hyperparam_optimization/inverted_pendulum/run_logs/train_ip_2023-04-04_10-43-04/train_ip_1fe95f8c_13_accuracy_tolerance=0.0100,act_fn=relu,b_size=16,hdim=256,loss_fn=mse,lr=0.0003,max_epochs=500,n_hlay=0,nn_arc_2023-04-04_11-38-55/data-points-run/fancy-totem-203/hyperparam_opt_ip/ju0gk1hp/checkpoints/epoch=498-step=1871250.ckpt'
    
    # params_path = 

    # model_config = torch.load(model_path)
    # print(model_config)
    with open(params_path, 'r') as f:
        config = json.load(f)

    # ran a few without this in the config
    config['calculates_xdot'] = False

    json.dumps(config)

    sys = IP_LearnedModel(checkpoint_path, config)
    analytical_sys = InvertedPendulum()
    forward_sim_func = sys.forward_simulate_dt
    Q = 1.0*np.diag([0, 1.0])
    Qf = 5.0*np.diag([0, 1.0])
    R = .0*np.diag([1.0])
    xgoal = np.zeros([sys.numStates, 1])
    ugoal=np.zeros([sys.numInputs, 1])
    horizon = 50
    numSims = 1000
    numParents = 200
    numStrangers = 600
    dt = .01
    mutation_probability = 0.6
    mutation_noise = .63
    useGPU = False
    x0 = np.array([0, -np.pi]).reshape(sys.numStates, 1)
    u0 = np.zeros([sys.numInputs, 1])
    visualize = True
    selection_method = 'elitism'
    tournament_size = 0
    crossover_method = 'knot_point'
    angle_wrapping = True

    controller = NonlinearEMPC(forward_sim_func, 
                                        CostFunc, 
                                        sys.numStates, 
                                        sys.numInputs, 
                                        sys.uMin, 
                                        sys.uMax, 
                                        horizon, 
                                        dt, 
                                        numSims, 
                                        numKnotPoints = 4, 
                                        useGPU = useGPU, 
                                        mutation_probability = mutation_probability, 
                                        mutation_noise = mutation_noise, 
                                        selection_method = selection_method, 
                                        tournament_size = tournament_size, 
                                        numParents = numParents, 
                                        numStrangers = numStrangers, 
                                        crossover_method = crossover_method,
                                        seed=True)

    realTimeHorizon = 800
    # Initial Conditions
    elapsedTime = 0
    x = x0 # initial state
    u = u0 # initial inputs
    x_hist = np.zeros([sys.numStates, realTimeHorizon])
    u_hist = np.zeros([sys.numInputs, realTimeHorizon])
    t = np.linspace(0,realTimeHorizon,num=int(realTimeHorizon*50+1))
    best_cost_hist = np.zeros(t.shape)
    goal_count = 0

    # Forward Simulate with Controller
    # for i in range(0, realTimeHorizon):  
    for i in tqdm(range(0, realTimeHorizon)):            

        x_hist[:, i] = x.flatten()
        u_hist[:, i] = u.flatten()

        # NEMPC
        # start = time.time()
        u = controller.solve_for_next_u(x, xgoal, ulast=u, ugoal = ugoal, mutation_noise = mutation_noise) # get optimal u from NEMPC
        # # print("solve time: ", time.time()-start) # how long it takes to calculate u
        
        x = analytical_sys.forward_simulate_dt(x, u, dt)
        analytical_sys.visualize(x) # visualize it          

    # Plotting                    
    plt.figure("Position")
    plt.subplot(3, 1, 1)
    plt.plot(x_hist[1, :].T)
    plt.ylabel("Theta (rad)")
    plt.title("Nonlinear EMPC Position")
    plt.grid(True)

    plt.subplot(3, 1, 2)
    plt.plot(x_hist[0, :].T)        
    plt.ylabel("Vel (rad/s)")
    plt.title("Nonlinear EMPC Velocity")
    plt.grid(True)

    plt.subplot(3, 1, 3)
    plt.plot(u_hist[0, :].T)
    plt.xlabel("Timestep")
    plt.ylabel("Torque (N")
    plt.title("Nonlinear EMPC Inputs")
    plt.grid(True)
    plt.ioff()
    plt.show()
