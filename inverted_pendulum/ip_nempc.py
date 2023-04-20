import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm 
import json
from PIL import Image

from nonlinear_empc.NonlinearEMPC import NonlinearEMPC
from rad_models.InvertedPendulum import InvertedPendulum
from ip_LearnedModel import IP_LearnedModel

"""
This file takes a learned model of the inverted pendulum and controls it with NEMPC
"""


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

    # angle wrapping (only wrap theta dot)
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
        Qx = np.abs(Q.dot(x-xgoal)**2.0)
        Ru = np.abs(R.dot(u-ugoal))
        cost = np.sum(Qx,axis=0) + np.sum(Ru,axis=0)
    return cost

if __name__=="__main__":
    # Very accurate Model (0.01 rads)
    params_path = '/home/daniel/research/catkin_ws/src/hyperparam_optimization/inverted_pendulum/run_logs/train_ip_2023-04-04_10-43-04/train_ip_1fe95f8c_13_accuracy_tolerance=0.0100,act_fn=relu,b_size=16,hdim=256,loss_fn=mse,lr=0.0003,max_epochs=500,n_hlay=0,nn_arc_2023-04-04_11-38-55/params.json'
    checkpoint_path = '/home/daniel/research/catkin_ws/src/hyperparam_optimization/inverted_pendulum/run_logs/train_ip_2023-04-04_10-43-04/train_ip_1fe95f8c_13_accuracy_tolerance=0.0100,act_fn=relu,b_size=16,hdim=256,loss_fn=mse,lr=0.0003,max_epochs=500,n_hlay=0,nn_arc_2023-04-04_11-38-55/data-points-run/fancy-totem-203/hyperparam_opt_ip/ju0gk1hp/checkpoints/epoch=498-step=1871250.ckpt'
    
    # Less accurate Model (0.1 rads)
    # params_path = '/home/daniel/research/catkin_ws/src/hyperparam_optimization/inverted_pendulum/run_logs/train_ip_2023-03-31_16-25-01/train_ip_8cfb3b02_45_accuracy_tolerance=0.1000,act_fn=relu,b_size=32,hdim=32,loss_fn=mse,lr=0.0011,max_epochs=250,n_hlay=1,nn_arch_2023-03-31_17-29-03/params.json'
    # checkpoint_path = '/home/daniel/research/catkin_ws/src/hyperparam_optimization/inverted_pendulum/run_logs/train_ip_2023-03-31_16-25-01/train_ip_8cfb3b02_45_accuracy_tolerance=0.1000,act_fn=relu,b_size=32,hdim=32,loss_fn=mse,lr=0.0011,max_epochs=250,n_hlay=1,nn_arch_2023-03-31_17-29-03/data-points-run/magic-meadow-135/hyperparam_opt_ip/wa40a705/checkpoints/epoch=248-step=77937.ckpt'
    
    with open(params_path, 'r') as f:
        config = json.load(f)
    # ran a few without this in the config
    config['calculates_xdot'] = False
    json.dumps(config)

    learned_sys = IP_LearnedModel(model_path=None, config=config)
    learned_forward = learned_sys.forward_simulate_dt

    analytical_sys = InvertedPendulum()
    analytical_forward = analytical_sys.forward_simulate_dt

    horizon = 50
    numSims = 1000
    numParents = 200
    numStrangers = 600
    mutation_probability = 0.6
    mutation_noise = .63
    selection_method = 'elitism'
    tournament_size = 0
    crossover_method = 'knot_point'

    dt = .01
    visualize = True
    useGPU = False
    angle_wrapping = True

    controller = NonlinearEMPC(learned_forward, 
                                        CostFunc, 
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
                                        seed=True)

    realTimeHorizon = 525

    # Initial Conditions
    elapsedTime = 0
    x0 = np.array([0, -np.pi]).reshape(learned_sys.numStates, 1)
    u0 = np.zeros([learned_sys.numInputs, 1])

    xgoal = np.zeros([learned_sys.numStates, 1])
    ugoal = np.zeros([learned_sys.numInputs, 1])
    
    x = x0 # initial state
    u = u0 # initial inputs
    x_hist = np.zeros([learned_sys.numStates, realTimeHorizon])
    u_hist = np.zeros([learned_sys.numInputs, realTimeHorizon])
    gifFrames = []
    t = np.linspace(0,realTimeHorizon,num=int(realTimeHorizon*50+1))
    best_cost_hist = np.zeros(t.shape)
    goal_count = 0
    makeGif = False
    plt.gcf().canvas.draw()

    # Forward Simulate with Controller
    for i in tqdm(range(0, realTimeHorizon)):            
        x_hist[:, i] = x.flatten()
        u_hist[:, i] = u.flatten()

        # NEMPC
        # start = time.time()
        u = controller.solve_for_next_u(x, xgoal, ulast=u, ugoal=ugoal, mutation_noise=mutation_noise) # get optimal u from NEMPC
        # print("solve time: ", time.time()-start) # how long it takes to calculate u
        
        x = analytical_forward(x, u, dt) # take the input and apply it to the analytical system
        analytical_sys.visualize(x) # visualize it on the physical system     

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
