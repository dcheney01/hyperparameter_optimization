import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm 
import json
from PIL import Image

from nonlinear_empc.NonlinearEMPC import NonlinearEMPC
from rad_models.InvertedPendulum import InvertedPendulum
from ip_LearnedModel import IP_LearnedModel

from ip_config import test_ip_config

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
    # params_path = '/home/daniel/research/catkin_ws/src/hyperparam_optimization/inverted_pendulum/run_logs/fnn_optimization/train_7ad1aaeb_156_accuracy_tolerance=0.0100,act_fn=relu,b_size=32,calculates_xdot=False,cpu_num=3,dataset_size=60000,dt=0.0100,ge_2023-05-04_03-41-09/params.json'
    checkpoint_path = '/home/daniel/research/catkin_ws/src/hyperparam_optimization/lightning_logs/version_1/checkpoints/epoch=499-step=1875000.ckpt'

    # params_path = '/home/daniel/research/catkin_ws/src/hyperparam_optimization/inverted_pendulum/run_logs/fnn_optimization/train_21f01584_85_accuracy_tolerance=0.0100,act_fn=relu,b_size=16,calculates_xdot=False,cpu_num=3,dataset_size=60000,dt=0.0100,gen_2023-05-03_20-58-32/params.json'
    # checkpoint_path = '/home/daniel/research/catkin_ws/src/hyperparam_optimization/inverted_pendulum/run_logs/fnn_optimization/train_21f01584_85_accuracy_tolerance=0.0100,act_fn=relu,b_size=16,calculates_xdot=False,cpu_num=3,dataset_size=60000,dt=0.0100,gen_2023-05-03_20-58-32/lightning_logs/version_0/checkpoints/epoch=498-step=1497000.ckpt'
    
    # params_path = '/home/daniel/research/catkin_ws/src/hyperparam_optimization/inverted_pendulum/run_logs/fnn_optimization/train_2528e9d6_184_accuracy_tolerance=0.0100,act_fn=relu,b_size=16,calculates_xdot=False,cpu_num=3,dataset_size=60000,dt=0.0100,ge_2023-05-04_04-34-57/params.json'
    # checkpoint_path = '/home/daniel/research/catkin_ws/src/hyperparam_optimization/inverted_pendulum/run_logs/fnn_optimization/train_2528e9d6_184_accuracy_tolerance=0.0100,act_fn=relu,b_size=16,calculates_xdot=False,cpu_num=3,dataset_size=60000,dt=0.0100,ge_2023-05-04_04-34-57/lightning_logs/version_0/checkpoints/epoch=498-step=1497000.ckpt'

    # params_path = '/home/daniel/research/catkin_ws/src/hyperparam_optimization/inverted_pendulum/run_logs/fnn_optimization/train_716827a7_60_accuracy_tolerance=0.0100,act_fn=relu,b_size=16,calculates_xdot=False,cpu_num=3,dataset_size=60000,dt=0.0100,gen_2023-05-03_16-45-47/params.json'
    # checkpoint_path = '/home/daniel/research/catkin_ws/src/hyperparam_optimization/inverted_pendulum/run_logs/fnn_optimization/train_716827a7_60_accuracy_tolerance=0.0100,act_fn=relu,b_size=16,calculates_xdot=False,cpu_num=3,dataset_size=60000,dt=0.0100,gen_2023-05-03_16-45-47/lightning_logs/version_0/checkpoints/epoch=498-step=1497000.ckpt'

    # params_path = '/home/daniel/research/catkin_ws/src/hyperparam_optimization/inverted_pendulum/run_logs/fnn_optimization/train_e49cea1a_125_accuracy_tolerance=0.0100,act_fn=relu,b_size=32,calculates_xdot=False,cpu_num=3,dataset_size=60000,dt=0.0100,ge_2023-05-04_01-56-18/params.json'
    # checkpoint_path = '/home/daniel/research/catkin_ws/src/hyperparam_optimization/inverted_pendulum/run_logs/fnn_optimization/train_e49cea1a_125_accuracy_tolerance=0.0100,act_fn=relu,b_size=32,calculates_xdot=False,cpu_num=3,dataset_size=60000,dt=0.0100,ge_2023-05-04_01-56-18/lightning_logs/version_0/checkpoints/epoch=498-step=748500.ckpt'

    # with open(params_path, 'r') as f:
    #     config = json.load(f)

    config = test_ip_config
    print(json.dumps(config, indent=4))

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

    realTimeHorizon = 800

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
