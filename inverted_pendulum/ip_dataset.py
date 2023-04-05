from torch.utils.data import Dataset 
import torch
import json
import numpy as np

from rad_models.InvertedPendulum import InvertedPendulum

"""
This file generates data to train a learned dynamics model of the Inverted Pendulum

    Currently calculates a random state and input pair and then calculates the next state based on 
    the [state, input] (Markov Assumption)

    Can give data as output is x (full state) or xdot (change in state between time steps)

    Finds data in discrete time steps of 0.01, can change below
"""

class InvertedPendulumDataset(Dataset):
    def __init__(self, path, learn_mode='x', generate_new=False, size=1024):
        self.path = path + 'ip_data.json'

        if not generate_new:
            try:
                with open(self.path) as f:
                    self.data_points = [x for x in json.load(f)]   
            except:
                print("No data found, will generate new data")
                generate_new = True
        
        if generate_new:
            self.data_points = self.generate_data(size, learn_mode=learn_mode)          
    

    def generate_data(self, size, learn_mode='x', normalized=False):
        ip = InvertedPendulum()
        data = []
        
        for j in range(size):
            state = (ip.xMax - ip.xMin)*np.random.rand(ip.numStates,1) + ip.xMin
            input = (ip.uMax - ip.uMin)*np.random.rand(ip.numInputs,1) + ip.uMin

            if learn_mode == 'xdot':
                next_state = ip.calc_state_derivs(state,input)
            elif learn_mode == 'x':
                next_state = ip.forward_simulate_dt(state, input, 0.01) # last arg is dt

            if normalized == True:
                state /= ip.xMax
                input /= ip.uMax
                next_state /= ip.xMax

            data.append(([state[0][0], state[1][0], input[0][0]], 
                         [next_state[0][0], next_state[1][0]]))

        # save data
        with open(self.path, 'w') as f:
            json.dump(data, f)

        return data

    def __len__(self):
        return len(self.data_points)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()
        return self.data_points[idx] #(state, input, next_state)
    

if __name__=='__main__':
    path = '/home/daniel/research/catkin_ws/src/hyperparam_optimization/inverted_pendulum/'
    
    val_path = path + 'data/validation_'
    train_path = path + 'data/train_'
    data_size = 60000 # 60000 is equivalent to 10 minutes of real time at dt=0.01 

    val_data = InvertedPendulumDataset(val_path, learn_mode='x', generate_new=True, size=int(data_size*0.2))
    train_data = InvertedPendulumDataset(train_path, learn_mode='x', generate_new=True, size=data_size)