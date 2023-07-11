from torch.utils.data import Dataset 
import torch
import json, sys
import numpy as np
from scipy.stats import qmc

"""
This file generates data to train a learned dynamics model of a given system. 
    If there is no data found or the flag is set, data will be generated and overwrite anything in the data folder

    Has the following options for data generation:
        Type of data:
            - random: generates random state and inputs to calculate the output
        Learning Mode (Output):
            - xdot: outputs data as [[current_state, input], [change_in_state]]
            - x: outputs data as [[current_state, input], [next_state]]

    Parameters from config that matter:
        - generate_new_data: whether or not to generate new data
        - learn_mode: determines the format of the output (see above)
        - dataset_size: total size of the dataset
        - normalized: whether or not to normalize all data based on system umax and xmax
        - system: system to generate data on
        - dt: sampling time
"""

class DatasetBaseClass(Dataset):
    def __init__(self, config: dict, system, validation=False):
        self.system = system()
        self.path = config['path'] + ('data/validation_' if validation else 'data/train_') + f'{self.system.name}.json'
        generate_new_data = config['generate_new_data']

        self.learn_mode = config.get('learn_mode', 'x')
        self.size = int(config['dataset_size'] * (0.2 if validation else 0.8)) # This ratio of validation to training data can be adjusted
        self.normalized = config['normalized_data']
        self.dt = config['dt']
        self.sampling_method = config.get('sampling_method', 'random') # default to random sampling if no method is provided

        assert type(self.system) is not None, "System type is None. Lightning Module Base class is not properly overrided"
        
        if not generate_new_data:
            try:
                with open(self.path) as f:
                    self.data_points = [x for x in json.load(f)]
                    if len(self.data_points) != self.size:
                        # Generate new data if the found data is not the same size as requested dataset size
                        print(f"Dataset size is incorrect. Expected {self.size} and got {len(self.data_points)}. Generating new data...")
                        generate_new_data = True
            except:
                print("No data found, will generate new data")
                generate_new_data = True
        
        if generate_new_data:
            self.data_points = self.generate_data()          
    

    def generate_data(self):
        data = []
        
        if self.sampling_method == 'random':
            state_array = ((self.system.xMax - self.system.xMin)*np.random.rand(self.system.numStates,self.size) + self.system.xMin).T
            input_array = ((self.system.uMax - self.system.uMin)*np.random.rand(self.system.numInputs,self.size) + self.system.uMin).T
        elif self.sampling_method == 'lhs':
            state_sampler = qmc.LatinHypercube(d=self.system.numStates)
            state_sample = state_sampler.random(n=self.size)
            state_array = qmc.scale(state_sample, self.system.xMin.squeeze(), self.system.xMax.squeeze())

            input_sampler = qmc.LatinHypercube(d=self.system.numInputs)
            input_sample = input_sampler.random(n=self.size)
            input_array = qmc.scale(input_sample, self.system.uMin, self.system.uMax)

        for j in range(self.size):
            state = state_array[j].reshape(self.system.numStates,1)
            input = input_array[j].reshape(self.system.numInputs,1)

            if self.learn_mode == 'xdot':
                next_state = self.system.calc_state_derivs(state, input)
            elif self.learn_mode == 'x':
                next_state = self.system.forward_simulate_dt(state, input, self.dt)

            if self.normalized:
                state /= self.system.xMax
                input /= self.system.uMax
                next_state /= self.system.xMax

            data.append(([np.vstack((state, input)).squeeze().tolist(), 
                         next_state.squeeze().tolist()]))

        # save data
        with open(self.path, 'w+') as f:
            json.dump(data, f)

        return data
    
    def __len__(self):
        return len(self.data_points)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()
        return self.data_points[idx] #(state, input, next_state)