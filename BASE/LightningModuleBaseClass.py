import torch
from torch import nn
import pytorch_lightning as pl
from torch.utils.data import DataLoader
import sys

sys.path.append('/home/daniel/research/catkin_ws/src/')
from hyperparam_optimization.BASE.NN_Architectures import *

""" 
Lightning Module Object that is compatible with PyTorch Lightning Trainer
"""

class LightningModuleBaseClass(pl.LightningModule):
    def __init__(self, config:dict):
        super().__init__()

        self.config = config
        self.system = None
        self.dataset = None

        # Set network architecture
        if config['nn_arch'] == 'simple_fnn':
            network_architecture = SimpleLinearNN
        elif config['nn_arch'] == 'lstm':
            network_architecture = LSTM_CUSTOM
        elif config['nn_arch'] == 'transformer':
            network_architecture = TransformerModule
        else:
            raise ValueError(f"Unknown Network Architecture. Got {config['nn_arch']}")

        # Get activation function
        if config['act_fn'] == 'relu':
            act_fn = nn.ReLU()
        elif config['act_fn'] == 'tanh':
            act_fn = nn.Tanh()
        elif config['act_fn'] == 'sigmoid':
            act_fn = nn.Sigmoid()
        elif config['act_fn'] == 'softmax': 
            act_fn = nn.Softmax()
        else:
            raise ValueError(f"Incorrect Activation Function in tune search space. Got {config['act_fn']}")
        
        # Get loss function
        if config['loss_fn'] == 'mse':
            self.loss = nn.MSELoss()
        elif config['loss_fn'] == 'mae':
            self.loss = nn.L1Loss()
        elif config['loss_fn'] == 'cross_entropy':
            self.loss = nn.CrossEntropyLoss()
        else:
            raise ValueError(f"Incorrect Loss Function in tune search space. Got {config['loss_fn']}")

        self.save_hyperparameters()
        
        self.model = network_architecture(
            config['n_inputs'],
            config['n_outputs'],
            config['n_hlay'],
            config['hdim'],
            act_fn
        )
        self.validation_step_outputs_loss = []


    def forward(self, x):
        return self.model(x)
    

    def accuracy(self, prediction, truth):
        """
        Calculates the accuracy of the models prediction
        """
        within_tol = torch.abs(prediction - truth) < self.config['accuracy_tolerance']
        accuracy = torch.mean(within_tol.float(), 0)
        return accuracy
    

    def calculate_metrics(self, x_hat, next_state, validation=False):
        raise NotImplementedError("Calculate Metrics has not been implemented by the base class")
    

    def training_step(self, batch, batch_idx):
        """
        Currently trains on the loss of prediction and ground truth
        """
        state_input, next_state = batch
        state_input = torch.stack(state_input).float().movedim(0, -1).cuda()
        next_state = torch.stack(next_state).float().movedim(0, -1).cuda()
        
        x_hat = self(state_input)
        train_loss = self.loss(x_hat, next_state).float()
        self.log("train/loss", train_loss)

        self.calculate_metrics(x_hat, next_state)  

        return train_loss
    

    def validation_step(self, batch, batch_idx):
        """
        Calculates validation stats 
        """
        state_input, next_state = batch
        state_input = torch.stack(state_input).float().movedim(0, -1).cuda()
        next_state = torch.stack(next_state).float().movedim(0, -1).cuda()
        
        x_hat = self(state_input)
        val_loss = self.loss(x_hat, next_state)

        metric_dict = self.calculate_metrics(x_hat, next_state, validation=True)
        metric_dict['val/loss'] = val_loss
        self.validation_step_outputs_loss.append(val_loss)

        return metric_dict


    def on_validation_epoch_end(self):
        avg_loss = torch.stack(self.validation_step_outputs_loss).mean()
        self.log("val/loss", avg_loss)
        self.validation_step_outputs_loss.clear()  # free memory

    def configure_optimizers(self):
        # Get optimizer
        if self.config['opt'] == 'adam':
            optimizer = torch.optim.Adam
        elif self.config['opt'] == 'sgd':
            optimizer = torch.optim.SGD
        elif self.config['opt'] == 'ada':
            optimizer = torch.optim.Adagrad
        elif self.config['opt'] == 'lbfgs':
            optimizer = torch.optim.LBFGS
        elif self.config['opt'] == 'rmsprop':
            optimizer = torch.optim.RMSprop
        else:
            raise ValueError(f"Incorrect Optimizer in tune search space. Got {self.config['opt']}")

        return optimizer(self.model.parameters(), self.config['lr'])
    
    def train_dataloader(self):
        train_dataset = self.dataset(self.config, self.system, validation=False)
        train_loader = DataLoader(train_dataset, batch_size=self.config['b_size'], num_workers=self.config['num_workers'])
        return train_loader

    def val_dataloader(self):
        val_dataset = self.dataset(self.config, self.system, validation=True)
        val_loader = DataLoader(val_dataset, batch_size=self.config['b_size'], num_workers=self.config['num_workers'])
        return val_loader

