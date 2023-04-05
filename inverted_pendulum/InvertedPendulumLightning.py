import torch
from torch import nn
import pytorch_lightning as pl
from torch.utils.data import DataLoader
import sys

sys.path.append('/home/daniel/research/catkin_ws/src/')
from hyperparam_optimization.NN_Architectures import SimpleLinearNN
from ip_dataset import InvertedPendulumDataset

class InvertedPendulumLightning(pl.LightningModule):
    """ Lightning Module Object that is compatible with PyTorch Lightning Trainer
            - config is a dictionary of options that are used to instantiate this class
            - training and validation data sets are created in this class
    """

    def __init__(self, config):
        """
        - instantiates the model, loss function, and saves the config dict to this object
        """
        super().__init__()

        if config['nn_arch'] == 'simple_fnn':
            network_architecture = SimpleLinearNN
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

        self.model = network_architecture(
            3, # inputs
            2, # outputs
            config['n_hlay'],
            config['hdim'],
            act_fn
        )

        self.config = config

        self.validation_step_outputs_loss = []
        self.validation_step_outputs_x_acc = []
        self.validation_step_outputs_x_dot_acc = []

        self.save_hyperparameters()

    def forward(self, x):
        return self.model(x)
    
    def accuracy(self, prediction, truth):
        """
        Calculates the accuracy of the models prediction
            - currently checks if the absolute difference in prediction and ground truth
                is less than a certain tolerance and takes the average of the whole tensor (for x and xdot)
        """
        within_tol = torch.abs(prediction - truth) < self.config['accuracy_tolerance']
        accuracy = torch.mean(within_tol.float(), 0)
        return accuracy
    
    def training_step(self, batch, batch_idx):
        """
        Currently trains on the loss of prediction and ground truth
        """
        state_input, next_state = batch
        state_input = torch.stack(state_input).float().movedim(0, -1).cuda()
        next_state = torch.stack(next_state).float().movedim(0, -1).cuda()
        
        x_hat = self(state_input)
        train_loss = self.loss(x_hat, next_state).float()
        accuracy = self.accuracy(x_hat, next_state)
            
        self.log("train/loss", train_loss)
        self.log("train/x_accuracy", accuracy[0]) 
        self.log("train/xdot_accuracy", accuracy[1]) 

        return train_loss
    
    def validation_step(self, batch, batch_idx):
        """
        Currently calculates validation stats based on the loss
        """
        state_input, next_state = batch
        state_input = torch.stack(state_input).float().movedim(0, -1).cuda()
        next_state = torch.stack(next_state).float().movedim(0, -1).cuda()
        
        x_hat = self(state_input)
        val_loss = self.loss(x_hat, next_state)

        accuracy = self.accuracy(x_hat, next_state)
        self.validation_step_outputs_loss.append(val_loss)
        self.validation_step_outputs_x_acc.append(accuracy[0])
        self.validation_step_outputs_x_dot_acc.append(accuracy[1])
        
        return {"val/loss": val_loss, "val/x_accuracy": accuracy[0], "val/xdot_accuracy": accuracy[1]}

    def on_validation_epoch_end(self):
        avg_loss = torch.stack(self.validation_step_outputs_loss).mean()
        x_acc = torch.stack(self.validation_step_outputs_x_acc).mean()
        xdot_acc = torch.stack(self.validation_step_outputs_x_dot_acc).mean()

        self.log("val/loss", avg_loss)
        self.log("val/x_accuracy", x_acc)
        self.log("val/xdot_accuracy", xdot_acc)

        self.validation_step_outputs_loss.clear()  # free memory
        self.validation_step_outputs_x_acc.clear() 
        self.validation_step_outputs_x_dot_acc.clear() 

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
        train_dataset = InvertedPendulumDataset(self.config['path']+'/data/train_')
        train_loader = DataLoader(train_dataset, batch_size=self.config['b_size'])
        return train_loader

    def val_dataloader(self):
        val_dataset = InvertedPendulumDataset(self.config['path']+'/data/validation_')
        val_loader = DataLoader(val_dataset, batch_size=self.config['b_size'])
        return val_loader

