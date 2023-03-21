import torch
import torch.nn as nn
import lightning.pytorch as pl
from torch.utils.data import DataLoader

from ip_dataset import InvertedPendulumDataset

class InvertedPendulumLightning(pl.LightningModule):
    def __init__(self, config):
        super().__init__()
        
        num_inputs = config['num_inputs'] 
        num_outputs = config['num_outputs']
        num_hidden_layers = config['num_hidden_layers']
        hdim = config['hdim']
        activation_fn = config['activation_fn']

        model = []
        model.append(nn.Sequential(nn.Linear(num_inputs, hdim), activation_fn))
        for _ in range(num_hidden_layers):
            model.append(nn.Sequential(
                                nn.Linear(hdim, hdim),
                                activation_fn))
        model.append(nn.Sequential(nn.Linear(hdim, num_outputs)))

        self.model = nn.Sequential(*model)
        self.optimizer = torch.optim.Adam(self.model.parameters(), config['lr'])
        self.loss = config['loss_fn']()

        self.accuracy_tolerance = config['accuracy_tolerance']

        self.config = config
        self.validation_step_outputs = []

        self.save_hyperparameters()

    def forward(self, x):
        return self.model(x)
    
    def accuracy(self, prediction, truth):
        within_tol = torch.abs(prediction - truth) < self.accuracy_tolerance
        accuracy = torch.mean(within_tol.float(), 0)
        return accuracy
    
    def training_step(self, batch, batch_idx):
        state_input, next_state = batch
        state_input = torch.stack(state_input).float().movedim(0, -1).cuda()
        next_state = torch.stack(next_state).float().movedim(0, -1).cuda()
        
        x_hat = self(state_input)
        train_loss = self.loss(x_hat, next_state)
        accuracy = self.accuracy(x_hat, next_state)
            
        self.log("train_loss", train_loss)
        self.log("train_accuracy", accuracy[0])    
        return train_loss.float()
    
    def validation_step(self, batch, batch_idx):
        state_input, next_state = batch
        state_input = torch.stack(state_input).float().movedim(0, -1).cuda()
        next_state = torch.stack(next_state).float().movedim(0, -1).cuda()
        
        x_hat = self(state_input)
        val_loss = self.loss(x_hat, next_state)

        accuracy = self.accuracy(x_hat, next_state)
        self.validation_step_outputs.append([val_loss, accuracy[0], accuracy[1]])
        return {"val_loss": val_loss, "val_accuracy": accuracy[0]}

    def on_validation_epoch_end(self):
        avg_loss = torch.stack(self.validation_step_outputs[0]).mean()
        avg_acc = torch.stack(self.validation_step_outputs[1]).mean()
        self.log("val_loss", avg_loss)
        self.log("val_accuracy", avg_acc)
        self.validation_step_outputs.clear()  # free memory

    def configure_optimizers(self):
        return self.optimizer
    
    def train_dataloader(self):
        train_dataset = InvertedPendulumDataset(self.config['path']+'train_', generate_new=self.config['generate_new_data'], size=self.config['training_size'])
        train_loader = DataLoader(train_dataset, batch_size=self.config['batch_size'], num_workers=6)
        return train_loader

    def val_dataloader(self):
        val_dataset = InvertedPendulumDataset(self.config['path']+'validation_', generate_new=self.config['generate_new_data'], size=self.config['validation_size'])
        val_loader = DataLoader(val_dataset, batch_size=self.config['batch_size'], num_workers=6)
        return val_loader

