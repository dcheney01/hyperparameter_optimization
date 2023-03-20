import torch
from torch import optim, nn, utils, Tensor
import lightning.pytorch as pl
from NN_Architectures import SimpleLinearNN



class InvertedPendulumLightning(pl.LightningModule):
    def __init__(self, config):
        super().__init__()

        self.model = SimpleLinearNN(config['num_inputs'], 
                                    config['num_outputs'],
                                    config['num_hidden_layers'],
                                    config['hdim'],
                                    config['activation_fn'])
        self.optimizer = torch.optim.Adam(self.model.parameters(), config['lr'])
        self.loss = config['loss_fn']()

    def forward(self, x):
        return self.model(x)
    
    def training_step(self, batch, batch_idx):
        state_input, next_state = batch
        state_input = torch.stack(state_input).float().movedim(0, -1).cuda()
        next_state = torch.stack(next_state).float().movedim(0, -1).cuda()
        
        x_hat = self(state_input)
        train_loss = self.loss(x_hat, next_state)

        self.log("train_loss", train_loss, on_step=True, on_epoch=True, prog_bar=True, logger=True)
            
        return train_loss.float()
    
    def validation_step(self, batch, batch_idx):
        state_input, next_state = batch
        state_input = torch.stack(state_input).float().movedim(0, -1).cuda()
        next_state = torch.stack(next_state).float().movedim(0, -1).cuda()
        
        x_hat = self(state_input)
        val_loss = self.loss(x_hat, next_state)

        within_tol = torch.abs(x_hat - next_state) < 0.1
        accuracy = torch.mean(within_tol.float(), 0)

        x_accuracy = accuracy[0]
        xdot_accuracy = accuracy[1]

        self.log("val_loss", val_loss, on_epoch=True)
        self.log("x_accuracy", x_accuracy, on_epoch=True) 
        self.log("xdot_accuracy", xdot_accuracy, on_epoch=True) 

    def configure_optimizers(self):
        return self.optimizer
