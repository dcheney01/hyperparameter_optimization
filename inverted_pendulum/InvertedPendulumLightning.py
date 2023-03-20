import torch
from torch import optim, nn, utils, Tensor
import lightning.pytorch as pl

class InvertedPendulumLightning(pl.LightningModule):
    def __init__(self, model, optimizer=torch.optim.Adam, lr=1e-3, loss=nn.MSELoss, logTensorBoard=False):
        super().__init__()
        self.model = model
        self.optimizer = optimizer(self.model.parameters(), lr)
        self.loss = loss()
        self.logTensorBoard = logTensorBoard

        # self.save_hyperparameters()

    def forward(self, x):
        return self.model(x)
    
    def training_step(self, batch, batch_idx):
        state_input, next_state = batch
        state_input = torch.cat(state_input).float().cuda()
        next_state = torch.cat(next_state).float().cuda()
        
        x_hat = self(state_input)
        loss = self.loss(x_hat, next_state)

        if self.logTensorBoard:
            # Logging to TensorBoard (if installed) by default
            self.log("train_loss", loss, on_step=True, on_epoch=True, prog_bar=True, logger=True)
            
        return loss.float()

    def configure_optimizers(self):
        return self.optimizer
