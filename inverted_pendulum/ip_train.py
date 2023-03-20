from torch.utils.data import DataLoader
from lightning.pytorch import Trainer, seed_everything
from argparse import ArgumentParser
from torch import nn

from ip_dataset import InvertedPendulumDataset
from InvertedPendulumLightning import InvertedPendulumLightning

def main(hparams):
    seed_everything(42, workers=True)

    path = '/home/daniel/research/catkin_ws/src/hyperparam_optimization/inverted_pendulum/'

    config = {
            'max_epochs': 20,
            'batch_size': 64,
            'num_inputs': 3, 
            'num_outputs': 2,
            'num_hidden_layers': 2,
            'hdim': 20,
            'activation_fn': nn.ReLU(),
            'lr': 0.001,
            'loss_fn': nn.MSELoss,
            }


    train_dataset = InvertedPendulumDataset(path+'train_', generate_new=True, size=10000)
    train_loader = DataLoader(train_dataset, batch_size=config['batch_size'], num_workers=6)

    val_dataset = InvertedPendulumDataset(path+'validation_', generate_new=True, size=2048)
    val_loader = DataLoader(val_dataset, batch_size=config['batch_size'], num_workers=6)


    ip_lightning = InvertedPendulumLightning(config)
    print('Lightning Module Set up successfully')

    trainer = Trainer(accelerator=hparams.accelerator, devices=hparams.devices, deterministic=True,\
                       max_epochs=config['max_epochs'], check_val_every_n_epoch=2, log_every_n_steps=25)
    print('Trainer Initialized Successfully')

    trainer.fit(ip_lightning, train_dataloaders=train_loader, val_dataloaders=val_loader)
    # validate the model on the validation dataset

    # Output the final accuracy of the model
    trainer.validate(ip_lightning, val_loader, verbose=True)



if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument("--accelerator", default='gpu')
    parser.add_argument("--devices", default=1)
    args = parser.parse_args()

    main(args)