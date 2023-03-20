from torch.utils.data import DataLoader
from lightning.pytorch import Trainer, seed_everything
from argparse import ArgumentParser

from NN_Architectures import SimpleLinearNN
from ip_dataset import InvertedPendulumDataset
from InvertedPendulumLightning import InvertedPendulumLightning

def main(hparams):
    seed_everything(42, workers=True)

    path = '/home/daniel/research/catkin_ws/src/hyperparam_optimization/inverted_pendulum/'

    train_dataset = InvertedPendulumDataset(path+'train_', generate_new=True, size=1024)
    train_loader = DataLoader(train_dataset, num_workers=6)

    # val_dataset = InvertedPendulumDataset(path+'validation_', generate_new=False, size=128)
    # val_loader = DataLoader(val_dataset)

    # model properties
    input_dim = 3 # theta, thetadot, force
    output_dim = 2 # theta+1, thetadot+1
    model = SimpleLinearNN(input_dim, output_dim)

    ip_lightning = InvertedPendulumLightning(model, logTensorBoard=False)
    print('Lightning Module Set up successfully')

    trainer = Trainer(accelerator=hparams.accelerator, devices=hparams.devices, deterministic=True, max_epochs=10)
    print('Trainer Initialized Successfully')

    trainer.fit(ip_lightning, train_dataloaders=train_loader)
        # validate the model on the validation dataset

        # Output the final accuracy of the model



if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument("--accelerator", default='gpu')
    parser.add_argument("--devices", default=1)
    args = parser.parse_args()

    main(args)