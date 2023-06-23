from pytorch_lightning import seed_everything
import torch
from ray import tune

import sys
sys.path.append('/home/daniel/research/catkin_ws/src/')
from hyperparam_optimization.BASE.optimize_learned_model import optimize_system, train
from hyperparam_optimization.BASE.DatasetBaseClass import DatasetBaseClass
from hyperparam_optimization.BASE.LightningModuleBaseClass import LightningModuleBaseClass

from BellowsGrub import BellowsGrub

class GrubDataset(DatasetBaseClass):
    pass

class GrubLightningModule(LightningModuleBaseClass):
    def __init__(self, config: dict):
        super().__init__(config)
        self.system = BellowsGrub
        self.dataset = GrubDataset

        self.validation_step_outputs_acc1 = []
        self.validation_step_outputs_acc2 = []


    def calculate_metrics(self, x_hat, next_state, validation=False):
        accuracy = self.accuracy(x_hat, next_state)

        if not validation:
            self.log("train/u_accuracy", accuracy[6]) 
            self.log("train/v_accuracy", accuracy[7]) 

        else:
            self.validation_step_outputs_acc1.append(accuracy[6])
            self.validation_step_outputs_acc2.append(accuracy[7])

            return {"val/u_accuracy": accuracy[6], 
                    "val/v_accuracy": accuracy[7]}
        
    def on_validation_epoch_end(self):
        super().on_validation_epoch_end() # to get the validation stats for loss

        acc1 = torch.stack(self.validation_step_outputs_acc1).mean()
        acc2 = torch.stack(self.validation_step_outputs_acc2).mean()

        self.log(f"val/u_accuracy", acc1)
        self.log(f"val/v_accuracy", acc2)

        self.validation_step_outputs_acc1.clear() 
        self.validation_step_outputs_acc2.clear() 


grub_config = {
    'project_name': 'hyperparam_opt_grub',
    'run_name': 'fnn_try4_moredata',

    'nn_arch': 'simple_fnn', #'lstm', # 'transformer'

    # Parameters to Optimize
    'b_size': tune.choice([16, 32, 64, 128, 256]), #512
    'n_hlay': tune.choice(list(i for i in range(7))),
    'hdim': tune.choice(list(2**i for i in range(3, 11))),
    'lr': tune.loguniform(1e-4, 1e-2),
    'act_fn': tune.choice(['relu', 'tanh', 'sigmoid']),
    'loss_fn': tune.choice(['mse', 'mae']),#, 'cross_entropy']),
    'opt': tune.choice(['adam', 'sgd']),# 'ada', 'lbfgs', 'rmsprop']),

    'metric': 'val/loss',
    'mode': 'min',
    'logmetric1_name': 'u_accuracy',
    'logmetric2_name': 'v_accuracy',

    # Parameters to Optimize
    # 'b_size': 32,
    # 'n_hlay': 1,
    # 'hdim': 128,
    # 'lr': 0.0001,
    # 'act_fn': 'relu',
    # 'loss_fn': 'mse',
    # 'opt': 'adam',

    'n_inputs': 12, # [p, qd, q, u] at t
    'n_outputs': 8, #[p, qd, q] at t+1

    # Other Configuration Parameters
    'accuracy_tolerance': 0.05, # This translates to about 2.9 degrees
    'num_workers': 6,
    'generate_new_data': False,
    'learn_mode': 'x',
    'dataset_size': 120000,
    'normalized_data': True,
    'dt': 0.01,
    'cpu_num': 7,
    'gpu_num': 1.0,
    
    # Optimization Tool Parameters
    'max_epochs': 500,
    'num_samples': 100,
    'path': '/home/daniel/research/catkin_ws/src/hyperparam_optimization/bellows_grub/',
}

if __name__ == '__main__':
    seed_everything(42, workers=True)

    # Generate New Data --------------------------------------------------
    # try:
    #     GrubDataset(grub_config, BellowsGrub)
    #     GrubDataset(grub_config, BellowsGrub, validation=True)
    #     print("Grub Dataset class loaded...")
    # except Exception as e:
    #     print(f"Could not instantiate the Grub Dataset. Got the following error: {e}")

    # # Test the Lightning Module class --------------------------------------------------
    # try:
    #     lightning_module = GrubLightningModule(grub_config)
    #     print("Inverted Pendulum Lightning Module loaded...")
    # except Exception as e:
    #     print(f"Could not instantiate the Inverted Pendulum Lightning Module. Got the following error: {e}")

    # # Test the training loop -------------------------------------------
    # try:
    #     train(grub_config, 
    #             GrubLightningModule,
    #             notune=True)
    #     print("\nTraining Loop Successfully Tested...")
    # except Exception as e: 
    #     print(f"\nCould not complete training loop due to {e}")

    # Run the actual optimization
    optimize_system(grub_config, GrubLightningModule)
