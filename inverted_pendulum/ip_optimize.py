from pytorch_lightning import seed_everything
import torch 
from ray import tune

import sys
sys.path.append('/home/daniel/research/catkin_ws/src/')
from hyperparam_optimization.BASE.optimize_learned_model import optimize_system, train
from hyperparam_optimization.BASE.DatasetBaseClass import DatasetBaseClass
from hyperparam_optimization.BASE.LightningModuleBaseClass import LightningModuleBaseClass

from InvertedPendulum import InvertedPendulum

class InvertedPendulumDataset(DatasetBaseClass):
    pass

class InvertedPendulumLightningModule(LightningModuleBaseClass):
    def __init__(self, config: dict):
        super().__init__(config)
        self.system = InvertedPendulum
        self.dataset = InvertedPendulumDataset

        self.validation_step_outputs_acc1 = []
        self.validation_step_outputs_acc2 = []

    def calculate_metrics(self, x_hat, next_state, validation=False):
        accuracy = self.accuracy(x_hat, next_state)
        if not validation:
            self.log("train/x_accuracy", accuracy[0]) 
            self.log("train/xdot_accuracy", accuracy[1]) 

        else:
            self.validation_step_outputs_acc1.append(accuracy[0])
            self.validation_step_outputs_acc2.append(accuracy[1])

            return {f"val/x_accuracy": accuracy[0], 
                    f"val/xdot_accuracy": accuracy[1]}

    def on_validation_epoch_end(self):
        super().on_validation_epoch_end() # to get the validation stats for loss

        acc1 = torch.stack(self.validation_step_outputs_acc1).mean()
        acc2 = torch.stack(self.validation_step_outputs_acc2).mean()

        self.log(f"val/x_accuracy", acc1)
        self.log(f"val/xdot_accuracy", acc2)

        self.validation_step_outputs_acc1.clear() 
        self.validation_step_outputs_acc2.clear() 

ip_config = {
        'project_name': 'hyperparam_opt_ip',
        'run_name': 'lstm_optimization_try1',

        'nn_arch': 'simple_fnn', # 'lstm', #'simple_fnn', 'transformer'

        # Parameters to Optimize
        'b_size': tune.choice([16, 32, 64, 128, 256]), #512
        'n_hlay': tune.choice(list(i for i in range(7))),
        'hdim': tune.choice(list(2**i for i in range(3, 11))),
        'lr': tune.loguniform(1e-6, 1e-2),
        'act_fn': tune.choice(['relu', 'tanh', 'sigmoid']),
        'loss_fn': tune.choice(['mse', 'mae']),#, 'cross_entropy']),
        'opt': tune.choice(['adam', 'sgd']),# 'ada', 'lbfgs', 'rmsprop']),

        'metric': 'val/loss',
        'mode': 'min',
        'logmetric1_name': 'x_accuracy',
        'logmetric2_name': 'xdot_accuracy',

        # Parameters to Optimize
        # 'b_size': 32,
        # 'n_hlay': 1,
        # 'hdim': 128,
        # 'lr': 0.0001,
        # 'act_fn': 'relu',
        # 'loss_fn': 'mse',
        # 'opt': 'adam',

        'n_inputs': 3, # [theta_dot, theta, tau] at t
        'n_outputs': 2, #[theta_dot, theta] at t+1

        # Other Configuration Parameters
        'accuracy_tolerance': 0.01, # This translates to about 1/2 a degree for inverted pendulum
        'num_workers': 6,
        'generate_new_data': False,
        'learn_mode': 'x',
        'dataset_size': 60000,
        'normalized_data': False,
        'dt': 0.01,
        'cpu_num': 8,
        'gpu_num': 1.0,
        
        # Optimization Tool Parameters
        'max_epochs': 4,
        'num_samples': 20,
        'path': '/home/daniel/research/catkin_ws/src/hyperparam_optimization/inverted_pendulum/',
        }

if __name__ == '__main__':
    seed_everything(42, workers=True)
    
    # Generate New Data --------------------------------------------------
    # try:
    #     InvertedPendulumDataset(ip_config, InvertedPendulum)
    #     InvertedPendulumDataset(ip_config, InvertedPendulum, validation=True)
    #     print("Inverted Pendulum Dataset class loaded...")
    # except Exception as e:
    #     print(f"Could not instantiate the Inverted Pendulum Dataset. Got the following error: {e}")

    # # Test the Lightning Module class --------------------------------------------------
    # try:
    #     lightning_module = InvertedPendulumLightningModule(ip_config)
    #     print("Inverted Pendulum Lightning Module loaded...")
    # except Exception as e:
    #     print(f"Could not instantiate the Inverted Pendulum Lightning Module. Got the following error: {e}")

    # # Test the training loop -------------------------------------------
    # try:
    #     train(ip_config, 
    #             InvertedPendulumLightningModule,
    #             notune=True)
    #     print("\nTraining Loop Successfully Tested...")
    # except Exception as e: 
    #     print(f"\nCould not complete training loop due to {e}")

    # Run the actual optimization
    optimize_system(ip_config, InvertedPendulumLightningModule)
