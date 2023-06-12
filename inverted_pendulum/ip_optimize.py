from pytorch_lightning import seed_everything

import sys
sys.path.append('/home/daniel/research/catkin_ws/src/')
from hyperparam_optimization.BASE.optimize_learned_model import optimize_system, train
from hyperparam_optimization.BASE.DatasetBaseClass import DatasetBaseClass
from hyperparam_optimization.BASE.LightningModuleBaseClass import LightningModuleBaseClass

from ip_config import ip_config, test_ip_config
from InvertedPendulum import InvertedPendulum

class InvertedPendulumDataset(DatasetBaseClass):
    pass

class InvertedPendulumLightningModule(LightningModuleBaseClass):
    def __init__(self, config: dict):
        super().__init__(config)
        self.system = InvertedPendulum
        self.dataset = InvertedPendulumDataset


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
    #     lightning_module = InvertedPendulumLightningModule(test_ip_config)
    #     print("Inverted Pendulum Lightning Module loaded...")
    # except Exception as e:
    #     print(f"Could not instantiate the Inverted Pendulum Lightning Module. Got the following error: {e}")

    # Test the training loop -------------------------------------------
    # try:
    # train(test_ip_config, 
    #         InvertedPendulumLightningModule,
    #         notune=True)
    # print("\nTraining Loop Successfully Tested...")
    # except Exception as e: 
    #     print(f"\nCould not complete training loop due to {e}")

    # Run the actual optimization
    optimize_system(ip_config, InvertedPendulumLightningModule)
