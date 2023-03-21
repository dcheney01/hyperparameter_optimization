from lightning.pytorch import Trainer, seed_everything
from torch import nn

from InvertedPendulumLightning import InvertedPendulumLightning

from pytorch_lightning.loggers import TensorBoardLogger
from ray import air, tune
from ray.tune import CLIReporter
from ray.tune.schedulers import ASHAScheduler, PopulationBasedTraining
from ray.tune.integration.pytorch_lightning import TuneReportCallback, \

import os

def train_ip(config, notune=False):
    if notune:
        ip_lightning = InvertedPendulumLightning(config)
        trainer = Trainer(accelerator=config['accelerator'], 
                            devices=config['devices'],
                            deterministic=True,
                            max_epochs=config['max_epochs'], 
                            )
        trainer.fit(ip_lightning)
        trainer.validate(ip_lightning, verbose=True)
    else:
        ip_lightning = InvertedPendulumLightning(config)

        trainer = Trainer(accelerator=config['accelerator'], 
                        devices=config['devices'],
                        deterministic=True,
                        max_epochs=config['max_epochs'], 
                        logger=TensorBoardLogger(save_dir=os.getcwd(), name="", version="."),
                        callbacks=[
                                TuneReportCallback(
                                    metrics={
                                        "loss": "val_loss",
                                        "mean_accuracy": "val_accuracy"
                                    }
                                    )
                        ]
                        )

        trainer.fit(ip_lightning)

        # Output the final accuracy of the model
        # trainer.validate(ip_lightning, val_loader, verbose=True)


def main(config):
    scheduler = ASHAScheduler(
                            max_t=config['max_epochs'],
                            grace_period=1,
                            reduction_factor=2)
    reporter = CLIReporter(
                        parameter_columns=["num_hidden_layers", "hdim", "batch_size"],
                        metric_columns=["val_loss", "val_accuracy", "training_iteration"])

    train_fn_with_parameters = tune.with_parameters(train_ip)
    
    resources_per_trial = {"cpu": 1, "gpu": 0.25}
    tuner = tune.Tuner(
        tune.with_resources(
            train_fn_with_parameters,
            resources_per_trial
        ),
        tune_config=tune.TuneConfig(
            metric="loss",
            mode="min",
            scheduler=scheduler,
            num_samples=config['num_samples'],
        ),
        run_config=air.RunConfig(
            name="tune_ip_firsttry",
            progress_reporter=reporter,
        ),
        param_space=config,
    )
    results = tuner.fit()
    print("Best hyperparameters found were: ", results.get_best_result().config)


if __name__ == '__main__':
    seed_everything(42, workers=True)

    config = {
        'max_epochs': 3,
        'training_size': 10000,
        'validation_size': 1000,
        # 'batch_size': 64,
        # 'num_hidden_layers': 2,
        # 'hdim': 20, 
        'batch_size': tune.choice([32, 64, 128, 256]),
        'num_hidden_layers': tune.choice(list(i for i in range(11))),
        'hdim': tune.choice(2**i for i in range(10)),
        'num_inputs': 3, 
        'num_outputs': 2,
        'activation_fn': nn.ReLU(),
        'lr': 0.001,
        'loss_fn': nn.MSELoss,
        'accuracy_tolerance': 0.1,
        'num_samples': 1,
        'devices': 1,
        'accelerator': 'gpu',
        'generate_new_data': False,
        'path': '/home/daniel/research/catkin_ws/src/hyperparam_optimization/inverted_pendulum/',
        }

    main(config)
    # train_ip(config, notune=True)