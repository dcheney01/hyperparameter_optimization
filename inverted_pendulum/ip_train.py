from pytorch_lightning import Trainer, seed_everything
import pytorch_lightning as pl
from lightning.pytorch.loggers import WandbLogger

from ray import air, tune
from ray.tune import CLIReporter
from ray.tune.schedulers import ASHAScheduler
from ray.tune.integration.pytorch_lightning import TuneReportCallback
from ray.tune.search.optuna import OptunaSearch
from ray.air.integrations.wandb import WandbLoggerCallback

import wandb
os.environ['WANDB_SILENT']="true"

import os, sys, json

sys.path.append('/home/daniel/research/catkin_ws/src/')
from InvertedPendulumLightning import InvertedPendulumLightning

class _WandbLoggerCallback(WandbLoggerCallback, pl.Callback):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

class _TuneReportCallback(TuneReportCallback, pl.Callback):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

def train_ip(config, notune=False):
    if notune:
        ip_lightning = InvertedPendulumLightning(config)
        trainer = Trainer(deterministic=True,
                            max_epochs=config['max_epochs'], 
                            enable_progress_bar=True
                            )
        trainer.fit(ip_lightning)
        trainer.validate(ip_lightning, verbose=True)
    else:
        wandb.finish()
        run = wandb.init(reinit=True)

        wandb_logger = WandbLogger(project=config['project_name'], log_model='all', 
                                   save_dir=f'{os.getcwd()}/')

        ip_lightning = InvertedPendulumLightning(config)

        trainer = Trainer(deterministic=True,
                          max_epochs=config['max_epochs'], 
                          enable_progress_bar=False,
                          logger=wandb_logger,
                          callbacks=[_TuneReportCallback(on="validation_end")]
                        )
        trainer.fit(ip_lightning)
        run.finish()


def main(config):
    scheduler = ASHAScheduler(
                            max_t=config['max_epochs'],
                            grace_period=1,
                            reduction_factor=4)
    reporter = CLIReporter(
                        parameter_columns=["n_hlay", "hdim", "b_size", 'lr', 'act_fn', 'loss_fn', 'opt'],
                        metric_columns=["val/loss", "val/x_accuracy", "training_iteration"])

    train_fn_with_parameters = tune.with_parameters(train_ip)
    
    resources_per_trial = {"cpu": 5, "gpu": .65}

    optuna_search = OptunaSearch()
    tuner = tune.Tuner(
        tune.with_resources(
            train_fn_with_parameters,
            resources_per_trial
        ),
        tune_config=tune.TuneConfig(
            search_alg=optuna_search,
            scheduler=scheduler,
            metric="val/x_accuracy", 
            mode="max",
            num_samples=config['num_samples'],
        ),
        run_config=air.RunConfig(
            # name=config["run_name"],
            progress_reporter=reporter,
            callbacks=[
                _WandbLoggerCallback(project=config['project_name'], log_config=True)
            ],
            local_dir=config['path']+'run_logs/'
        ),
        param_space=config,
    )
    results = tuner.fit()
    print("Best hyperparameters found were: ", json.dumps(results.get_best_result().config, indent=4))


if __name__ == '__main__':
    seed_everything(42, workers=True)

    config = {
        'run_name': 'data-points-run',
        'project_name': 'hyperparam_opt_ip',

        # Parameters to Optimize
        # 'b_size': 32,
        # 'n_hlay': 3,
        # 'hdim': 20,
        # 'lr': 0.001,
        # 'act_fn': 'relu',
        # 'loss_fn': 'mse',
        # 'opt': 'adam',

        'b_size': tune.choice([16, 32, 64, 128, 256, 512]),
        'n_hlay': tune.choice(list(i for i in range(11))),
        'hdim': tune.choice(2**i for i in range(3, 11)),
        'lr': tune.loguniform(1e-5, 1e-1),
        'act_fn': tune.choice(['relu', 'tanh']),#, 'sigmoid']),#, 'softmax']),
        'loss_fn': tune.choice(['mse', 'mae']),#, 'cross_entropy']),
        'opt': tune.choice(['adam', 'sgd']),# 'ada', 'lbfgs', 'rmsprop']),

        # Parameters that I eventually want to optimize
        'nn_arch': 'simple_fnn',

        # Other Configuration Parameters
        # 'accuracy_tolerance': 0.1, # This translates to about 5 degrees
        'accuracy_tolerance': 0.01, # This translates to about 1/2 a degree
        'calculates_xdot':False,
        'num_workers': 6,

        # Optimization Tool Parameters
        'max_epochs': 500,
        'num_samples': 250,
        'path': '/home/daniel/research/catkin_ws/src/hyperparam_optimization/inverted_pendulum/',
        }
# 
    main(config)
    # train_ip(config, notune=True)