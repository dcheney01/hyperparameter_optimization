from pytorch_lightning import Trainer
import pytorch_lightning as pl
from lightning.pytorch.loggers import WandbLogger, TensorBoardLogger

from ray import air, tune
from ray.tune import CLIReporter
from ray.tune.schedulers import ASHAScheduler
from ray.tune.integration.pytorch_lightning import TuneReportCallback
from ray.tune.search.optuna import OptunaSearch
from ray.air.integrations.wandb import WandbLoggerCallback, setup_wandb

import os, sys, json
import wandb
os.environ['WANDB_SILENT']="true"

sys.path.append('/home/daniel/research/catkin_ws/src/')
from hyperparam_optimization.BASE.LightningModuleBaseClass import LightningModuleBaseClass

# from hyperparam_optimization.inverted_pendulum.LightningModuleBaseClass import LightningModuleBaseClass

# from LightningModuleBaseClass import LightningModuleBaseClass


""" Currently have to inherit these callbacks due to known issue with ray tune call backs
            https://github.com/ray-project/ray/issues/33426#issuecomment-1477464856
"""
class _WandbLoggerCallback(WandbLoggerCallback, pl.Callback):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)


class _TuneReportCallback(TuneReportCallback, pl.Callback):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)


def train(config:dict, 
          lighting_module_given:LightningModuleBaseClass,
          notune=False):
    if notune:
        lightning_module = lighting_module_given(config)
        trainer = Trainer(deterministic=True,
                            max_epochs=config['max_epochs'], 
                            enable_progress_bar=True,
                            logger=TensorBoardLogger(save_dir=config['path'])
                            )
        trainer.fit(lightning_module)
        trainer.validate(lightning_module, verbose=True)
    else:
        wandb = setup_wandb(config)
        # wandb.finish()
        wandb_logger = WandbLogger(project=config['project_name'], log_model='all', 
                                   save_dir=f'{os.getcwd()}/')
        # run = wandb.init()
        # print(f'################################{run.name}##############################')

        lightning_module = lighting_module_given(config)

        trainer = Trainer(deterministic=True,
                          max_epochs=config['max_epochs'], 
                          enable_progress_bar=False,
                          logger=wandb_logger,
                          callbacks=[_TuneReportCallback()]
                        )
        trainer.fit(lightning_module)
        # run.finish()


def optimize_system(config:dict,           
                    lighting_module_given:LightningModuleBaseClass):
    scheduler = ASHAScheduler(
                            max_t=config['max_epochs'],
                            grace_period=1,
                            reduction_factor=4)
    reporter = CLIReporter(
                        parameter_columns=["n_hlay", "hdim", "b_size", 'lr', 'act_fn', 'loss_fn', 'opt'],
                        metric_columns=["val/loss", "val/x_accuracy", "training_iteration"])

    train_fn_with_parameters = tune.with_parameters(train, lighting_module_given=lighting_module_given)
    
    resources_per_trial = {"cpu": config['cpu_num'], "gpu": config['gpu_num']}

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
        run_config=air.RunConfig(name=config['run_name'], 
                                 progress_reporter=reporter,
                                 callbacks=[
                                    _WandbLoggerCallback(project=config['project_name'], 
                                                        log_config=True)
                                    ],
                                 local_dir=config['path']+'run_logs/'
        ),
        param_space=config,
    )
    results = tuner.fit()
    print("Best hyperparameters found were: ", json.dumps(results.get_best_result().config, indent=4))