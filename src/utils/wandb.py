import argparse
import wandb
import numpy as np


def init_wandb(parameters):
    wandb.init(project="", entity="", config=parameters)
    wandb.config.update(parameters)
    parameters = argparse.Namespace(**wandb.config)
    parameters.wandb_run_name = wandb.run.name
    parameters.model_subdir = wandb.run.sweep_id
    return parameters


def report_step_mean(opt, time_list):
    if opt.use_wandb:
        wandb.log({'Avg step time': np.mean(time_list)})
        wandb.log({'Std step time': np.std(time_list)})
