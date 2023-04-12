import torch
from torch.utils.data import DataLoader
from torch.profiler import profile, record_function, ProfilerActivity, schedule

import matplotlib.pyplot as plt

import logging
from datetime import datetime
from math import sqrt

from args import *
from gelib import *

from network_radial_filters import *
from network_edge_update import *
from network_input_mpnn import *
from network_model import *

from utils_relative_positions import *
from utils_init import *

from lightning_engine import Engine
from lightning_datamodule import DataModule

from collate import collate_fn

import lightning.pytorch as pl
from lightning.pytorch.loggers import WandbLogger

from ray.tune.integration.pytorch_lightning import TuneReportCallback
from ray import tune, air
from ray.tune.schedulers import ASHAScheduler

# hyperparameters 
from param_config import param_config

# print(torch.cuda.memory_summary())
# import gc 
# gc.collect()
import torch
import os
# os.environ['CUDA_VISIBLE_DEVICES']="0"
# print(torch.cuda.get_device_name(torch.device('cuda:0')))


# This makes printing tensors more readable.
torch.set_printoptions(linewidth=1000, threshold=100000)

# logging.basicConfig(level = logging.INFO, format = "%(message)s")
logger = logging.getLogger('')

def main(param_config): 
    # Initialize arguments and cuda
    args = init_args()
    args = init_file_paths(args)
    init_logger(args)

     # Instantiate logger
    logger = WandbLogger(project="simple_cg_net", name = args.prefix) 

    # Tune or manually train? 
    if args.tune == True: 
        train_with_tune(param_config, args,logger) 
    elif args.tune == False: 
        train(param_config, args, logger)

def train_with_tune(param_config, args, logger): 
    # scheduler
    scheduler = ASHAScheduler(
        max_t=255,
        grace_period=1,
        reduction_factor=2)

    # What metrics to report to tune 
    tune_callback = _TuneReportCallback(["val_loss"], on = "validation_end")
   
    ## Tuning configurations ## 
    # Minimize which obj, use which scheduler / search alg on how many samples. 
    # Leaving the latter arguments as default for now. 
    tune_config = tune.TuneConfig(metric = "val_loss", 
                                mode = "min", 
                                scheduler = scheduler, 
                                trial_dirname_creator = create_trial_directory,
                                trial_name_creator = create_trial_name,
                                chdir_to_trial_dir=False)

    ## Runtime configurations ## 
    # leaving as default for now 
    run_config = air.RunConfig(local_dir = "ray_results") 

    # Wrap training function 
    tune_train = tune.with_parameters(train, args = args, logger = logger, tune_callback = tune_callback) 
    tune_train = tune.with_resources(tune_train,{"cpu":24,"gpu":3}) # Otherwise no resources will be allocated.

    # Run 
    print("Tuning!")
    reporter = tune.CLIReporter(
        # parameter_columns=["l1", "l2", "lr", "batch_size"],
        metric_columns=["val_loss","training_iteration"])
   
    #tune.run(tune_train, config = param_config, num_samples = 1, progress_reporter = reporter)
    # Instantiate Tuner 
    tuner = tune.Tuner(tune_train,
                        tune_config = tune_config, 
                        run_config = run_config, 
                        param_space = param_config
                        )

    # Tune 
    print("Defined Tuner")
    results = tuner.fit()
    
    print("Best hyperparameters found were: ", results.get_best_result().config)

def create_trial_directory(Trial):
    return Trial.trial_id 

def create_trial_name(Trial):
    return Trial.trial_id

# taken from https://github.com/ray-project/ray/issues/33426
class _TuneReportCallback(TuneReportCallback, pl.Callback):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)


def train(param_config, args, logger, tune_callback = None):
    """ 
    Instantiates, trains and tests model on dataset. 
    
    """ 
    print("hello")
    device, dtype = init_cuda(args)
    # replace 
    for hparam in param_config:
        setattr(args, str(hparam), param_config[hparam])
    
    # Instantiate datamodule class
    datamodule = DataModule(args, collate_fn)
    args, datasets, num_species, charge_scale, stats = datamodule.setup()

    # Instantiate the pytorch lightning model and datamodule class 
    model = Engine(args, num_species, charge_scale, stats, device, dtype)

    # Apply the covariance and permutation invariance tests.( THIS IS STILL MISSING)
    # cormorant_tests(model, dataloaders['train'], args, charge_scale=charge_scale)

    # Instantiate the training class
    callbacks = [] # to avoid defining trainer twice 
    if tune_callback != None: 
        callbacks.append(tune_callback)
    trainer = pl.Trainer(max_epochs = args.num_epoch, 
                        logger = logger,
                        accelerator = "gpu",
                        devices = 1 ,
                        callbacks = callbacks,
                        profiler = 'advanced')
   
    # log gradients and model topology
    logger.watch(model)
    
    # # Load from checkpoint file. If no checkpoint file exists, automatically does nothing.
    # trainer.load_checkpoint()

    # # Train model.
    print("Fitting!")
    trainer.fit(model, datamodule)

    # # Test predictions on best model and also last checkpointed model.
    trainer.test(model,datamodule) 


if __name__ == '__main__':
    main(param_config)




