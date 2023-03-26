import torch
from torch.utils.data import DataLoader
from torch.profiler import profile, record_function, ProfilerActivity, schedule

import matplotlib.pyplot as plt

import logging
from datetime import datetime
from math import sqrt

from gelib import *
from args import *

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

# print(torch.cuda.memory_summary())


# This makes printing tensors more readable.
torch.set_printoptions(linewidth=1000, threshold=100000)

# logging.basicConfig(level = logging.INFO, format = "%(message)s")
logger = logging.getLogger('')



def main():
    # Initialize arguments and cuda
    args = init_args()
    args = init_file_paths(args)
    init_logger(args)
    device, dtype = init_cuda(args)
    
    # logger
    wandb_logger = WandbLogger(project="simple_cg_net", name = args.prefix) 

    # Instantiate datamodule class
    datamodule = DataModule(args, collate_fn)
    args, datasets, num_species, charge_scale, stats = datamodule.setup()
    
    # Instantiate the pytorch lightning model and datamodule class 
    model = Engine(args, num_species, charge_scale, stats, device, dtype)

    # Apply the covariance and permutation invariance tests.( THIS IS STILL MISSING)
    # cormorant_tests(model, dataloaders['train'], args, charge_scale=charge_scale)

    # Instantiate the training class
    trainer = pl.Trainer(max_epochs = args.num_epoch, 
                        accelerator = str(device), 
                        devices = 1,
                        logger = wandb_logger)
                        
    # log gradients and model topology
    wandb_logger.watch(model)
    
    # # Load from checkpoint file. If no checkpoint file exists, automatically does nothing.
    # trainer.load_checkpoint()

    # # Train model.
    trainer.fit(model, datamodule)

    # # Test predictions on best model and also last checkpointed model.
    trainer.test(model) 

    # # Plot training and validation loss 
    # plt.plot(trainer.train_mae_arr, label = "training")
    # plt.plot(trainer.val_mae_arr, label = "validation")
    # plt.xlabel("Epoch")
    # plt.ylabel("Mean Absolute Error ")
    # plt.legend()
    # plt.savefig("loss_curve_{}".format(args.prefix))

if __name__ == '__main__':
    main()




