import torch
from torch.utils.data import DataLoader

import logging
from datetime import datetime
from math import sqrt

from gelib import *
from args import *

from network_radial_filters import *
from network_edge_update import *
from network_input_mpnn import *
from network_model import *

from utils_initialize_dataset import *
from utils_relative_positions import *
from utils_init import *

from engine import Engine

from collate import collate_fn



# This makes printing tensors more readable.
torch.set_printoptions(linewidth=1000, threshold=100000)

logging.basicConfig(level = logging.INFO, format = "%(message)s")
logger = logging.getLogger('')


def main():
    # Initialize arguments and cuda
    args = init_args()
    device, dtype = init_cuda(args)

    # Initialize dataloader.
    # datasets is a ProcessedDataset object.
    args, datasets, num_species, charge_scale = initialize_datasets(args, args.datadir, 'qm9', subtract_thermo=args.subtract_thermo,
                                                                    force_download=args.force_download
                                                                    )

    qm9_to_eV = {'U0': 27.2114, 'U': 27.2114, 'G': 27.2114, 'H': 27.2114, 'zpve': 27211.4, 'gap': 27.2114, 'homo': 27.2114, 'lumo': 27.2114}

    for dataset in datasets.values():
        dataset.convert_units(qm9_to_eV)

    # Construct PyTorch dataloaders from datasets
    # dataloader is a dictionary. dataloader['train'].dataset is a ProcessedDataset object.
    dataloaders = {split: DataLoader(dataset,
                                     batch_size=args.batch_size,
                                     shuffle=args.shuffle if (split == 'train') else False,
                                     num_workers=args.num_workers,
                                     collate_fn=collate_fn)
                         for split, dataset in datasets.items()}



    # # Initialize model
    model = FullModel(args.max_l, args.max_sh, args.num_cg_layers, args.num_channels, num_species,
                 args.cutoff_type, args.hard_cut_rad, args.soft_cut_rad, args.soft_cut_width,
                 args.weight_init, args.level_gain, args.charge_power, args.basis_set,
                 charge_scale, args.gaussian_mask,
                 args.top, args.input, args.num_mpnn_layers,
                 device=device, dtype=dtype)
    # dataloaders['train'].dataset.data.keys()
    # prediction = model(dataloaders['train'].dataset.data)

    # Initialize the scheduler and optimizer
    optimizer = init_optimizer(args, model)
    scheduler, restart_epochs = init_scheduler(args, optimizer)

    # Define a loss function. Just use L2 loss for now.
    loss_fn = torch.nn.functional.mse_loss

    # Apply the covariance and permutation invariance tests.
    # cormorant_tests(model, dataloaders['train'], args, charge_scale=charge_scale)

    # Instantiate the training class
    trainer = Engine(args, dataloaders, model, loss_fn, optimizer, scheduler, restart_epochs, device, dtype)

    # Load from checkpoint file. If no checkpoint file exists, automatically does nothing.
    trainer.load_checkpoint()

    # Train model.
    trainer.train()

    # Test predictions on best model and also last checkpointed model.
    trainer.evaluate()

if __name__ == '__main__':
    main()
