import torch
from torch.utils.data import DataLoader

import logging
from datetime import datetime
from math import sqrt

from args import *
from utils_initialize_dataset import *
from collate import collate_fn


from network_radial_filters import *
from utils_relative_positions import *
from network_edge_update import *
from network_input_mpnn import *

from gelib import *

# This makes printing tensors more readable.
torch.set_printoptions(linewidth=1000, threshold=100000)

logging.basicConfig(level = logging.INFO, format = "%(message)s")
logger = logging.getLogger('')


def main():
    # Initialize arguments
    args = init_args()

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


    # test

    print(dataset.data['one_hot'].size())
    print(dataset.data['charges'].size())
    print(dataloaders['train'].dataset)


    def prepare_input(data):
        """
        Extracts input from data class

        Parameters
        ----------
        data : ?????
            Information on the state of the system.

        Returns
        -------
        atom_scalars : :obj:`torch.Tensor`
            Tensor of scalars for each atom.
        atom_mask : :obj:`torch.Tensor`
            Mask used for batching data.
        atom_positions: :obj:`torch.Tensor`
            Positions of the atoms
        edge_mask: :obj:`torch.Tensor`
            Mask used for batching data.
        """
        charge_power, charge_scale, device, dtype = 2, 1, 'cpu', torch.float

        atom_positions = data['positions'].to(device, dtype)
        one_hot = data['one_hot'].to(device, dtype)
        charges = data['charges'].to(device, dtype)

        atom_mask = data['atom_mask'].to(device)
        edge_mask = data['edge_mask'].to(device)

        charge_tensor = (charges.unsqueeze(-1)/charge_scale).pow(torch.arange(charge_power+1., device=device, dtype=dtype))
        charge_tensor = charge_tensor.view(charges.shape + (1, charge_power+1))
        atom_scalars = (one_hot.unsqueeze(-1) * charge_tensor).view(charges.shape[:2] + (-1,))
        print('one_hot: ', one_hot.size())
        edge_scalars = torch.tensor([])

        return atom_scalars, atom_mask, edge_scalars, edge_mask, atom_positions



    charge_power = 2
    num_channels = 5*(charge_power+1) # number of edge channels has to match feature channels = num_species * charge tensor length
    rad_filt = RadPolyTrig(2,(3,3),num_channels,mix = 'real') # cannot match channels if not mix.

    data = dataloaders['val'].dataset.data
    atom_mask = data['charges'] > 0
    edge_mask = atom_mask.unsqueeze(1) * atom_mask.unsqueeze(2)

    data['atom_mask'] = atom_mask
    data['edge_mask'] = edge_mask

    atom_scalars, atom_mask,edge_scalars, edge_mask, atom_positions = prepare_input(data)
    print(edge_mask.size())

    norms = get_norm(atom_positions, atom_positions)
    rad = rad_filt(norms,edge_mask)
    rad = rad[0][..., 0].unsqueeze(-1)
    # this inputs into the message passing step
    # without mixing, num_channels_out for rad functions becomes the number of basis functions.
    edge_update = MaskLevel(rad.size(3), None, 2e-3, 1, ['soft'])
    edge = edge_update(rad,edge_mask,norms)
    edge = edge.squeeze(-1)

    # # Now pass messages using matrix multiplication with the edge features
    # # Einsum b: batch, a: atom, c: channel, x: to be summed over
    features_mp = torch.einsum('baxc,bxc->bac', edge, atom_scalars)
    features_mp = torch.cat([atom_scalars,features_mp],-1)

    # try instantiating an input mlp object
    # channels in: size of the entire processed feature vector (5 atoms * charge tensor length )
    # channels out: arbitrary if mixing
    input_func = InputMPNN(num_channels, 12,num_layers = 3, soft_cut_rad=2e-3, soft_cut_width=1, hard_cut_rad=None, cutoff_type=['soft'])
    features = input_func(atom_scalars, atom_mask, edge_scalars, edge_mask, norms)

    print(features.size())

    # let's put into gelib
    batch_size = features.size(0)
    num_atoms  = features.size(1)
    num_channels = features.size(2)
    max_l = 0
    features_size = num_channels*np.ones(max_l+1).astype(int)
    features_so3 = SO3vecArr.zeros(batch_size,[num_atoms],features_size)
    features_so3.parts[0] = features

    print(features_so3)
    # # # Initialize model
    # model = CormorantQM9(args.maxl, args.max_sh, args.num_cg_levels, args.num_channels, num_species,
    #                     args.cutoff_type, args.hard_cut_rad, args.soft_cut_rad, args.soft_cut_width,
    #                     args.weight_init, args.level_gain, args.charge_power, args.basis_set,
    #                     charge_scale, args.gaussian_mask,
    #                     args.top, args.input, args.num_mpnn_levels,
    #                     device=device, dtype=dtype)
    #
    # # Initialize the scheduler and optimizer
    # optimizer = init_optimizer(args, model)
    # scheduler, restart_epochs = init_scheduler(args, optimizer)
    #
    # # Define a loss function. Just use L2 loss for now.
    # loss_fn = torch.nn.functional.mse_loss
    #
    # # Apply the covariance and permutation invariance tests.
    # cormorant_tests(model, dataloaders['train'], args, charge_scale=charge_scale)
    #
    # # Instantiate the training class
    # trainer = Engine(args, dataloaders, model, loss_fn, optimizer, scheduler, restart_epochs, device, dtype)
    #
    # # Load from checkpoint file. If no checkpoint file exists, automatically does nothing.
    # trainer.load_checkpoint()
    #
    # # Train model.
    # trainer.train()
    #
    # # Test predictions on best model and also last checkpointed model.
    # trainer.evaluate()

if __name__ == '__main__':
    main()
