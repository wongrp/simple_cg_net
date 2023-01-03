import torch
from torch import nn
from network_CG import *
from network_input_mpnn import *

from utils_relative_positions import *


class FullModel(nn.Module):
    def __init__(self,max_l, max_sh, num_cg_layers, num_channels, num_species,
                 cutoff_type, hard_cut_rad, soft_cut_rad, soft_cut_width,
                 weight_init, level_gain, charge_power, basis_set,
                 charge_scale, gaussian_mask,
                 top, input, num_mpnn_layers, activation='leakyrelu',
                 device=None, dtype=None):
        super().__init__()


        # args #
        # input args
        self.charge_power = charge_power
        self.charge_scale = charge_scale
        self.device = device
        self.dtype = dtype

        # input mpnn args
        self.num_species = num_species
        num_input_channels_in = self.num_species * (self.charge_power + 1)
        num_input_channels_out = num_channels[0]
        print('number of channels in input vector is ', num_channels)
        #num_input_channels_out = num_channels[0]
        # cg args
        self.num_atoms = 29

        # output args
        # self.num_scalars
        # self.num_mixed
        # self.activation
        # self.device
        # self.dtype


        # layers
        self.input_layers = InputMPNN(max_l, num_input_channels_in, num_input_channels_out, num_mpnn_layers,
                                      soft_cut_rad, soft_cut_width, hard_cut_rad,
                                      activation = activation, device = self.device, dtype = self.dtype)

        self.cg_layers = CGLayers(num_cg_layers, self.num_atoms, num_channels, max_l, hard_cut_rad)
        # self.output_layers = OutputMLP(num_scalars, num_mixed, activation, device,dtype)

    def forward(self,data):
        # prepare input -- features, edges and corresponding masks .
        atom_scalars, atom_mask, edge_scalars, edge_mask, atom_positions = self.prepare_input(data)
        atom_scalars = atom_scalars[:10,...]
        atom_mask = atom_mask[:10,...]
        edge_scalars = edge_scalars[:10,...]
        edge_mask = edge_mask[:10,...]
        atom_positions = atom_positions[:10,...]
        print('just using the first 10 molecules for now.')
        # input message passing network
        norms = get_norm(atom_positions, atom_positions)
        atom_vec_in = self.input_layers(atom_scalars, atom_mask, edge_scalars, edge_mask, norms)

        # cg network
        rel_pos = get_rel_pos(atom_positions,atom_positions)
        atom_scalars_cg = self.cg_layers(atom_vec_in, rel_pos, norms)

        # output network

        return

    def prepare_input(self, data):
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
        charge_power, charge_scale, device, dtype = self.charge_power, self.charge_scale, self.device, self.dtype

        atom_positions = data['positions'].to(device, dtype)
        one_hot = data['one_hot'].to(device, dtype)
        charges = data['charges'].to(device, dtype)

        # should write a masking function just for qm9
        if set(['atom_mask','edge_mask']).issubset(data.keys()):
            atom_mask = data['atom_mask'].to(device)
            edge_mask = data['edge_mask'].to(device)
        else:
            atom_mask = data['charges'] > 0
            edge_mask = atom_mask.unsqueeze(1) * atom_mask.unsqueeze(2)


        charge_tensor = (charges.unsqueeze(-1)/charge_scale).pow(torch.arange(charge_power+1., device=device, dtype=dtype))
        charge_tensor = charge_tensor.view(charges.shape + (1, charge_power+1))
        atom_scalars = (one_hot.unsqueeze(-1) * charge_tensor).view(charges.shape[:2] + (-1,))

        edge_scalars = torch.tensor([])

        return atom_scalars, atom_mask, edge_scalars, edge_mask, atom_positions
