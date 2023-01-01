import torch
from torch import nn
from network_CG import *
from network_mlp import *

from utils_relative_positions import *


class FullModel(nn.Module):
    def __init__(self,):
        super().__init__(maxl, max_sh, num_cg_levels, num_channels, num_species,
                     cutoff_type, hard_cut_rad, soft_cut_rad, soft_cut_width,
                     weight_init, level_gain, charge_power, basis_set,
                     charge_scale, gaussian_mask,
                     top, input, num_mpnn_layers, activation='leakyrelu',
                     device=None, dtype=None)


        # args #
        # cg args
        self.num_cg_layer = num_cg_levels
        self.num_atoms =
        self.num_channels =
        self.max_l =
        self.connectivity =
        self.rel_positions =

        # output args
        self.num_scalars
        self.num_mixed
        self.activation
        self.device
        self.dtype

        # layers
        self.input_layers = InputMPNN(num_input_channels_in, num_input_channels_out, num_mpnn_layers,
                                      soft_cut_rad, soft_cut_width, hard_cut_rad,
                                      activation = activation, device = self.device, dtype = self.dtype)

        self.cg_layers = CG_layers(num_cg_layer,connectivity, num_atoms, num_input_channels_out, max_l)
        # self.output_layers = OutputMLP(num_scalars, num_mixed, activation, device,dtype)

    def forward(data):
        # prepare input -- features, edges and corresponding masks .
        atom_scalars, atom_mask, edge_scalars, edge_mask, atom_positions = prepare_input(data)

        # input message passing network
        norms = get_norm(atom_positions)
        atom_vec_in = self.input_layers(atom_scalars, atom_mask, edge_scalars, edge_mask, norms)

        # cg network
        rel_pos = rel_pos(atom_positions,atom_positions)
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

        atom_mask = data['atom_mask'].to(device)
        edge_mask = data['edge_mask'].to(device)

        charge_tensor = (charges.unsqueeze(-1)/charge_scale).pow(torch.arange(charge_power+1., device=device, dtype=dtype))
        charge_tensor = charge_tensor.view(charges.shape + (1, charge_power+1))
        atom_scalars = (one_hot.unsqueeze(-1) * charge_tensor).view(charges.shape[:2] + (-1,))

        edge_scalars = torch.tensor([])

        return atom_scalars, atom_mask, edge_scalars, edge_mask, atom_positions
