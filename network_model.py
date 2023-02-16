import torch
from torch import nn
from network_CG import *
from network_input_mpnn import *
from network_output_mlp import *
from utils_relative_positions import *

from datetime import datetime
import logging
logger = logging.getLogger(__name__)
torch.autograd.set_detect_anomaly(True)



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

        # cg args
        self.max_l = max_l[0]
        self.num_channels = num_channels[0]
        self.num_cg_layers = num_cg_layers

        # output args
        self.num_scalars = self.get_num_scalars()

        # layers
        self.input_layers = InputMPNN(max_l, num_input_channels_in, num_input_channels_out, num_mpnn_layers,
                                      soft_cut_rad, soft_cut_width, hard_cut_rad,
                                      activation = activation, device = self.device, dtype = self.dtype)
        self.cg_layers = CGLayers(self.num_cg_layers, self.num_channels,
                                    self.max_l, hard_cut_rad, device = self.device, dtype = self.dtype)
        self.output_layers = OutputMLP(self.num_scalars, activation = activation,
                                        device = self.device,dtype = self.dtype)

    def forward(self,data):
        # prepare input -- features, edges and corresponding masks .
        atom_scalars, atom_mask, edge_scalars, edge_mask, atom_positions = self.prepare_input(data)

        # for debugging...
        # atom_scalars = atom_scalars[:8,...]
        # atom_mask = atom_mask[:8,...]
        # edge_scalars = edge_scalars[:8,...]
        # edge_mask = edge_mask[:8,...]
        # atom_positions = atom_positions[:8,...]
        # print('just using the first 8 molecules for now.')

        # input message passing network
        input_t = datetime.now()
        norms = get_norm(atom_positions, atom_positions)
        atom_vec_in = self.input_layers(atom_scalars, atom_mask, edge_scalars, edge_mask, norms)
        

        # cg network
        cg_t = datetime.now() 
        rel_pos = get_rel_pos(atom_positions,atom_positions)
        atom_scalars_cg = self.cg_layers(atom_vec_in, rel_pos, norms)
        atom_scalars = atom_scalars_cg
     
        # output network
        output_t = datetime.now()
        prediction = self.output_layers(atom_scalars, atom_mask)
        final_t = datetime.now() 

        # log time elapsed
        input_dt = (cg_t-input_t).total_seconds()
        cg_dt = (output_t-cg_t).total_seconds()
        output_dt = (final_t-output_t).total_seconds()
        logstring = "input, cg, output networks took {},{},{} seconds".format(input_dt,cg_dt, output_dt)
        logging.info(logstring)

        return prediction

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

    def get_num_scalars(self):
        """
        Calculates number of scalars per atom in each molecule.
        """
        return self.num_cg_layers*(2*self.num_channels + (self.max_l+1)*self.num_channels**2)
