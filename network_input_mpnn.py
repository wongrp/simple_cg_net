
import numpy as np

import torch
import torch.nn as nn

from gelib import *
from network_output_mlp import *
from network_edge_update import *
from network_radial_filters import *



# from cormorant.nn.generic_levels import BasicMLP
# from cormorant.nn.position_levels import RadPolyTrig
# from cormorant.nn.mask_levels import MaskLevel
#
# from cormorant.so3_lib import SO3Tau, SO3Vec


############# Input to network #############

class InputLinear(nn.Module):
    """
    Module to create rotationally invariant atom feature vectors
    at the input level.

    This module applies a simple linear mixing matrix to a one-hot of atom
    embeddings based upon the number of atomic types.

    Parameters
    ----------
    channels_in : :class:`int`
        Number of input features before mixing (i.e., a one-hot atom-type embedding).
    channels_out : :class:`int`
        Number of output features after mixing.
    bias : :class:`bool`, optional
        Include a bias term in the linear mixing level.
    device : :class:`torch.device`, optional
        Device to instantite the module to.
    dtype : :class:`torch.dtype`, optional
        Data type to instantite the module to.
    """
    def __init__(self, channels_in, channels_out, bias=True,
                 device=torch.device('cpu'), dtype=torch.float):
        super(InputLinear, self).__init__()

        self.channels_in = channels_in
        self.channels_out = channels_out

        self.lin = nn.Linear(channels_in, 2*channels_out, bias=bias)
        self.lin.to(device=device, dtype=dtype)

        self.zero = torch.tensor(0, dtype=dtype, device=device)

    def forward(self, atom_features, atom_mask, ignore, edge_mask, norms):
        """
        Forward pass for :class:`InputLinear` layer.

        Parameters
        ----------
        atom_features : :class:`torch.Tensor`
            Input atom features, i.e., a one-hot embedding of the atom type,
            atom charge, and any other related inputs.
        atom_mask : :class:`torch.Tensor`
            Mask used to account for padded atoms for unequal batch sizes.
        edge_features : :class:`torch.Tensor`
            Unused. Included only for pedagogical purposes.
        edge_mask : :class:`torch.Tensor`
            Unused. Included only for pedagogical purposes.
        norms : :class:`torch.Tensor`
            Unused. Included only for pedagogical purposes.

        Returns
        -------
        :class:`SO3Vec`
            Processed atom features to be used as input to Clebsch-Gordan layers
            as part of Cormorant.
        """
        atom_mask = atom_mask.unsqueeze(-1)

        out = torch.where(atom_mask, self.lin(atom_features), self.zero)
        out = out.view(atom_features.shape[0:2] + (self.channels_out, 1, 2))

        return out

    @property
    def tau(self):
        return self.channels_out



class InputMPNN(nn.Module):
    """
    Module to create rotationally invariant atom feature vectors
    at the input level.

    This module applies creates a scalar

    Parameters
    ----------
    channels_in : :class:`int`
        Number of input features before mixing (i.e., a one-hot atom-type embedding).
    channels_out : :class:`int`
        Number of output features after mixing.
    num_layers : :class:`int`
        Number of message passing layers.
    soft_cut_rad : :class:`float`
        Radius of the soft cutoff used in the radial position functions.
    soft_cut_width : :class:`float`
        Radius of the soft cutoff used in the radial position functions.
    hard_cut_rad : :class:`float`
        Radius of the soft cutoff used in the radial position functions.
    bias : :class:`bool`, optional
        Include a bias term in the linear mixing level.
    device : :class:`torch.device`, optional
        Device to instantite the module to.
    dtype : :class:`torch.dtype`, optional
        Data type to instantite the module to.
    """
    def __init__(self, max_l, channels_in, channels_out, num_layers=1,
                 soft_cut_rad=None, soft_cut_width=None, hard_cut_rad=None, cutoff_type=['learn'],
                 channels_mlp=-1, num_hidden=1, layer_width=256,
                 activation='leakyrelu', basis_set=(3, 3),
                 device=torch.device('cpu'), dtype=torch.float):
        super(InputMPNN, self).__init__()

        self.soft_cut_rad = soft_cut_rad
        self.soft_cut_width = soft_cut_width
        self.hard_cut_rad = hard_cut_rad
        self.max_l = max_l

        if channels_mlp < 0:
            channels_mlp = max(channels_in, channels_out)

        # List of channels at each level. The factor of two accounts for
        # the fact that the passed messages are concatenated with the input states.
        channels_lvls = [channels_in] + [channels_mlp]*(num_layers-1) + [2*channels_out]

        self.channels_in = channels_in
        self.channels_mlp = channels_mlp
        self.channels_out = channels_out

        # Set up MLPs
        self.mlps = nn.ModuleList()
        self.masks = nn.ModuleList()
        self.rad_filts = nn.ModuleList()

        for chan_in, chan_out in zip(channels_lvls[:-1], channels_lvls[1:]):
            rad_filt = RadPolyTrig(0, basis_set, chan_in, mix='real', device=device, dtype=dtype)
            mask = MaskLevel(1, hard_cut_rad, soft_cut_rad, soft_cut_width, ['soft', 'hard'], device=device, dtype=dtype)
            mlp = BasicMLP(2*chan_in, chan_out, num_hidden=num_hidden, layer_width=layer_width, device=device, dtype=dtype)

            self.mlps.append(mlp)
            self.masks.append(mask)
            self.rad_filts.append(rad_filt)

        self.dtype = dtype
        self.device = device

    def forward(self, features, atom_mask, edge_features, edge_mask, norms):
        """
        Forward pass for :class:`InputMPNN` layer.

        Parameters
        ----------
        features : :class:`torch.Tensor`
            Input atom features, i.e., a one-hot embedding of the atom type,
            atom charge, and any other related inputs.
        atom_mask : :class:`torch.Tensor`
            Mask used to account for padded atoms for unequal batch sizes.
        edge_features : :class:`torch.Tensor`
            Unused. Included only for pedagogical purposes.
        edge_mask : :class:`torch.Tensor`
            Mask used to account for padded edges for unequal batch sizes.
        norms : :class:`torch.Tensor`
            Matrix of relative distances between pairs of atoms.

        Returns
        -------
        :class:`SO3Vec`
            Processed atom features to be used as input to Clebsch-Gordan layers
            as part of Cormorant.
        """
        # Unsqueeze the atom mask to match the appropriate dimensions later
        atom_mask = atom_mask.unsqueeze(-1)
    
        # Get the shape of the input to reshape at the end
        s = features.shape
        
        # Loop over MPNN levels. There is no "edge network" here.
        # Instead, there is just masked radial functions, that take
        # the role of the adjacency matrix.
        layer = 1 
        for mlp, rad_filt, mask in zip(self.mlps, self.rad_filts, self.masks):
            print("layer {}".format(layer))
            layer += 1 
            # Construct the learnable radial functions
            rad = rad_filt(norms, edge_mask)
            # TODO: Real-valued SO3Scalar so we don't need any hacks
            # Convert to a form that MaskLevel expects
            # Hack to account for the lack of real-valued SO3Scalar and
            # structure of RadialFilters.
            rad = rad[0][..., 0].unsqueeze(-1)

            # Mask the position function if desired
            edge = mask(rad, edge_mask, norms)
            # Convert to a form that MatMul expects
            edge = edge.squeeze(-1)

            # Now pass messages using matrix multiplication with the edge features
            # Einsum b: batch, a: atom, c: channel, x: to be summed over
            features_mp = torch.einsum('baxc,bxc->bac', edge, features)

            # Concatenate the passed messages with the original features
            # last argumet is dim.
            features_mp = torch.cat([features_mp, features], -1)
            
            # Now apply a masked MLP
            features = mlp(features_mp, mask=atom_mask)
            
        # The output are the MLP features reshaped into a set of complex numbers.
        features = features.view(s[0:2] + (1, 1, 2,self.channels_out))

        # construct complex tensor for GELib
        features = torch.complex(features[...,0,:],features[...,1,:])
    
        # Convert to SO3VecArr object.
        features_size = self.channels_out*np.ones((self.max_l[0]+1)).astype(int)
        out = SO3vecArr.zeros(features.size(0),[features.size(1),1],features_size, device = self.device)
        out.parts[0] = features
        
        return out

    @property
    def tau(self):
        return self.channels_out
