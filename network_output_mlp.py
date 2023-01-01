import torch
from torch import nn as nn

"""
Taken from Cormorant
"""

class OutputMLP(nn.Module):
    """
    Module to create prediction based upon a set of rotationally invariant
    atom feature vectors.

    This is peformed in a three-step process::

    (1) A MLP is applied to each set of scalar atom-features.
    (2) The environments are summed up.
    (3) Another MLP is applied to the output to predict a single learning target.

    Parameters
    ----------
    num_scalars : :class:`int`
        Number scalars that will be used in the prediction at the output
        of the network.
    bias : :class:`bool`, optional
        Include a bias term in the linear mixing level.
    device : :class:`torch.device`, optional
        Device to instantite the module to.
    dtype : :class:`torch.dtype`, optional
        Data type to instantite the module to.
    """
    def __init__(self, num_scalars, num_mixed=64, activation='leakyrelu', device=torch.device('cpu'), dtype=torch.float):
        super(OutputPMLP, self).__init__()

        self.num_scalars = num_scalars # inputs
        self.num_mixed = num_mixed

        self.mlp1 = BasicMLP(2*num_scalars, num_mixed, num_hidden=1, activation=activation, device=device, dtype=dtype)
        self.mlp2 = BasicMLP(num_mixed, 1, num_hidden=1, activation=activation, device=device, dtype=dtype)

        self.zero = torch.tensor(0, device=device, dtype=dtype)

    def forward(self, atom_scalars, atom_mask):
        """
        Forward step for :class:`OutputPMLP`

        Parameters
        ----------
        atom_scalars : :class:`torch.Tensor`
            Scalar features for each atom used to predict the final learning target.
        atom_mask : :class:`torch.Tensor`
            Unused. Included only for pedagogical purposes.

        Returns
        -------
        predict : :class:`torch.Tensor`
            Tensor used for predictions.
        """
        # Reshape scalars appropriately;
        atom_scalars = atom_scalars.view(atom_scalars.shape[:2] + (2*self.num_scalars,))

        # First MLP applied to each atom
        x = self.mlp1(atom_scalars)

        # Reshape to sum over each atom in molecules, setting non-existent atoms to zero.
        atom_mask = atom_mask.unsqueeze(-1) # ....
        x = torch.where(atom_mask, x, self.zero).sum(1) # sum across atoms.

        # Prediction on permutation invariant representation of molecules
        predict = self.mlp2(x)

        predict = predict.squeeze(-1)

        return predict


class BasicMLP(nn.Module):
    """
    Multilayer perceptron used in various locations.  Operates only on the last axis of the data.

    Parameters
    ----------
    num_in : int
        Number of input channels
    num_out : int
        Number of output channels
    num_hidden : int, optional
        Number of hidden layers.
    layer_width : int, optional
        Width of each hidden layer (number of channels).
    activation : string, optional
        Type of nonlinearity to use.
    device : :obj:`torch.device`, optional
        Device to initialize the level to
    dtype : :obj:`torch.dtype`, optional
        Data type to initialize the level to
    """

    def __init__(self, num_in, num_out, num_hidden=1, layer_width=256, activation='leakyrelu', device=torch.device('cpu'), dtype=torch.float):
        super(BasicMLP, self).__init__()

        self.num_in = num_in # number of neurons in

        self.linear = nn.ModuleList() # every layer's linear function
        self.linear.append(nn.Linear(num_in, layer_width)) # append input linear function
        for i in range(num_hidden-1):
            self.linear.append(nn.Linear(layer_width, layer_width)) # append hidden linear functions
        self.linear.append(nn.Linear(layer_width, num_out)) # append output linear function

        activation_fn = get_activation_fn(activation) # get activation function

        self.activations = nn.ModuleList() # every layer's nonlinearities
        for i in range(num_hidden):
            self.activations.append(activation_fn)

        self.zero = torch.tensor(0, device=device, dtype=dtype)

        self.to(device=device, dtype=dtype)

    def forward(self, x, mask=None):
        # Standard MLP. Loop over a linear layer followed by a non-linear activation
        for (lin, activation) in zip(self.linear, self.activations):
            x = activation(lin(x))

        # After last non-linearity, apply a final linear mixing layer
        x = self.linear[-1](x)

        # If mask is included, mask the output
        if mask is not None:
            x = torch.where(mask, x, self.zero)

        return x

    def scale_weights(self, scale):
        self.linear[-1].weight *= scale
        if self.linear[-1].bias is not None:
            self.linear[-1].bias *= scale


def get_activation_fn(activation):
    activation = activation.lower()
    if activation == 'leakyrelu':
        activation_fn = nn.LeakyReLU()
    elif activation == 'relu':
        activation_fn = nn.ReLU()
    elif activation == 'elu':
        activation_fn = nn.ELU()
    elif activation == 'sigmoid':
        activation_fn = nn.Sigmoid()
    else:
        raise ValueError('Activation function {} not implemented!'.format(activation))
    return activation_fn
