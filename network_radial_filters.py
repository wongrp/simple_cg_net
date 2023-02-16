import torch
import torch.nn as nn

from math import pi




class RadialFilters(nn.Module):
    """
    Generate a set of learnable scalar functions for the aggregation/point-wise
    convolution step.

    One set of radial filters is created for each irrep (l = 0, ..., max_sh).

    Parameters
    ----------
    max_sh : :class:`int`
        Maximum l to use for the spherical harmonics.
    basis_set : iterable of :class:`int`
        Parameters of basis set to use. See :class:`RadPolyTrig` for more details.
    num_channels_out : :class:`int`
        Number of output channels to mix the resulting function into if mix
        is set to True in RadPolyTrig
    num_levels : :class:`int`
        Number of CG levels in the Cormorant.
    """
    def __init__(self, max_sh, basis_set, num_channels_out,
                 num_levels, device=torch.device('cpu'), dtype=torch.float):
        super(RadialFilters, self).__init__()

        self.num_levels = num_levels
        self.max_sh = max_sh

        rad_funcs = [RadPolyTrig(max_sh[level], basis_set, num_channels_out[level], device=device, dtype=dtype) for level in range(self.num_levels)]
        self.rad_funcs = nn.ModuleList(rad_funcs)
        self.tau = [rad_func.tau for rad_func in self.rad_funcs]

        self.num_rad_channels = self.tau[0][0]

        # Other things
        self.device = device
        self.dtype = dtype

        self.zero = torch.tensor(0, device=device, dtype=dtype)

    def forward(self, norms, base_mask):
        """
        Forward pass of the network.

        Parameters
        ----------
        norms : :class:`torch.Tensor`
            Pairwise distance matrix between atoms.
        base_mask : :class:`torch.Tensor`
            Masking tensor with 1s on locations that correspond to active edges
            and zero otherwise.

        Returns
        -------
        rad_func_vals :  list of :class:`RadPolyTrig`
            Values of the radial functions.
        """

        return [rad_func(norms, base_mask) for rad_func in self.rad_funcs]


class RadPolyTrig(nn.Module):
    """
    A variation/generalization of spherical bessel functions.
    Rather than than introducing the bessel functions explicitly we just write out a basis
    that can produce them. Then, when apply a weight mixing matrix to reduce the number of channels
    at the end.
    """
    def __init__(self, max_sh, basis_set, num_channels, mix=False, device=torch.device('cpu'), dtype=torch.float):
        super(RadPolyTrig, self).__init__()

        trig_basis, rpow = basis_set
        print("trig_basis: {}".format(trig_basis))
        self.rpow = rpow
        self.max_sh = max_sh

        assert(trig_basis >= 0 and rpow >= 0)

        self.num_rad = (trig_basis+1)*(rpow+1)
        self.num_channels = num_channels

        # This instantiates a set of functions sin(2*pi*n*x/a), cos(2*pi*n*x/a) with a=1.
        self.scales = torch.cat([torch.arange(trig_basis+1), torch.arange(trig_basis+1)]).view(1, 1, 1, -1).to(device=device, dtype=dtype)
        self.phases = torch.cat([torch.zeros(trig_basis+1), pi/2*torch.ones(trig_basis+1)]).view(1, 1, 1, -1).to(device=device, dtype=dtype)

        # This avoids the sin(0*r + 0) = 0 part from wasting computations.
        self.phases[0, 0, 0, 0] = pi/2

        # Now, make the above learnable
        self.scales = nn.Parameter(self.scales)
        self.phases = nn.Parameter(self.phases)

        # If desired, mix the radial components to a desired shape
        self.mix = mix
        if (mix == 'cplx') or (mix is True):
            self.linear = nn.ModuleList([nn.Linear(2*self.num_rad, 2*self.num_channels).to(device=device, dtype=dtype) for _ in range(max_sh+1)])
            # self.tau = SO3Tau((num_channels,) * (max_sh + 1))
        elif mix == 'real':
            self.linear = nn.ModuleList([nn.Linear(2*self.num_rad, self.num_channels).to(device=device, dtype=dtype) for _ in range(max_sh+1)])
            # self.tau = SO3Tau((num_channels,) * (max_sh + 1))
        elif (mix == 'none') or (mix is False):
            self.linear = None
            # self.tau = SO3Tau((self.num_rad,) * (max_sh + 1))
        else:
            raise ValueError('Can only specify mix = real, cplx, or none! {}'.format(mix))

        self.zero = torch.tensor(0, device=device, dtype=dtype)

    def forward(self, norms, edge_mask):
        # Shape to resize at end
        s = norms.shape
        # s = (13000,29,29)
        # Mask and reshape
        edge_mask = (edge_mask * (norms > 0)).unsqueeze(-1)
        norms = norms.unsqueeze(-1)

        # Get inverse powers
        print("self.scales: {}".format(self.scales))
        rad_powers = torch.stack([torch.where(edge_mask, norms.pow(-pow), self.zero) for pow in range(self.rpow+1)], dim=-1)
        
        # Calculate trig functions
        rad_trig = torch.where(edge_mask, torch.sin((2*pi*self.scales)*norms+self.phases), self.zero).unsqueeze(-1)
        
        # Take the product of the radial powers and the trig components and reshape
        rad_prod = (rad_powers*rad_trig).view(s + (1, 2*self.num_rad,))
        
        # Apply linear mixing function, if desired
        if self.mix == 'cplx':
            radial_functions = [linear(rad_prod).view(s + (self.num_channels, 2)) for linear in self.linear]
        elif self.mix == 'real':
            radial_functions = [linear(rad_prod).view(s + (self.num_channels,)) for linear in self.linear]
            # Hack because real-valued SO3Scalar class has not been implemented yet.
            # TODO: Implement real-valued SO3Scalar and fix this...
            radial_functions = [torch.stack([rad, torch.zeros_like(rad)], dim=-1) for rad in radial_functions]
        else:
            radial_functions = [rad_prod.view(s + (self.num_rad, 2))] * (self.max_sh + 1)
            # shape = (13000, 29, 29, 16, 2)
        
        return radial_functions # shape = (B, C, 2)
