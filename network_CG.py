import torch
import torch.nn as nn
import cnine
from gelib import *
import numpy as np
import os
from torch.profiler import profiler
from datetime import datetime

import logging
logger = logging.getLogger(__name__)
# print(os.environ.get('CUDA_HOME'))

class CGLayers(nn.Module):
    def __init__(self, num_CG_layers, num_channels, max_l, hard_cut_rad, device = None, dtype = torch.float32):
        super().__init__()

        # parameters and args
        self.device = device
        self.dtype = dtype
        self.hard_cut_rad = hard_cut_rad

        # track types after CG products, concatenation and mixing
        self.type = num_channels*np.ones((max_l+1)).astype(int)
        self.type_after_nl = np.array(CGproductType(self.type,self.type, maxl = max_l)).astype(int)
        self.type_sph = 1*np.ones((max_l+1)).astype(int) # type resets after each mix # this is tunable to some degree.
        self.type_after_rel = np.array(CGproductType(self.type,self.type_sph, maxl = max_l)).astype(int)

        # make layer module list
        cg_layers = nn.ModuleList() # list-iterable , but registered as modules.
        for layer in range(num_CG_layers):
            cg_layers.append(CGLayer(max_l, num_channels, device))
        self.cg_layers = cg_layers

        # scalars module
        self.get_scalars = GetScalars(num_channels, max_l, device = self.device)

        # initialize weights
        cg_layers.apply(self._init_weights)

    def forward(self, vertices, rel_pos, norms):

        # maximum atom number
        self.num_atoms = get_num_atom(vertices)
        self.get_scalars.num_atoms = self.num_atoms
    
        vertices_all = [] # each entry is the set of vertices from one layer. len(vertices) = num_CG_layers
        scalars_all = torch.empty(0, device = self.device) 
        # relative positions -> spherical harmonics and norms -> connectivity
        make_sph = SphArr(vertices.getb(),[self.num_atoms,self.num_atoms], self.type_sph, device = self.device)

        sph_ti = datetime.now() #time initial
        sph = make_sph(rel_pos)
        sph_tf = datetime.now() #time final

         # some print statements to FIX
        print("the sph fn is on {}".format(make_sph.device))
        # print("NaNs in spherical harmonic vectors? {}:".format(sph))

        connectivity = (norms < self.hard_cut_rad).float()

        # CG layers
        cglayer_dt = 0 
        scalars_dt = 0 
    
        for idx, cg_layer in enumerate(self.cg_layers):
            cglayer_ti = datetime.now()
          
            vertices = cg_layer(vertices, connectivity, sph)
            cglayer_tf = datetime.now()

            vertices_all.append(vertices)
            scalars_ti = datetime.now()
            scalars = self.get_scalars(vertices)
            scalars_tf = datetime.now()

            # concatenate along the layer dimension
            scalars_all = torch.cat((scalars_all,scalars),-1)
            # scalars_all should have num_CG_layers*((num_channels)+(max_l+1)*num_channels^2) elements

            # calculate time elapsed
            cglayer_dt += (cglayer_tf-cglayer_ti).total_seconds() 
            scalars_dt += (scalars_tf-scalars_ti).total_seconds()
        
        sph_dt = (sph_tf-sph_ti).total_seconds() 
        
        logstring1 = "\n In cg network, cg layers took {}s and scalars too {}s \n ".format(cglayer_dt,scalars_dt)
        logstring2 = "The spherical harmonics were constructed in {}s \n ".format(sph_dt)
        logging.info(logstring1)
        logging.info(logstring2)
        
        return scalars_all

    def _init_weights(self, module):
        if isinstance(module, CGLayer):
            # nonlinear weights mixes activations after a CG nonlinearity.
            module.weights_nl = SO3weights.randn(self.type_after_nl, self.type, device = self.device)
            # print(module.weights_nl.parts[0].get_device())
            # rel weights mixes activations after spherical harmonics CG product.
            module.weights_rel = SO3weights.randn(self.type_after_rel,self.type, device = self.device)

class GetScalars(nn.Module):
    def __init__(self,num_channels, max_l, device = torch.device('cpu'), dtype = None):
        super().__init__()
        self.so3norm = GetSO3Norm(num_channels,max_l, device = device, dtype = dtype)
        self.num_channels = num_channels
        self.max_l = max_l
        self.dtype = dtype 

    def forward(self, vertices):
        num_channels = self.num_channels
        max_l = self.max_l
        self.so3norm.num_atoms = self.num_atoms

        # Take l=0 component. Reshape to dim = 3 to fit SO(3) norm.
        #scalars_part0 = torch.reshape(vertices.parts[0], (vertices.getb(),self.num_atoms,self.num_channels))
        scalars_part0 = torch.reshape(torch.view_as_real(vertices.parts[0]), (vertices.getb(),self.num_atoms,2*self.num_channels)) # FIX
        # Take the SO(3) invariant norm
        scalars_norm = self.so3norm(vertices)
        # concatenate along the last (channel) dimension
        scalars = torch.cat((scalars_part0,scalars_norm),-1)

        
        return scalars

class GetSO3Norm(nn.Module):
    """
    Calculate the SO(3) and permutation invariant norm.
    """
    def __init__(self, num_channels, max_l, device = torch.device('cpu'), dtype = None):
        super().__init__()
        self.num_channels = num_channels
        self.max_l = max_l
        self.device = device
        self.dtype = dtype 
    def forward(self, vertices):
        num_channels = self.num_channels
        max_l = self.max_l

        scalars_norm = torch.zeros(vertices.getb(), self.num_atoms,(max_l+1)*num_channels**2, device = self.device, dtype = self.dtype)
        vertices_conj = SO3vecArr.zeros(1,[self.num_atoms],num_channels*np.ones((max_l+1)).astype(int), device = self.device) # complex conjugate
        channels_count = 0
        # loop through every vector component and take the norm for permutation invariance.
        for l in range(len(vertices.parts)):
            vertices_conj.parts[l] = torch.resolve_conj(vertices.parts[l]) # take the complex conjugate
            for channel1 in range(num_channels): # iterate all combinations of channels.
                for channel2 in range(num_channels):
                    vertices1 = vertices.parts[l][:,:,:,:,channel1]
                    vertices2 = vertices_conj.parts[l][:,:,:,:,channel2]
                    # take norm F_l^tau1(F_l^tau2) -- sum along each part 2l+1
                    scalar_norm = torch.sum(torch.mul(vertices1, vertices2),dim = (-1), keepdim = False) # sum along the m axis (with 2l+1 elements)
                    scalars_norm[...,channels_count] = scalar_norm.squeeze(-1)
                
                    channels_count += 1
        return scalars_norm


class CGLayer(nn.Module):
    def __init__(self,max_l, num_channels, device):
        super().__init__()
        self.normalize = NormalizeVecArr(max_l, num_channels)
        self.max_l = max_l
        self.device = device
        # self.copy_array = CopyArray()

    def forward(self, vertices, connectivity, sph):
        vertices_mp = self.message_pass(vertices,connectivity)
        

        # quick fix, delete when gather is fixed on the cnine side.#
        # for l in range(len(vertices_mp.tau())):
        #     vertices_mp.parts[l] = torch.unsqueeze(vertices_mp.parts[l],0)
        # assert vertices_mp.getb() == 1

        # quick fix, delete when gather is fixed on the cnine side.#
        vertices_cg_nl = CGproduct(vertices_mp, vertices_mp, maxl = self.max_l)

        vertices_mixed_nl = vertices_cg_nl*self.weights_nl
        

        # spherical harmonics are NxN, activations are Nx1. Repeat activations by N to take CG product.
        # make a repeated copy of the mixed nonlinear vertices.
        new_adims = sph.get_adims()
        old_adims = vertices_mixed_nl.get_adims()
        repeat_dims = (np.array(new_adims)/np.array(old_adims)).astype(int)

        current_tau = np.array(vertices_mixed_nl.tau()).astype(int)

        vertices_mixed_nl_repeat = SO3vecArr.zeros(sph.getb(),new_adims, current_tau)
        for l in range(self.max_l+1):
            vertices_mixed_nl_repeat.parts[l] = vertices_mixed_nl.parts[l].repeat(1, repeat_dims[0],repeat_dims[1],1,1) # batch, adim 1, adim 2, 2l+1, num_channels
        print("vertices: {}".format(vertices_mixed_nl_repeat.parts[0].device))
        print("sph: {}".format(sph.parts[0].device))
        vertices_mixed_nl_repeat.to(self.device)
        sph.to(self.device)
        print("sph b:{}".format(sph.getb()))
        print("vertices b:{}".format(vertices_mixed_nl_repeat.getb()))
        print("sph a:{}".format(sph.get_adims()))
        print("vertices a:{}".format(vertices_mixed_nl_repeat.get_adims()))
        print("vertices type: {}".format(vertices_mixed_nl_repeat.tau()))
        print("sph type: {}".format(sph.tau()))
        vertices_cg_rel = CGproduct(vertices_mixed_nl_repeat, sph, maxl = self.max_l)

        # make sure the weight matrix matches
        # assert self.weights_rel.get_adims() == vertices_cg_rel.get_adims(), \
        # "'rel' weights has adims {} while activations have adims {}!".format(self.weights_rel.get_adims(),vertices_cg_rel.get_adims())
        # assert self.weights_rel.get_tau1() == vertices_cg_rel.tau(), \
        # "'rel' weights has tau {} while activations have tau {}!".format(self.weights_rel.tau(),vertices_cg_rel.tau())

        # mix and sum across atoms. replace sph.getb()!!
        vertices_mixed_rel = vertices_cg_rel*self.weights_rel
        vertices_sum = SO3vecArr.zeros(sph.getb(),old_adims,vertices.tau())

        for l in range(self.max_l+1):
            # array dims won't match without keepdim
            vertices_sum.parts[l] = torch.sum(vertices_mixed_rel.parts[l], 2, keepdim = True)

        # equivariant norm
        vertices_normed = self.normalize(vertices_sum)

        # check that array dimensions match original.
        assert vertices_sum.get_adims() == vertices.get_adims(), \
        "summed activations has adims {} while input activations have adims {}!".format(vertices_sum.get_adims(), vertices.get_adims())

        return vertices_sum

    def message_pass(self, reps, connectivity):
        mask = cnine.Rmask1(connectivity)
        reps_mp= reps.gather(mask)
        return reps_mp

class NormalizeVecArr(nn.Module):
    def __init__(self,max_l, num_channels):
        super().__init__()
        self.max_l = max_l
        self.num_channels = num_channels
    def forward(self,reps):
        for l in range(self.max_l+1):
            
            norm_factor = torch.sum(reps.parts[l]**2)
            print("normalization constant is {}".format(norm_factor))
            
            reps.parts[l] /= torch.sqrt(norm_factor/self.num_channels)
            assert torch.isnan(reps.parts[l].view(-1)).sum().item()==0
        return reps



class SphArr(nn.Module):
    """
    Initialize SphArr object with batch, array (atom or particle) dimensions,
    and type (max_l dimensional channel array). Input relative position tensor of
    type [n_atom, n_atom, 3]. the SO3part.spharm function in forward() must use
    x,y,z arguments instead of the X arguments.
    """
    def __init__(self, batch ,array_dims, type, device="cpu"):
        super().__init__()
        self.batch = batch
        self.array_dims = array_dims
        self.type = type
        self.device = device
    def forward(self, X):
        # self.type should be R.tau()
        # R = SO3vecArr.zeros(self.batch,self.array_dims,self.type, device = self.device)
        R = SO3vecArr()
        print("relative position dims: {}".format(X.dim()))
        print("relative position sizes: {}".format(X.size()))
        # print("R is on {}".format(R.parts.device))
        for l in range(0,len(self.type)):
        #     for b in range(self.batch):
        #         for i in range(self.array_dims[0]):
        #             for j in range(self.array_dims[1]):
        #                 R.parts[l][b,i,j] = SO3part.spharm(l,X[b,i,j,0],X[b,i,j,1],X[b,i,j,2])
                # R.parts[l] = SO3part.spharm(l,X[b,:,:,:])
            # R.parts.append(SO3partArr.spharm(l,X.unsqueeze(-1), device = self.device))
            Rp = SO3partArr.spharm(l,X.unsqueeze(-1), device = self.device)
            R.parts.append(Rp)

    
        try:
            indices = torch.argwhere(torch.isnan(torch.tensor(Rp)))
            assert torch.isnan(R.parts[l].view(-1)).sum().item()==0,\
                "Encountered {} NaN values at l={} at the following indices: {}.".format( 
                torch.isnan(R.parts[l].view(-1)).sum().item(), l, indices)
        except AssertionError as e:
            print(e)
            i = indices[:,0:3] # first index is "all atoms", second is "the rest of the indices"
            print("Which correspond to relative positions of {}".format(X[i[:,0],i[:,1],i[:,2],:]))
        return R


def get_num_atom(vertices):
    return vertices.parts[0].size(1)
