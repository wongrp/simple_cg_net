import torch
import torch.nn as nn
from torch.linalg import norm as norm 
import cnine
from gelib import *
import numpy as np
import os
from torch.profiler import profiler
from datetime import datetime
import sys

import logging
logger = logging.getLogger(__name__)
# print(os.environ.get('CUDA_HOME'))

class CGLayers(nn.Module):
    def __init__(self, num_CG_layers, num_channels, max_l, hard_cut_rad, device = None, dtype = torch.float32):
        super().__init__()
        print("Initializing CG layers")
        # parameters and args
        self.device = device
        self.dtype = dtype
        self.hard_cut_rad = hard_cut_rad

        # track types after CG products, concatenation and mixing
        self.type = num_channels*np.ones((max_l+1)).astype(int)
        self.type_after_nl = np.array(DiagCGproductType(self.type,self.type, maxl = max_l)).astype(int)
        self.type_sph = num_channels*np.ones((max_l+1)).astype(int) # type resets after each mix # this is tunable to some degree.
        self.type_after_rel = np.array(DiagCGproductType(self.type,self.type_sph, maxl = max_l)).astype(int)

        

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
        
        setup_ti = datetime.now()

        # maximum atom number
        self.num_atoms = get_num_atom(vertices)
        self.get_scalars.num_atoms = self.num_atoms
    
        vertices_all = [] # each entry is the set of vertices from one layer. len(vertices) = num_CG_layers
        scalars_all = torch.empty(0, device = self.device) 
        # relative positions -> spherical harmonics and norms -> connectivity
        make_sph = SphArr(vertices.getb(),[self.num_atoms,self.num_atoms], self.type_sph, device = self.device)
        
        sph_ti = datetime.now() #time initial
        print(rel_pos.device)
        sph = make_sph(rel_pos)
        sph_tf = datetime.now() #time final

        connectivity = (norms < self.hard_cut_rad).float()

        setup_tf = datetime.now()

        # CG layers
        cglayer_dt = 0 
        scalars_dt = 0 
    
        print("CG Layers")
        for idx, cg_layer in enumerate(self.cg_layers):
            print(torch.cuda.memory_summary())
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

            # calculate time elapsed\
            cglayer_dt += (cglayer_tf-cglayer_ti).total_seconds() 
            scalars_dt += (scalars_tf-scalars_ti).total_seconds()

        setup_dt = (setup_tf-setup_ti).total_seconds()
        sph_dt = (sph_tf-sph_ti).total_seconds() 
        
        logstring1 = "\n In CG network, set up took {}s, cg layers took {}s and scalars took {}s \n ".format(setup_dt,cglayer_dt,scalars_dt)
        logstring2 = "The spherical harmonics were constructed in {}s \n ".format(sph_dt)
        logging.info(logstring1)
        logging.info(logstring2)
        
        return scalars_all

    def _init_weights(self, module):
        if isinstance(module, CGLayer):
            # nonlinear weights mixes activations after a CG nonlinearity.
            module.weights_mp = SO3weights.randn(self.type, self.type, device = self.device)
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

        
        scalars_norm = torch.empty(0,device=self.device, dtype=torch.float)
        vertices_conj = SO3vecArr.zeros(1,[self.num_atoms],num_channels*np.ones((max_l+1)).astype(int), device = self.device) # complex conjugate
        channels_count = 0
        # loop through every vector component and take the norm for permutation invariance.
        for l in range(len(vertices.parts)):
            vertices_conj.parts[l] = torch.resolve_conj(vertices.parts[l]) # take the complex conjugate
            
            # SO3 invariant norm (across m = -l to l dimension --> take real because of numerical error)
            scalar_norm = torch.einsum("ijklc,ijklc -> ijkc", 
            vertices.parts[l],vertices.parts[l].conj()).real.squeeze(2)

            scalars_norm = torch.cat((scalars_norm,scalar_norm),dim = -1)
        
        return scalars_norm


class CGLayer(nn.Module):
    def __init__(self,max_l, num_channels, device):
        super().__init__()
        self.normalize = NormalizeVecArr(max_l, num_channels)
        self.max_l = max_l
        self.device = device
        # self.copy_array = CopyArray()

    def forward(self, vertices, connectivity, sph):
        vertices_cg_rel = CGproduct(vertices,sphx, maxl = self.max_l)
        vertices_mp = self.message_pass(vertices_cg_rel,connectivity)
        

        # quick fix, delete when gather is fixed on the cnine side.#
        # for l in range(len(vertices_mp.tau())):
        #     vertices_mp.parts[l] = torch.unsqueeze(vertices_mp.parts[l],0)
        # assert vertices_mp.getb() == 1

        # quick fix, delete when gather is fixed on the cnine side.#
        ti = datetime.now()
        print("adding diag")
        vertices_cg_nl = DiagCGproduct(vertices_mp, vertices_mp, maxl = self.max_l)
        print("diag added")
        tf = datetime.now()
        vertices_mixed_nl = vertices_cg_nl*self.weights_nl
         
        sph.to(self.device)
        sphx = SO3vecArr.zeros(sph.getb(),sph.get_adims(), sph.tau())
        for l in range(self.max_l+1):
            sphx.parts[l] = torch.einsum('bijmc->bimc',sph.parts[l])

        # print("nl CG product took {}".format((tf-ti).total_seconds()))

        
      
        ti = datetime.now()
        

        
        tf = datetime.now()
        # print((tf-ti).total_seconds())

        # mix and sum across atoms. replace sph.getb()!!
        # vertices_mixed_rel = vertices_cg_rel*self.weights_rel

        # equivariant norm
        vertices_normed = self.normalize(vertices_mp)

   
        return vertices_normed

    def message_pass(self, reps, connectivity):
        mask = cnine.Rmask1(connectivity)
        reps_mp= reps.gather(mask)
        return reps_mp

class NormalizeVecArr(nn.Module):
    def __init__(self,max_l, num_channels, catch_nan = True, norm_option = "norm"):
        super().__init__()
        self.max_l = max_l
        self.num_channels = num_channels
        self.catch_nan = catch_nan
        self.norm_option = norm_option
    def forward(self,reps):
        for l in range(self.max_l+1):
            
            norm_factor = torch.sum(torch.pow(reps.parts[l],2))
            norm_factor = (2*l+1)*torch.linalg.norm(reps.parts[l])

            assert self.norm_option == "norm" or self.norm_option == "component", "must indicate norm_option as component or norm!"
            # norm normalization 
            if self.norm_option == "norm": 
                reps.parts[l] = reps.parts[l]/(norm_factor/self.num_channels)
            
            # component normalization ; <x_i^2> = 1 
            elif self.norm_option == "component":
                reps.parts[l] = reps.parts[l]/(norm_factor/self.num_channels) 

                

            if self.catch_nan == True: 
                try: 
                    assert torch.isnan(reps.parts[l].view(-1)).sum().item()==0
                except AssertionError as e: 
                    print(e)
                    print("normalization constant is {}".format(norm_factor))
                    sys.exit()
        return reps



class SphArr(nn.Module):
    """
    Initialize SphArr object with batch, array (atom or particle) dimensions,
    and type (max_l dimensional channel array). Input relative position tensor of
    type [n_atom, n_atom, 3]. the SO3part.spharm function in forward() must use
    x,y,z arguments instead of the X arguments.
    """
    def __init__(self, batch ,array_dims, type, device="cpu", catch_nan = False):
        super().__init__()
        self.batch = batch
        self.array_dims = array_dims
        self.type = type
        self.device = device
        self.catch_nan = catch_nan
    def forward(self, X):
        # self.type should be R.tau()
        # R = SO3vecArr.zeros(self.batch,self.array_dims,self.type, device = self.device)
        
        R = SO3vecArr()
        #print("relative positions is on the {}:".format(X.device))
        # X = X.to('cpu') # GElib does not support x being on GPU atm. There should be an error message. Returns NaNs. 
        #print("relative position dims: {}".format(X.dim()))
        #print("relative position sizes: {}".format(X.size()))
        for l in range(0,len(self.type)):
            Rp = SO3partArr.spharm(l,X.unsqueeze(-1), device = self.device)
            R.parts.append(Rp)
            
            if self.catch_nan == True: 
                try:
                    indices = torch.argwhere(torch.isnan(torch.tensor(Rp)))
                    assert torch.isnan(R.parts[l].view(-1)).sum().item()==0,\
                        "Encountered {} NaN values at l={} at the following indices: {}.".format( 
                        torch.isnan(R.parts[l].view(-1)).sum().item(), l, indices)
                except AssertionError as e:
                    print(e)
                    id = indices[:,0:3] # first index is "all atoms", second is "the rest of the indices"
                    print("Check if they truly are NaNs by printing the first one: {}".format(torch.tensor(Rp)[list(indices[2,:])]))
                    print("The indices correspond to relative positions of {}".format(X[id[:,0],id[:,1],id[:,2],:]))
                    
                    # check if the nans come from X (they shouldn't)
                    assert torch.isnan(X.view(-1)).sum().item()==0 

                    # reproduce
                    for idx in id:
                        print(idx)
                        print(X[idx[0],idx[1],idx[2],:])
                        print(SO3part.spharm(l,X[idx[0],idx[1],idx[2],0], X[idx[0],idx[1],idx[2],1], X[idx[0],idx[1],idx[2],2]))
                    
                    # print first of both old and new
                    print("old spharm (SO3part): {}".format(SO3part.spharm(l,X[0,0,1,0],X[0,0,1,1],X[0,0,1,2])))
                    print("new spharm (SO3partArr): {}".format(SO3partArr.spharm(l,(X[0,0,1,:]).unsqueeze(-1).unsqueeze(0))))
                    print("Check if the relative positions match: old: {} new: {}".format(
                        (X[0,0,1,:]).unsqueeze(-1).unsqueeze(0),[X[0,0,1,0],X[0,0,1,1],X[0,0,1,2]]))
                    sys.exit()
                    
        return R


def get_num_atom(vertices):
    return vertices.parts[0].size(1)
