import torch.nn as nn
import cnine
from gelib import *
import numpy as np



class CGLayers(nn.Module):
    def __init__(self, num_CG_layers, norms, n_atoms, n_channels, L, hard_cut_rad):
        super().__init__()
        # track types after CG products, concatenation and mixing
        type = n_channels*np.ones((L+1)).astype(int)
        type_after_nl = np.array(CGproductType(type,type, maxl = L)).astype(int)
        type_sph = type # type resets after each mix # this is tunable to some degree.
        type_after_rel = np.array(CGproductType(type,type_sph, maxl = L)).astype(int)

        # initialize weights
        weights_nl = SO3weightsArr.randn([1,n_atoms],type_after_nl, type)
        weights_rel = SO3weightsArr.randn([n_atoms,n_atoms],type_after_rel,type)

        # make layer module list
        cg_layers = nn.ModuleList() # list-iterable , but registered as modules.
        for layer in range(num_CG_layers):
            cg_layers.append(CGLayer(connectivity, weights_nl, weights_rel, sph, L))
        self.cg_layers = cg_layers

        # scalars module
        self.get_scalars = GetScalars(n_atoms,n_channels, L)

        #
        self.hard_cut_rad = hard_cut_rad
        
    def forward(self, vertices, rel_pos, norms):
        vertices_all = [] # each entry is the set of vertices from one layer. len(vertices) = num_CG_layers
        scalars_all = torch.empty(0)

        # initialize spherical harmonics
        make_sph = SphArr(1,[n_atoms,n_atoms], type_sph)
        sph = make_sph(rel_pos)

        # connectivity
        connectivity = norms < self.hard_cut_rad

        # CG layers
        for idx, cg_layer in enumerate(self.cg_layers):
            vertices = cg_layer(vertices)
            vertices_all.append(vertices)
            scalars = self.get_scalars(vertices)
            scalars_all = torch.cat((scalars_all,scalars))

            # scalars_all should have num_CG_layers*((n_channels)+(L+1)*n_channels^2) elements
        return scalars_all

class GetScalars(nn.Module):
    def __init__(self, n_atoms,n_channels, L):
        super().__init__()
        self.so3norm = GetSO3Norm(n_atoms, n_channels,L)
        self.n_atoms = n_atoms
        self.n_channels = n_channels
        self.L = L

    def forward(self, vertices):
        n_atoms = self.n_atoms
        n_channels = self.n_channels
        L = self.L

        # Take l=0 component
        scalars_part0 = torch.reshape(vertices.parts[0],(n_channels,n_atoms))
        # Take the SO(3) invariant norm
        scalars_norm = self.so3norm(vertices)
        # concatenate
        scalars = torch.cat((scalars_part0,scalars_norm))
        return scalars

class GetSO3Norm(nn.Module):
    def __init__(self, n_atoms, n_channels, L):
        super().__init__()
        self.n_atoms = n_atoms
        self.n_channels = n_channels
        self.L = L

    def forward(self, vertices):
        n_atoms = self.n_atoms
        n_channels = self.n_channels
        L = self.L

        scalars_norm = torch.zeros((L+1)*n_channels**2,n_atoms)
        vertices_conj = SO3vecArr.zeros(1,[n_atoms],n_channels*np.ones((L+1)).astype(int)) # complex conjugate
        channels_count = 0
        for l in range(len(vertices.parts)):
            vertices_conj.parts[l] = torch.resolve_conj(vertices.parts[l]) # take the complex conjugate
            for channel1 in range(n_channels): # iterate all combinations of channels.
                for channel2 in range(n_channels):
                    # dim(SO3vecArr.parts = (parts = L, batches = 1, num_atoms, m, channels))
                    vertices1 = (torch.reshape(vertices.parts[l],(n_atoms,2*l+1,n_channels)))[:,:,channel1] # all elements along channel at part l.
                    vertices2 = (torch.reshape(vertices_conj.parts[l],(n_atoms,2*l+1,n_channels)))[:,:,channel2]
                    assert vertices1.size() == vertices2.size()
                    assert vertices1.size() == torch.Size([n_atoms,2*l+1])

                    # take norm F_l^tau1(F_l^tau2)*
                    scalar_norm = torch.sum(torch.mul(vertices1, vertices2),dim = (1)) # sum along the m axis (with 2l+1 elements)
                    scalars_norm[channels_count,:] = scalar_norm
                    channels_count += 1
        return scalars_norm


class CGLayer(nn.Module):
    def __init__(self,connectivity, weights_nl, weights_rel,sph, L):
        super().__init__()
        self.message_pass = MessagePass(connectivity)
        self.normalize = NormalizeVecArr(L)

        self.weights_nl = weights_nl
        self.weights_rel = weights_rel
        self.L = L
        self.sph = sph


        # self.copy_array = CopyArray()

    def forward(self, vertices):
        vertices_mp = self.message_pass(vertices)
        # quick fix, delete when gather is fixed on the cnine side.#
        for l in range(len(vertices_mp.tau())):
            vertices_mp.parts[l] = torch.unsqueeze(vertices_mp.parts[l],0)
        assert vertices_mp.getb() == 1
        # quick fix, delete when gather is fixed on the cnine side.#
        vertices_cg_nl = CGproduct(vertices_mp, vertices_mp, maxl = self.L)
        vertices_mixed_nl = vertices_cg_nl*self.weights_nl

        # make a repeated copy of the mixed nonlinear vertices.
        new_adims = self.sph.get_adims()
        old_adims = vertices_mixed_nl.get_adims()
        repeat_dims = (np.array(new_adims)/np.array(old_adims)).astype(int)
        current_tau = np.array(vertices_mixed_nl.tau()).astype(int)

        vertices_mixed_nl_repeat = SO3vecArr.zeros(1,new_adims, current_tau)
        for l in range(self.L+1):
            vertices_mixed_nl_repeat.parts[l] = vertices_mixed_nl.parts[l].repeat(1, repeat_dims[0],repeat_dims[1],1,1) # batch, adim 1, adim 2, 2l+1, n_channels
        vertices_cg_rel = CGproduct(vertices_mixed_nl_repeat, self.sph, maxl = self.L)

        # make sure the weight matrix matches
        assert self.weights_rel.get_adims() == vertices_cg_rel.get_adims()
        assert self.weights_rel.get_tau1() == vertices_cg_rel.tau()

        # mix and sum across atoms.
        vertices_mixed_rel = vertices_cg_rel*self.weights_rel
        vertices_sum = SO3vecArr.zeros(1,old_adims,vertices.tau())
        for l in range(self.L+1):
            vertices_sum.parts[l] = torch.sum(vertices_mixed_rel.parts[l], 1)

        # normalize in equivariant manner
        vertices_normed = self.normalize(vertices_sum)

        assert vertices_sum.get_adims() == vertices.get_adims()

        return vertices_sum

class NormalizeVecArr(nn.Module):
    def __init__(self,L):
        super().__init__()
        self.L = L
    def forward(self,reps):
        for l in range(self.L+1):
            norm_factor = torch.sum(reps.parts[l])
            reps.parts[l] /= norm_factor
        return reps

class SphArr(nn.Module):
    """
    Initialize SphArr object with batch, array (atom or particle) dimensions,
    and type (L dimensional channel array). Input relative position tensor of
    type [n_atom, n_atom, 3]. the SO3part.spharm function in forward() must use
    x,y,z arguments instead of the X arguments.
    """
    def __init__(self, batch ,array_dims, type, device="cpu"):
        super().__init__()
        self.batch = batch
        self.array_dims = array_dims
        self.type = type

    def forward(self, X):
        # self.type should be R.tau()
        R = SO3vecArr.zeros(self.batch,self.array_dims,self.type)
        for l in range(0,len(self.type)):
            for b in range(self.batch)
                for i in range(self.array_dims[0]):
                    for j in range(self.array_dims[1]):
                        R.parts[l][b,i,j] = SO3part.spharm(l,X[b,i,j,0],X[b,i,j,1],X[b,i,j,2])
        return R

class MessagePass(nn.Module):
    def __init__(self, connectivity):
        super().__init__()
        self.mask = cnine.Rmask1(connectivity)
    def forward(self, reps):
        reps_mp= reps.gather(self.mask)
        return reps_mp
