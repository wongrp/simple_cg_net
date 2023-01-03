import torch


def get_rel_pos(pos1,pos2):
    """
    Parameters:
    pos1, pos2: torch.Tensor with dimensions (num_atoms,3)
    ---
    returns relative position matrix.
    """
    return pos1.unsqueeze(-2) - pos2.unsqueeze(-3)


def get_norm(pos1,pos2):
    return (get_rel_pos(pos1,pos2).norm(dim = -1, keepdim = True)).squeeze(-1)
