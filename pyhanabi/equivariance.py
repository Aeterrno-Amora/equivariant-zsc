import torch


in_color_indices, out_color_indices = None, None

def _init_color_indices():
    global in_color_indices, out_color_indices
    if in_color_indices:
        return

    # color_indices: LongTensor[5, k], each row contains indices related to a color
    color = torch.arange(0, 5, dtype=torch.long).view(5, 1)
    card = torch.arange(0, 25, dtype=torch.long).view(5, 5)
    hand = torch.stack([card + 25 * i for i in range(5)], dim=1)
    v0 = torch.stack([card, color + 25], dim=1)
    v0 = torch.stack([v0 + 35 * i for i in range(10)], dim=1)
    in_color_indices = torch.stack([
        hand, # partner hand
        card + 167, # fireworks
        torch.arange(0, 50, dtype=torch.long).view(5, 10) + 203, # discard
        color + 261, card + 281, # last action
        v0 + 308, # V0
        color + 666, card + 686, # greedy action
    ], dim=1)
    out_color_indices = color + 10

def build_perms(priv_in_dim, out_dim, symmetries) -> torch.LongTensor:
    '''-> priv_in_perms, out_perms: Tensor[num_symmetries, priv_in_dim / out_dim]'''
    _init_color_indices()
    priv_in_perms = torch.arange(priv_in_dim, dtype=torch.long).tile(symmetries.size(0), 1)
    priv_in_perms[:, in_color_indices] = in_color_indices[symmetries]
    out_perms = torch.arange(out_dim, dtype=torch.long).tile(symmetries.size(0), 1)
    out_perms[:, out_color_indices[symmetries]] = out_color_indices
    return priv_in_perms, out_perms
