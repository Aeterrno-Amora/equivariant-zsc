from itertools import permutations
import math
import torch
from torch import nn


color_indices = None

def _init_color_indices(priv_in_dim, out_dim):
    global color_indices
    if color_indices:
        return color_indices

    # in_color: LongTensor[5, k], each row contains indices related to a color
    # in_nocolor: LongTensor[k], indices not related to any color
    color = torch.arange(0, 5, dtype=torch.long).view(5, 1)
    card = torch.arange(0, 25, dtype=torch.long).view(5, 5)
    hand = torch.stack([card + 25 * i for i in range(5)], dim=1)
    v0 = torch.stack([card, color + 25], dim=1)
    v0 = torch.stack([v0 + 35 * i for i in range(10)], dim=1)
    in_color = torch.stack([
        hand, # partner hand
        card + 167, # fireworks
        torch.arange(0, 50, dtype=torch.long).view(5, 10) + 203, # discard
        color + 261, card + 281, # last action
        v0 + 308, # V0
        color + 666, card + 686, # greedy action
    ], dim=1)
    out_color = color + 10

    in_mask = torch.full((priv_in_dim,), 1, dtype=torch.bool)
    in_mask[in_color.flatten()] = 0
    in_nocolor = in_mask.nonzero().flatten()
    out_mask = torch.full((out_dim,), 1, dtype=torch.bool)
    out_mask[out_color.flatten()] = 0
    out_nocolor = out_mask.nonzero().flatten()

    color_indices = (in_color, out_color, in_nocolor, out_nocolor)
    return color_indices

def build_perms(priv_in_dim, out_dim, symmetries) -> tuple[torch.LongTensor, torch.LongTensor]:
    '''-> priv_in_perms, out_perms: Tensor[num_symmetries, priv_in_dim / out_dim]'''
    in_color, out_color, _, _ = _init_color_indices(priv_in_dim, out_dim)
    priv_in_perms = torch.arange(priv_in_dim, dtype=torch.long).tile(symmetries.size(0), 1)
    priv_in_perms[:, in_color] = in_color[symmetries]
    out_perms = torch.arange(out_dim, dtype=torch.long).tile(symmetries.size(0), 1)
    out_perms[:, out_color[symmetries]] = out_color
    return priv_in_perms, out_perms


# Equivariant network design
#
# feature x: list[Tensor[..., permutation, channel]]
# kernels: dict[(in_len: int, out_len: int), (
#     weight: Tensor[num_weights, out_channel, in_channel],
#     weight_idx: LongTensor[out_permutation, kernel_size],
#     neighbors: LongTensor[out_permutation, kernel_size]
#     in_channel: int, out_channel: int
# )]

def num_perms(n, m):
    return math.factorial(n) // math.factorial(n - m)

def overlap(n, s, t):
    # the signature function described in the paper
    inv_t = torch.full(n, -1, dtype=torch.long)
    inv_t[t] = torch.arange(len(t), dtype=torch.long)
    return tuple(i * 10 + inv_t[i] for i in s if inv_t[i] >= 0)

def pack_gset(input):
    '''list[Tensor[..., permutation, channel]] -> Tensor[..., dim]'''
    return torch.cat([x.flatten(-2, -1) for x in input], dim=-1)

def unpack_gset(input, n, channels):
    '''Tensor[..., dim] -> list[Tensor[..., permutation, channel]]'''
    output = []
    index = 0
    for len, ch in enumerate(channels):
        n_perms = num_perms(n, len)
        output.append(input[..., index : index + n_perms * ch].view(*input.size()[:-1], -1, ch))
        index += n_perms * ch
    return output

def total_size(n, channels):
    return sum(num_perms(n, len) * ch for len, ch in enumerate(channels))


class EqLinear(nn.Module):
    __constants__ = ['n', 'in_channels', 'out_channels']

    def __init__(self, n : int, in_channels, out_channels, radius=-1, bias=True):
        super().__init__()
        self.n = n
        self.in_channels = in_channels
        self.out_channels = out_channels

        self.weight = nn.ParameterDict()
        self.kernels = {}
        for len2, ch2 in enumerate(self.out_channels):
            for len1, ch1 in enumerate(self.in_channels):
                if abs(len1 - len2) > radius:
                    continue
                weight_map = {}
                weight_idx, neighbors = [], []
                for perm2 in permutations(range(n), len2):
                    weight_id, neighbor = [], []
                    for i1, perm1 in enumerate(permutations(range(n), len1)):
                        sig = overlap(n, perm1, perm2)
                        if len1 + len2 - 2 * len(sig) > radius:
                            continue
                        if sig not in weight_map:
                            weight_map[sig] = len(weight_map)
                        weight_id.append(weight_map[sig])
                        neighbor.append(i1)
                    weight_idx.append(weight_id)
                    neighbors.append(neighbor)
                self.weight[f'{len1}{len2}'] = nn.Parameter(torch.empty(len(weight_map), ch2, ch1))
                self.kernels[len1, len2] = (
                    self.weight[f'{len1}{len2}'],
                    torch.tensor(weight_idx, dtype=torch.long),
                    torch.tensor(neighbors, dtype=torch.long),
                    ch1, ch2
                )
        if bias:
            self.bias = nn.ParameterList([nn.Parameter(torch.empty(ch)) for ch in out_channels])
        else:
            self.bias = None
        self.reset_parameters()

    def reset_parameters(self):
        assert(len(list(self.parameters())))
        fan_in = [0 for _ in self.out_channels]
        for (_, len2), (_, _, neighbors, ch1, _) in self.kernels.items():
            fan_in[len2] += neighbors.size(-1) * ch1
        std = [nn.init.calculate_gain('relu') / math.sqrt(fan) for fan in fan_in]
        for (_, len2), (weight, _, _, _, _) in self.kernels.items():
            nn.init.normal_(weight, 0, -std[len2])
        if self.bias is not None:
            for bias in self.bias:
                nn.init.zeros_(bias)

    def forward(self, input, pack=False):
        '''
        input / output: list[Tensor[..., permutation, in/out_channel]]
        '''
        if isinstance(input, torch.Tensor):
            input = unpack_gset(input, self.n, self.in_channels)
        output = [None for _ in self.out_channels]
        for (len1, len2), (weight, weight_idx, neighbors, _, _) in self.kernels.items():
            x = input[len1]
            # x[..., neighbors, :]: Tensor[batch, out_permutation, kernel_size, in_channel]
            # weight[weight_idx]: Tensor[out_permutation, kernel_size, out_channel, in_channel]
            y = (x[..., neighbors, :].unsequeeze(-2) * weight[weight_idx]).sum((-3, -1))
            if output[len2] is None:
                output[len2] = y
            else:
                output[len2] = output[len2] + y
        if pack:
            output = pack_gset(output)
        return output


class EqLSTMCell(nn.Module):
    __constants__ = ['n', 'in_channels', 'hid_channels', 'gate_channels']

    def __init__(self, n, in_channels, hid_channels):
        super().__init__()
        self.n = n
        self.in_channels = in_channels
        self.hid_channels = hid_channels
        self.gate_channels = tuple(ch * 4 for ch in hid_channels)

        self.linear = EqLinear(n, tuple(ch1 + ch2 for ch1, ch2 in zip(in_channels, hid_channels)), self.gate_channels)
        self.reset_parameters()

    def reset_parameters(self):
        assert(len(list(self.parameters())))
        stdv = 1.0 / math.sqrt(self.hid_size)
        for weight in self.parameters():
            nn.init.uniform_(weight, -stdv, stdv)

    def forward(self, input, hid=None, pack=False):
        '''
        input: list[Tensor[..., permutation, in_channel]]
        h0, c0: list[Tensor[..., permutation, hid_channel]]
        '''
        if isinstance(input, torch.Tensor):
            input = unpack_gset(input, self.n, self.in_channels)
        if hid is None:
            h0 = c0 = [torch.zeros(*x.size()[:-1], self.hid_channels, device=x.device) for x in input]
        else:
            h0, c0 = hid
            if isinstance(c0, torch.Tensor):
                h0 = unpack_gset(h0, self.n, self.hid_channels)
                c0 = unpack_gset(c0, self.n, self.hid_channels)

        gates = self.linear([torch.stack([x, h], dim=-1) for x, h in zip(input, h0)])
        h1, c1 = [], []
        for i, gate in enumerate(gates):
            g, input_gate, forget_gate, output_gate = gate.chunk(4, dim=-1)
            c1.append(g.tanh() * input_gate.sigmoid() + c0[i] * forget_gate.sigmoid())
            h1.append(c1[-1].tanh() * output_gate.sigmoid())
        if pack:
            h1, c1 = pack_gset(h1), pack_gset(c1)
        return h1, c1

class EqLSTM(nn.Module):
    __constants__ = ['n', 'in_channels', 'hid_channels', 'num_layers']

    def __init__(self, n, in_channels, hid_channels, num_layers):
        super().__init__()
        self.n = n
        self.in_channels = in_channels
        self.hid_channels = hid_channels
        self.num_layers = num_layers

        self.lstm = nn.ModuleList(
            EqLSTMCell(self.n, self.hid_channels if i else self.in_channels, self.hid_channels)
            for i in range(self.num_layers)
        )

    def forward(self, input, hid=None, pack=False):
        '''
        input / output: Tensor[seq_len, batch, dim] or list[list[Tensor[batch, permutation, channel]]
        h0, c0: Tensor[num_layers, batch, dim] or list[list[Tensor[batch, permutation, channel]]
        '''
        if hid is None:
            hid = [None for _ in range(self.num_layers)]
        else:
            if not isinstance(hid, list):
                hid = list(zip(*hid))

        output = []
        for x in input:
            for i in self.num_layers:
                hid[i] = self.lstm[i](x, hid[i])
            output.append(hid[-1][0])

        h0, c0 = zip(*hid)
        if pack:
            output = torch.stack([pack_gset(y) for y in output], dim=0)
            h0 = torch.stack([pack_gset(h) for h in h0], dim=0)
            c0 = torch.stack([pack_gset(c) for c in c0], dim=0)
        return output, (h0, c0)