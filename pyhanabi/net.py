# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#
import torch
from torch import nn, Tensor
from equivariance import EqLinear, EqLSTM, pack_gset


def build_mlp(layer, in_dim, hid_dim, num_layer):
    layers = [layer(in_dim, hid_dim), nn.ReLU()]
    for _ in range(1, num_layer):
        layers.append(layer(hid_dim, hid_dim))
        layers.append(nn.ReLU())
    return nn.Sequential(*layers)


class FFWDNet(torch.jit.ScriptModule):
    def __init__(self, priv_in_dim, hid_dim):
        super().__init__()
        self.net = build_mlp(nn.Linear, priv_in_dim, hid_dim, 3)

    @torch.jit.script_method
    def forward(
        self, priv_s: Tensor, publ_s: Tensor, hid: dict[str, Tensor]
    ) -> tuple[Tensor, dict[str, Tensor]]:
        return self.net(priv_s), hid


class LSTMNet(torch.jit.ScriptModule):
    __constants__ = ["num_lstm_layer", "public"]

    def __init__(self,
        priv_in_dim, publ_in_dim, hid_dim, num_lstm_layer, num_ff_layer=1,
        public=True, equivariant=False, **kwargs
    ):
        super().__init__()
        self.num_lstm_layer = num_lstm_layer
        self.public = public
        if equivariant:
            priv_in_dim = kwargs['priv_in_channels']
            publ_in_dim = kwargs['publ_in_channels']
            hid_dim = kwargs['hid_channels']
            linear_cls, lstm_cls = EqLinear, EqLSTM
        else:
            linear_cls, lstm_cls = nn.Linear, nn.LSTM

        if not self.public:
            self.priv_net = build_mlp(linear_cls, priv_in_dim, hid_dim, num_ff_layer)
        else:
            self.priv_net = build_mlp(linear_cls, priv_in_dim, hid_dim, 3)
            self.publ_net = build_mlp(linear_cls, publ_in_dim, hid_dim, num_ff_layer)
        self.lstm = lstm_cls(hid_dim, hid_dim, num_layers=num_lstm_layer)
        if not equivariant:
            self.lstm.flatten_parameters()

    # @torch.jit.script_method
    # def get_h0(self, batchsize: int) -> dict[str, Tensor]:
    #     shape = (self.num_lstm_layer, batchsize, self.hid_dim)
    #     hid = {"h0": torch.zeros(*shape), "c0": torch.zeros(*shape)}
    #     return hid
    # Empty hid defaults to zeros.

    @torch.jit.script_method
    def forward(
        self, priv_s: Tensor, publ_s: Tensor, hid: dict[str, Tensor]
    ) -> tuple[Tensor, dict[str, Tensor]]:
        '''
        net(
            priv_s: Tensor[(seq_len), batch, priv_in_dim]
            publ_s: Tensor[(seq_len), batch, publ_in_dim]
            hid: dict[str, Tensor[batch, num_layer, num_player, hid_dim]]
        ) -> (
            o: Tensor[(seq_len), batch, hid_dim]
            hid: dict[str, Tensor[batch, num_layer, num_player, hid_dim]]
        )
        '''
        one_step = priv_s.dim() == 2
        if one_step: # expand seq_len dim
            priv_s = priv_s.unsqueeze(0)
            publ_s = publ_s.unsqueeze(0)
        assert priv_s.dim() == 3, f"dim = 3 or 2, [(seq_len), batch, dim], get {priv_s.shape}"

        if not self.public:
            x = self.priv_net(priv_s)
        else:
            x = self.publ_net(publ_s)
        if hid:
            # hid size: [batch, num_layer, num_player, dim] -> [num_layer, batch x num_player, dim]
            # hid = {k: v.transpose(0, 1).flatten(1, 2).contiguous() for k, v in hid.items()}
            o, (h, c) = self.lstm(x, (hid["h0"], hid["c0"]))
        else:
            o, (h, c) = self.lstm(x)
        if self.public:
            priv_o = self.priv_net(priv_s)
            o = priv_o * o

        # hid size: [num_layer, batch x num_player, dim] -> [batch, num_layer, num_player, dim]
        # batchsize = priv_s.size(-2)
        # interim_hid_shape = (self.num_lstm_layer, batchsize, -1, self.hid_dim)
        # h = h.view(*interim_hid_shape).transpose(0, 1)
        # c = c.view(*interim_hid_shape).transpose(0, 1)

        if one_step: # squeeze seq_len dim
            o = o.squeeze(0)
        return o, {"h0": h, "c0": c}
