# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#
from itertools import permutations
import torch
from torch import nn
import torch.nn.functional as F

from common_utils.hparams import hparams
from common_utils.checkpoint import build_object_from_config
from equivariance import _init_color_indices, build_perms, color_indices, EqLinear, total_size


class Model(torch.jit.ScriptModule):
    def __init__(self, hparams=hparams):
        super().__init__()
        # for backward compatibility
        self.config = hparams['net']

        if self.config['equivariant']:
            in_color, in_nocolor, out_color, out_nocolor = _init_color_indices(
                self.config['priv_in_dim'], self.config['out_dim']
            )
            hid_channels = self.config['hid_channels']
            self.config['hid_dim'] = total_size(5, hid_channels)
            self.config['priv_in_channels'] = (in_nocolor.size(-1), in_color.size(-1))
            self.config['publ_in_channels'] = (in_nocolor.size(-1), in_color.size(-1) - 25)
            self.config['out_channels'] = (out_nocolor.size(-1), out_color.size(-1))

            self.fc_v = EqLinear(5, hid_channels, (1,))
            self.fc_a = EqLinear(5, hid_channels, self.config['out_channels'])
            self.pred_1st = EqLinear(5, hid_channels, (5 * 3,)) # not sure
        else:
            hid_dim = self.config['hid_dim']
            self.fc_v = nn.Linear(hid_dim, 1)
            self.fc_a = nn.Linear(hid_dim, self.config['out_dim'])
            self.pred_1st = nn.Linear(hid_dim, 5 * 3) # for aux task

        self.eqc = self.config["eqc"]
        if self.eqc:
            group = self.config["group"]
            if group == "cyclic":
                self.symmetries = torch.tensor([
                    [0,1,2,3,4], [4,0,1,2,3], [3,4,0,1,2], [2,3,4,0,1], [1,2,3,4,0],
                ])
            elif group == "dihedral":
                self.symmetries = torch.tensor([
                    [0,1,2,3,4], [1,2,3,4,0], [2,3,4,0,1], [3,4,0,1,2], [4,0,1,2,3],
                    [4,3,2,1,0], [3,2,1,0,4], [2,1,0,4,3], [1,0,4,3,2], [0,4,3,2,1],
                ])
            elif group == "symmetric":
                self.symmetries = torch.tensor(list(permutations(range(5))))
            else:
                raise ValueError(f"Unsupported EQC group: {group}")
            self.num_symmetries = len(self.symmetries)
            self.priv_in_perms, self.out_perms = build_perms(
                self.config['priv_in_dim'], self.config['out_dim'], self.symmetries
            )
            self.publ_in_perms = self.priv_in_perms[:, 125:] - 125

        self.net = build_object_from_config(self.config, parent_cls=nn.Module)
        # net(
        #   priv_s: Tensor[(seq_len), batch, priv_in_dim]
        #   publ_s: Tensor[(seq_len), batch, publ_in_dim]
        #   hid: dict[str, Tensor[batch, num_layer, num_player, hid_dim]]
        # ) -> (
        #   o: Tensor[(seq_len), batch, hid_dim]
        #   hid: dict[str, Tensor[batch, num_layer, num_player, hid_dim]]
        # )

    @torch.jit.script_method
    def eqc_permute_input(
        self, priv_s: torch.Tensor, publ_s: torch.Tensor, hid: dict[str, torch.Tensor]
    ) -> tuple[torch.Tensor, torch.Tensor, dict[str, torch.Tensor]]:
        '''
        self.permute_input(
            priv_s: Tensor[(seq_len), batch, priv_in_dim],
            publ_s: Tensor[(seq_len), batch, publ_in_dim],
            hid: dict[str, Tensor[batch, num_layer, num_player, hid_dim]],
        ) -> (priv_s, publ_s, hid)
            with batch expanded to batch * num_symmetries
        using index arrays
            self.priv_in_perms: LongTensor[num_symmetries, priv_in_dim]
            self.publ_in_perms: LongTensor[num_symmetries, publ_in_dim]
        '''
        priv_s = priv_s[..., self.priv_in_perms].flatten(-2, -3)
        publ_s = publ_s[..., self.publ_in_perms].flatten(-2, -3)
        hid = {k: v.tile(self.num_symmetries, 1, 1, 1) for k, v in hid.items()}
        return priv_s, publ_s, hid

    @torch.jit.script_method
    def eqc_symmetrize_output(
        self, a: torch.Tensor, hid: dict[str, torch.Tensor]
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        '''
        self.symmetrize_output(
            a: Tensor[(seq_len), batch * num_symmetries, out_dim],
            hid: dict[str, Tensor[batch * num_symmetries, num_layer, num_player, hid_dim]],
        ) -> (a, hid)
            with batch * num_symmetries aggregated to batch
        using
            self.out_perms: LongTensor[num_symmetries, out_dim]
        '''
        a = a.view(*a.shape[:-2], -1, self.num_symmetries, a.size(-1))
        a = a.gather(-1, self.out_perms.expand_as(a)).mean(-2)
        hid = {k: v.view(-1, self.num_symmetries, *v.size()[1:]).mean(1) for k, v in hid.items()}
        return a, hid

    @torch.jit.script_method
    def equivariant_input(
        self, priv_s: torch.Tensor, publ_s: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        '''
        self.equivariant_input(
            priv_s: Tensor[batch, priv_in_dim],
            publ_s: Tensor[batch, publ_in_dim],
        ) -> (priv_s, publ_s)
        '''
        in_color, _, in_nocolor, _ = color_indices
        priv_s = [priv_s[..., in_nocolor], priv_s[..., in_color]]
        publ_s = [publ_s[..., in_nocolor - 125], publ_s[..., in_color[25:] - 125]]
        return priv_s, publ_s

    @torch.jit.script_method
    def equivariant_output(self, a: torch.Tensor) -> torch.Tensor:
        '''
        self.equivariant_output(
            a: list[Tensor[batch, permutation, out_dim]]
        ) -> a
        '''
        a0, a1 = a[0].squeeze(-2), a[1].squeeze(-1)
        return torch.stack([a0[..., :10], a1, a0[..., 10:]], dim=-1)

    @torch.jit.script_method
    def get_h0(self, batchsize: int) -> dict[str, torch.Tensor]:
        return {}

    @torch.jit.script_method
    def act(
        self, priv_s: torch.Tensor, publ_s: torch.Tensor, hid: dict[str, torch.Tensor]
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        '''
        model.act(
            priv_s: Tensor[batch, priv_in_dim],
            publ_s: Tensor[batch, publ_in_dim],
            hid: dict[str, Tensor[batch, num_layer, num_player, hid_dim]]
        ) -> (
            a: Tensor[batch, num_action],
            hid: dict[str, Tensor[batch, num_layer, num_player, hid_dim]]
        )
        '''
        assert priv_s.dim() == 2, f"dim should be 2, [batch, dim], get {priv_s.shape}"

        if self.eqc:
            priv_s, publ_s, hid = self.eqc_permute_input(priv_s, publ_s, hid)
        elif self.equivariant:
            priv_s, publ_s = self.equivariant_input(priv_s, publ_s)

        # hid: [batch, num_layer, num_player, dim] -> [num_layer, batch x num_player, dim]
        if hid:
            assert hid["h0"].dim() == 4
            hid = {k: v.transpose(0, 1).flatten(1, 2).contiguous() for k, v in hid.items()}
        # TODO: act has, but forward doesn't. If both need, move to Net.

        o, hid = self.net(priv_s, publ_s, hid)
        a = self.fc_a(o)

        # hid: [num_layer, batch x num_player, dim] -> [batch, num_layer, num_player, dim]
        if hid:
            batchsize = priv_s.size(-2)
            interim_hid_shape = (self.num_lstm_layer, batchsize, -1, self.hid_dim)
            hid = {k: v.view(*interim_hid_shape).transpose(0, 1) for k, v in hid.items()}

        if self.eqc:
            a, hid = self.eqc_symmetrize_output(a, hid)
        elif self.equivariant:
            a = self.equivariant_output(a)

        return a, hid

    @torch.jit.script_method
    def forward(
        self,
        priv_s: torch.Tensor,
        publ_s: torch.Tensor,
        legal_move: torch.Tensor,
        action: torch.Tensor,
        hid: dict[str, torch.Tensor],
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        '''
        model(
            priv_s: Tensor[(seq_len), batch, priv_in_dim],
            publ_s: Tensor[(seq_len), batch, publ_in_dim],
            legal_move: Tensor[(seq_len), batch, num_action],
            action: LongTensor[(seq_len), batch],
            hid: dict[str, Tensor[(seq_len), batch, num_layer, num_player, hid_dim]]
        ) -> (
            qa: Tensor[(seq_len), batch],
            greedy_action: LongTensor[(seq_len), batch],
            q: Tensor[(seq_len), batch, num_action],
            o: Tensor[(seq_len), batch, hid_dim]
        )
        '''
        assert priv_s.dim() == 3 or priv_s.dim() == 2, f"dim = 3 or 2, [(seq_len), batch, dim], get {priv_s.shape}"
        if self.eqc:
            priv_s, publ_s, hid = self.eqc_permute_input(priv_s, publ_s, hid)
        o, hid = self.net(priv_s, publ_s, hid)
        a = self.fc_a(o) # [(seq_len), batch]
        if self.eqc:
            a, hid = self.eqc_symmetrize_output(a, hid)
        elif self.equivariant:
            a = self.equivariant_output(a)
        v = self.fc_v(o, pack=True) # [(seq_len), batch]
        legal_a = a * legal_move
        q = v + legal_a - legal_a.mean(-1, keepdim=True)
        qa = q.gather(-1, action.unsqueeze(-1)).squeeze(-1)
        legal_q = (1 + q - q.min()) * legal_move
        greedy_action = legal_q.argmax(-1)
        return qa, greedy_action.detach(), q, o

    def pred_loss_1st(self, o, target_p, hand_slot_mask, seq_len):
        # EQC is not implemented for aux task
        '''
        model.pred_loss_1st(
            o: Tensor[seq_len, batch, hid_dim],
            target_p: Tensor[seq_len, batch, (num_player), 5, 3],
            hand_slot_mask: Tensor[seq_len, batch, (num_player), 5],
            seq_len: Tensor[batch]
        ) -> (
            xent: Tensor[batch],
            avg_xent: float,
            q: Tensor[seq_len, batch, (num_player), 5, 3],
            seq_xent: Tensor[seq_len, batch]
        )
        '''
        logit = self.pred_1st(o).view(target_p.size())
        q = F.softmax(logit, -1)
        logq = F.log_softmax(logit, -1)
        plogq = (target_p * logq).sum(-1)
        xent = -(plogq * hand_slot_mask).sum(-1) / hand_slot_mask.sum(-1).clamp(min=1e-6)

        if xent.dim() == 3: # [seq_len, batch, num_player]
            xent = xent.mean(2)
        seq_xent = xent # save before sum out
        xent = xent.sum(0)
        assert xent.size() == seq_len.size()
        avg_xent = (xent / seq_len).mean().item()
        return xent, avg_xent, q, seq_xent.detach()


class R2D2Agent(torch.jit.ScriptModule):
    __constants__ = ["vdn", "multi_step", "gamma", "eta", "boltzmann", "uniform_priority", "net"]

    def __init__(self, hparams=hparams):
        super().__init__()
        self.hparams = hparams
        self.online_net = Model(hparams)
        self.target_net = Model(hparams)
        for p in self.target_net.parameters():
            p.requires_grad = False
        #self.vdn = vdn
        #self.multi_step = multi_step
        #self.gamma = gamma
        #self.eta = eta
        #self.net = net
        #self.num_lstm_layer = num_lstm_layer
        #self.boltzmann = boltzmann_act
        #self.uniform_priority = uniform_priority
        #self.off_belief = off_belief
        #self.equivariant = equivariant

    @torch.jit.script_method
    def get_h0(self, batchsize: int) -> dict[str, torch.Tensor]:
        return self.online_net.get_h0(batchsize)

    def clone(self, device, override=None):
        if override is not None:
            hparams = self.hparams.copy()
            hparams.update(override)
        else:
            hparams = self.hparams
        cloned = type(self)(hparams)
        cloned.load_state_dict(self.state_dict())
        cloned.train(self.training)
        return cloned.to(device)

    def sync_target_with_online(self):
        self.target_net.load_state_dict(self.online_net.state_dict())

    @torch.jit.script_method
    def greedy_act(
        self,
        priv_s: torch.Tensor,
        publ_s: torch.Tensor,
        legal_move: torch.Tensor,
        hid: dict[str, torch.Tensor],
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        adv, new_hid = self.online_net.act(priv_s, publ_s, hid)
        legal_adv = (1 + adv - adv.min()) * legal_move
        greedy_action = legal_adv.argmax(1).detach()
        return greedy_action, new_hid

    @torch.jit.script_method
    def boltzmann_act(
        self,
        priv_s: torch.Tensor,
        publ_s: torch.Tensor,
        legal_move: torch.Tensor,
        temperature: torch.Tensor,
        hid: dict[str, torch.Tensor],
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor], torch.Tensor]:
        adv, new_hid = self.online_net.act(priv_s, publ_s, hid)
        temperature = temperature.unsqueeze(1)
        assert adv.dim() == temperature.dim()
        logit = adv / temperature
        legal_logit = logit - (1 - legal_move) * 1e30
        assert legal_logit.dim() == 2
        prob = F.softmax(legal_logit, 1)
        action = prob.multinomial(1).squeeze(1).detach()
        return action, new_hid, prob

    @torch.jit.script_method
    def act(self, obs: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        """
        Acts on the given obs, with eps-greedy policy.
        output: {'a' : actions}, a long Tensor of shape
            [batchsize] for IQL or [batchsize, num_player] for VDN
        """
        priv_s = obs["priv_s"]
        publ_s = obs["publ_s"]
        legal_move = obs["legal_move"]
        if "eps" in obs:
            eps = obs["eps"].flatten(0, 1)
        else:
            eps = torch.zeros((priv_s.size(0),), device=priv_s.device)

        if self.hparams['vdn']:
            bsize, num_player = obs["priv_s"].size()[:2]
            priv_s = obs["priv_s"].flatten(0, 1)
            publ_s = obs["publ_s"].flatten(0, 1)
            legal_move = obs["legal_move"].flatten(0, 1)
        else:
            bsize, num_player = obs["priv_s"].size()[0], 1

        hid = {"h0": obs["h0"], "c0": obs["c0"]}

        if self.hparams['boltzmann']:
            temp = obs["temperature"].flatten(0, 1)
            greedy_action, new_hid, prob = self.boltzmann_act(
                priv_s, publ_s, legal_move, temp, hid
            )
            reply = {"prob": prob}
        else:
            greedy_action, new_hid = self.greedy_act(priv_s, publ_s, legal_move, hid)
            reply = {}

        if self.hparams['greedy']:
            action = greedy_action
        else:
            assert greedy_action.size() == eps.size()
            random_action = legal_move.multinomial(1).squeeze(1)
            rand = torch.rand(greedy_action.size(), device=greedy_action.device)
            action = torch.where(rand < eps, random_action, greedy_action).detach().long()

        if self.hparams['vdn']:
            action = action.view(bsize, num_player)
            greedy_action = greedy_action.view(bsize, num_player)
            # rand = rand.view(bsize, num_player)

        reply["a"] = action.detach().cpu()
        reply["h0"] = new_hid["h0"].detach().cpu()
        reply["c0"] = new_hid["c0"].detach().cpu()
        return reply

    @torch.jit.script_method
    def compute_target(self, input_: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        assert self.hparams['multi_step'] == 1
        priv_s = input_["priv_s"]
        publ_s = input_["publ_s"]
        legal_move = input_["legal_move"]
        act_hid = {
            "h0": input_["h0"],
            "c0": input_["c0"],
        }
        fwd_hid = {
            "h0": input_["h0"].transpose(0, 1).flatten(1, 2).contiguous(),
            "c0": input_["c0"].transpose(0, 1).flatten(1, 2).contiguous(),
        }
        reward = input_["reward"]
        terminal = input_["terminal"]

        if self.hparams['boltzmann']:
            temp = input_["temperature"].flatten(0, 1)
            next_a, _, next_pa = self.boltzmann_act(
                priv_s, publ_s, legal_move, temp, act_hid
            )
            next_q = self.target_net(priv_s, publ_s, legal_move, next_a, fwd_hid)[2]
            qa = (next_q * next_pa).sum(1)
        else:
            next_a = self.greedy_act(priv_s, publ_s, legal_move, act_hid)[0]
            qa = self.target_net(priv_s, publ_s, legal_move, next_a, fwd_hid)[0]

        assert reward.size() == qa.size()
        target = reward + (1 - terminal) * self.hparams['gamma'] * qa
        return {"target": target.detach()}

    @torch.jit.script_method
    def compute_priority(self, input_: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        if self.hparams['uniform_priority']:
            return {"priority": torch.ones_like(input_["reward"].sum(1))}

        # swap batch_dim and seq_dim
        for k, v in input_.items():
            if k != "seq_len":
                input_[k] = v.transpose(0, 1).contiguous()

        obs = {
            "priv_s": input_["priv_s"],
            "publ_s": input_["publ_s"],
            "legal_move": input_["legal_move"],
        }
        if self.hparams['boltzmann']:
            obs["temperature"] = input_["temperature"]

        if self.hparams['off_belief']:
            obs["target"] = input_["target"]

        hid = {"h0": input_["h0"], "c0": input_["c0"]}
        action = {"a": input_["a"]}
        reward = input_["reward"]
        terminal = input_["terminal"]
        bootstrap = input_["bootstrap"]
        seq_len = input_["seq_len"]
        err, _, _ = self.td_error(
            obs, hid, action, reward, terminal, bootstrap, seq_len
        )
        priority = err.abs()
        priority = self.aggregate_priority(priority, seq_len).detach().cpu()
        return {"priority": priority}

    @torch.jit.script_method
    def td_error(
        self,
        obs: dict[str, torch.Tensor],
        hid: dict[str, torch.Tensor],
        action: dict[str, torch.Tensor],
        reward: torch.Tensor,
        terminal: torch.Tensor,
        bootstrap: torch.Tensor,
        seq_len: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        max_seq_len = obs["priv_s"].size(0)
        priv_s = obs["priv_s"]
        publ_s = obs["publ_s"]
        legal_move = obs["legal_move"]
        action = action["a"]

        for k, v in hid.items():
            hid[k] = v.flatten(1, 2).contiguous()

        bsize, num_player = priv_s.size(1), 1
        if self.hparams['vdn']:
            num_player = priv_s.size(2)
            priv_s = priv_s.flatten(1, 2)
            publ_s = publ_s.flatten(1, 2)
            legal_move = legal_move.flatten(1, 2)
            action = action.flatten(1, 2)

        # this only works because the trajectories are padded,
        # i.e. no terminal in the middle
        online_qa, greedy_a, online_q, lstm_o = self.online_net(
            priv_s, publ_s, legal_move, action, hid
        )

        if self.hparams['off_belief']:
            target = obs["target"]
        else:
            target_qa, _, target_q, _ = self.target_net(
                priv_s, publ_s, legal_move, greedy_a, hid
            )

            if self.hparams['boltzmann']:
                temperature = obs["temperature"].flatten(1, 2).unsqueeze(2)
                # online_q: [seq_len, bathc * num_player, num_action]
                logit = online_q / temperature.clamp(min=1e-6)
                # logit: [seq_len, batch * num_player, num_action]
                legal_logit = logit - (1 - legal_move) * 1e30
                assert legal_logit.dim() == 3
                pa = F.softmax(legal_logit, 2).detach()
                # pa: [seq_len, batch * num_player, num_action]

                assert target_q.size() == pa.size()
                target_qa = (pa * target_q).sum(-1).detach()
                assert online_qa.size() == target_qa.size()

            if self.hparams['vdn']:
                online_qa = online_qa.view(max_seq_len, bsize, num_player).sum(-1)
                target_qa = target_qa.view(max_seq_len, bsize, num_player).sum(-1)
                lstm_o = lstm_o.view(max_seq_len, bsize, num_player, -1)

            target_qa = torch.cat(
                [target_qa[self.multi_step :], target_qa[: self.multi_step]], 0
            )
            target_qa[-self.multi_step :] = 0
            assert target_qa.size() == reward.size()
            target = reward + bootstrap * (self.gamma ** self.multi_step) * target_qa

        mask = torch.arange(0, max_seq_len, device=seq_len.device)
        mask = (mask.unsqueeze(1) < seq_len.unsqueeze(0)).float()
        err = (target.detach() - online_qa) * mask
        if self.off_belief and "valid_fict" in obs:
            err = err * obs["valid_fict"]
        return err, lstm_o, online_q

    def aux_task_iql(self, lstm_o, hand, seq_len, rl_loss_size, stat):
        seq_size, bsize, _ = hand.size()
        own_hand = hand.view(seq_size, bsize, 5, 3)
        own_hand_slot_mask = own_hand.sum(3)
        pred_loss1, avg_xent1, _, _ = self.online_net.pred_loss_1st(
            lstm_o, own_hand, own_hand_slot_mask, seq_len
        )
        assert pred_loss1.size() == rl_loss_size
        stat["aux"].feed(avg_xent1)
        return pred_loss1

    def aux_task_vdn(self, lstm_o, hand, t, seq_len, rl_loss_size, stat):
        """1st and 2nd order aux task used in VDN"""
        seq_size, bsize, num_player, _ = hand.size()
        own_hand = hand.view(seq_size, bsize, num_player, 5, 3)
        own_hand_slot_mask = own_hand.sum(4)
        pred_loss1, avg_xent1, belief1, _ = self.online_net.pred_loss_1st(
            lstm_o, own_hand, own_hand_slot_mask, seq_len
        )
        assert pred_loss1.size() == rl_loss_size

        stat["aux"].feed(avg_xent1)
        return pred_loss1

    def aggregate_priority(self, priority, seq_len):
        p_mean = priority.sum(0) / seq_len
        p_max = priority.max(0)[0]
        agg_priority = self.eta * p_max + (1.0 - self.eta) * p_mean
        return agg_priority

    def loss(self, batch, aux_weight, stat):
        err, lstm_o, online_q = self.td_error(
            batch.obs,
            batch.h0,
            batch.action,
            batch.reward,
            batch.terminal,
            batch.bootstrap,
            batch.seq_len,
        )
        rl_loss = F.smooth_l1_loss(
            err, torch.zeros_like(err), reduction="none"
        )
        rl_loss = rl_loss.sum(0)
        stat["rl_loss"].feed((rl_loss / batch.seq_len).mean().item())

        priority = err.abs()
        priority = self.aggregate_priority(priority, batch.seq_len).detach().cpu()

        loss = rl_loss
        if aux_weight <= 0:
            return loss, priority, online_q

        if self.vdn:
            pred1 = self.aux_task_vdn(
                lstm_o,
                batch.obs["own_hand"],
                batch.obs["temperature"],
                batch.seq_len,
                rl_loss.size(),
                stat,
            )
            loss = rl_loss + aux_weight * pred1
        else:
            pred = self.aux_task_iql(
                lstm_o,
                batch.obs["own_hand"],
                batch.seq_len,
                rl_loss.size(),
                stat,
            )
            loss = rl_loss + aux_weight * pred

        return loss, priority, online_q

    def behavior_clone_loss(self, online_q, batch, t, clone_bot, stat):
        max_seq_len = batch.obs["priv_s"].size(0)
        priv_s = batch.obs["priv_s"]
        publ_s = batch.obs["publ_s"]
        legal_move = batch.obs["legal_move"]

        bsize, num_player = priv_s.size(1), 1
        if self.vdn:
            num_player = priv_s.size(2)
            priv_s = priv_s.flatten(1, 2)
            publ_s = publ_s.flatten(1, 2)
            legal_move = legal_move.flatten(1, 2)

        with torch.no_grad():
            target_logit, _ = clone_bot(priv_s, publ_s, None)
            target_logit = target_logit - (1 - legal_move) * 1e10
            target = F.softmax(target_logit, 2)

        logit = online_q / t
        # logit: [seq_len, batch * num_player, num_action]
        legal_logit = logit - (1 - legal_move) * 1e10
        log_distq = F.log_softmax(legal_logit, 2)

        assert log_distq.size() == target.size()
        assert log_distq.size() == legal_move.size()
        xent = -(target.detach() * log_distq).sum(2) / legal_move.sum(2).clamp(min=1e-3)
        if self.vdn:
            xent = xent.view(max_seq_len, bsize, num_player).sum(2)

        mask = torch.arange(0, max_seq_len, device=batch.seq_len.device)
        mask = (mask.unsqueeze(1) < batch.seq_len.unsqueeze(0)).float()
        assert xent.size() == mask.size()
        xent = xent * mask
        xent = xent.sum(0)
        stat["bc_loss"].feed(xent.mean().detach())
        return xent
