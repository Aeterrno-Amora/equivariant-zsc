# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#
# model saver that saves top-k performing model
import os
import torch


class TopkSaver:
    def __init__(self, work_dir, topk):
        self.work_dir = work_dir
        self.topk = topk
        self.worse_perf = -float("inf")
        self.worse_perf_idx = 0
        self.perfs = [self.worse_perf]

        if not os.path.exists(work_dir):
            os.makedirs(work_dir)

    def save(self, model, state_dict, perf, save_latest=False, force_save_name=None):
        if force_save_name is not None:
            model_name = "%s.pthm" % force_save_name
            weight_name = "%s.pthw" % force_save_name
            if model is not None:
                model.save(os.path.join(self.work_dir, model_name))
            if state_dict is not None:
                torch.save(state_dict, os.path.join(self.work_dir, weight_name))

        if save_latest:
            model_name = "latest.pthm"
            weight_name = "latest.pthw"
            if model is not None:
                model.save(os.path.join(self.work_dir, model_name))
            if state_dict is not None:
                torch.save(state_dict, os.path.join(self.work_dir, weight_name))

        if perf <= self.worse_perf:
            # print('i am sorry')
            # [print(i) for i in self.perfs]
            return False

        model_name = "model%i.pthm" % self.worse_perf_idx
        weight_name = "model%i.pthw" % self.worse_perf_idx
        if model is not None:
            model.save(os.path.join(self.work_dir, model_name))
        if state_dict is not None:
            torch.save(state_dict, os.path.join(self.work_dir, weight_name))

        if len(self.perfs) < self.topk:
            self.perfs.append(perf)
            return True

        # neesd to replace
        self.perfs[self.worse_perf_idx] = perf
        worse_perf = self.perfs[0]
        worse_perf_idx = 0
        for i, perf in enumerate(self.perfs):
            if perf < worse_perf:
                worse_perf = perf
                worse_perf_idx = i

        self.worse_perf = worse_perf
        self.worse_perf_idx = worse_perf_idx
        return True
