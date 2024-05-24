// Copyright (c) Facebook, Inc. and its affiliates.
// All rights reserved.
//
// This source code is licensed under the license found in the
// LICENSE file in the root directory of this source tree.
//
#include <pybind11/pybind11.h>
#include <torch/extension.h>

#include "rela/batch_runner.h"
#include "rela/context.h"
#include "rela/prioritized_replay.h"
#include "rela/thread_loop.h"
#include "rela/transition.h"

namespace py = pybind11;
using namespace pybind11::literals;
using namespace rela;

PYBIND11_MODULE(rela, m) {
  py::class_<RNNTransition, std::shared_ptr<RNNTransition>>(m, "RNNTransition")
      .def_readwrite("obs", &RNNTransition::obs)
      .def_readwrite("h0", &RNNTransition::h0)
      .def_readwrite("action", &RNNTransition::action)
      .def_readwrite("reward", &RNNTransition::reward)
      .def_readwrite("terminal", &RNNTransition::terminal)
      .def_readwrite("bootstrap", &RNNTransition::bootstrap)
      .def_readwrite("seq_len", &RNNTransition::seqLen);

  py::class_<RNNPrioritizedReplay, std::shared_ptr<RNNPrioritizedReplay>>(
      m, "RNNPrioritizedReplay")
      .def(py::init<int, int, float, float, int>(),
           "capacity"_a, "seed"_a, "alpha"_a, "beta"_a, "prefetch"_a)
           // alpha: priority exponent
           // beta: importance sampling exponent
      .def("clear", &RNNPrioritizedReplay::clear)
      .def("terminate", &RNNPrioritizedReplay::terminate)
      .def("size", &RNNPrioritizedReplay::size)
      .def("num_add", &RNNPrioritizedReplay::numAdd)
      .def("sample", &RNNPrioritizedReplay::sample, "batchsize"_a, "device"_a)
      .def("update_priority", &RNNPrioritizedReplay::updatePriority, "priority"_a)
      .def("get", &RNNPrioritizedReplay::get, "idx"_a);

  py::class_<TensorDictReplay, std::shared_ptr<TensorDictReplay>>(m, "TensorDictReplay")
      .def(py::init<int, int, float, float, int>(),
           "capacity"_a, "seed"_a, "alpha"_a, "beta"_a, "prefetch"_a)
           // alpha: priority exponent
           // beta: importance sampling exponent
      .def("size", &TensorDictReplay::size)
      .def("num_add", &TensorDictReplay::numAdd)
      .def("sample", &TensorDictReplay::sample, "batchsize"_a, "device"_a)
      .def("update_priority", &TensorDictReplay::updatePriority, "priority"_a)
      .def("get", &TensorDictReplay::get, "idx"_a);

  py::class_<ThreadLoop, std::shared_ptr<ThreadLoop>>(m, "ThreadLoop");

  py::class_<Context>(m, "Context")
      .def(py::init<>())
      .def("push_thread_loop", &Context::pushThreadLoop, py::keep_alive<1, 2>(), "env"_a)
      .def("start", &Context::start)
      .def("pause", &Context::pause)
      .def("resume", &Context::resume)
      .def("join", &Context::join)
      .def("terminated", &Context::terminated);

  py::class_<BatchRunner, std::shared_ptr<BatchRunner>>(m, "BatchRunner")
      .def(py::init<py::object, const std::string&, int, const std::vector<std::string>&>(),
                    "model"_a, "device"_a, "log_freq"_a, "methods"_a)
      .def(py::init<py::object, const std::string&>())
      .def("add_method", &BatchRunner::addMethod, "method"_a, "batch_size"_a)
      .def("start", &BatchRunner::start)
      .def("stop", &BatchRunner::stop)
      .def("update_model", &BatchRunner::updateModel, "agent"_a)
      .def("set_log_freq", &BatchRunner::setLogFreq, "log_freq"_a);
}
