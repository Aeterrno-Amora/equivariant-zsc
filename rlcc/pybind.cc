// Copyright (c) Facebook, Inc. and its affiliates.
// All rights reserved.
//
// This source code is licensed under the license found in the
// LICENSE file in the root directory of this source tree.
//
#include <pybind11/pybind11.h>

#include "hanabi-learning-environment/hanabi_lib/canonical_encoders.h"
#include "hanabi-learning-environment/hanabi_lib/hanabi_card.h"
#include "hanabi-learning-environment/hanabi_lib/hanabi_game.h"
#include "hanabi-learning-environment/hanabi_lib/hanabi_hand.h"
#include "hanabi-learning-environment/hanabi_lib/hanabi_move.h"
#include "hanabi-learning-environment/hanabi_lib/hanabi_observation.h"

#include "rlcc/clone_data_generator.h"
#include "rlcc/hanabi_env.h"
#include "rlcc/thread_loop.h"

namespace py = pybind11;
using namespace pybind11::literals;
using namespace hanabi_learning_env;

PYBIND11_MODULE(hanalearn, m) {
  py::class_<HanabiEnv, std::shared_ptr<HanabiEnv>>(m, "HanabiEnv")
      .def(py::init<const std::unordered_map<std::string, std::string>&, int, bool>(),
           "game_params"_a, "max_len"_a = -1, "verbose"_a = false)
      .def("feature_size", &HanabiEnv::featureSize, "sad"_a = false)
      .def("num_action", &HanabiEnv::numAction)
      .def("reset", &HanabiEnv::reset)
      .def("reset_with_deck", &HanabiEnv::resetWithDeck, "deck"_a)
      .def("step", &HanabiEnv::step, "move"_a)
      .def("terminated", &HanabiEnv::terminated)
      .def("get_current_player", &HanabiEnv::getCurrentPlayer)
      .def("last_episode_score", &HanabiEnv::lastEpisodeScore)
      .def("deck_history", &HanabiEnv::deckHistory)
      .def("get_num_players", &HanabiEnv::getNumPlayers)
      .def("get_score", &HanabiEnv::getScore)
      .def("get_life", &HanabiEnv::getLife)
      .def("get_info", &HanabiEnv::getInfo)
      .def("get_fireworks", &HanabiEnv::getFireworks)
      .def("get_hle_state", &HanabiEnv::getHleState)
      .def("get_move", &HanabiEnv::getMove, "uid"_a)
      .def("get_obs_show_cards", &HanabiEnv::getObsShowCards)
      .def("get_last_action", &HanabiEnv::getLastAction)
      .def("get_step", &HanabiEnv::numStep)
      .def("set_color_reward", &HanabiEnv::setColorReward, "color_reward"_a);

  py::class_<CloneDataGenerator, std::shared_ptr<CloneDataGenerator>>(
      m, "CloneDataGenerator")
      .def(py::init<std::shared_ptr<rela::RNNPrioritizedReplay>, int, int, bool, bool, int>(),
           "replay_buffer"_a, "num_player"_a, "max_len"_a, "shuffle_color"_a, "trinary"_a, "num_thread"_a)
      .def("set_game_params", &CloneDataGenerator::setGameParams, "game_params"_a)
      .def("add_game", &CloneDataGenerator::addGame, "deck"_a, "moves"_a)
      .def("start_data_generation", &CloneDataGenerator::startDataGeneration, "inf_loop"_a, "seed"_a)
      .def("terminate", &CloneDataGenerator::terminate);

  py::class_<R2D2Actor, std::shared_ptr<R2D2Actor>>(m, "R2D2Actor")
      .def(py::init<std::shared_ptr<rela::BatchRunner>, int, int, int,
                    const std::vector<float>&, const std::vector<float>&,
                    bool, bool, bool, bool, bool,
                    std::shared_ptr<rela::RNNPrioritizedReplay>,
                    int, int, float>(),
           "runner"_a, "seed"_a, "num_player"_a, "player_idx"_a,
           "eps_list"_a, "temp_list"_a, // list of eps and temperatures to sample from
           "vdn"_a, "sad"_a, "shuffle_color"_a, // shuffle_color for other-play
           "hide_action"_a, "trinary"_a, // trinary aux or full aux
           "replay_buffer"_a, // if replay buffer is None, then all params below are not used
           "multi_step"_a, "seq_len"_a, "gamma"_a)
      // simpler constructor for eval mode
      .def(py::init<std::shared_ptr<rela::BatchRunner>, int, int, bool, bool, bool>(),
           "runner"_a, "num_player"_a, "player_idx"_a, "vdn"_a, "sad"_a, "hide_action"_a)
      .def("set_partners", &R2D2Actor::setPartners, "partners"_a)
      .def("set_belief_runner", &R2D2Actor::setBeliefRunner, "belief_model"_a)
      .def("get_success_fict_rate", &R2D2Actor::getSuccessFictRate)
      .def("get_played_card_info", &R2D2Actor::getPlayedCardInfo);

  m.def("observe", py::overload_cast<const hle::HanabiState&, int, bool>(&observe),
        "state"_a, "player_idx"_a, "hide_action"_a);

  m.def("observe_op",
        py::overload_cast<const hle::HanabiState&, int,
                          bool, const std::vector<int>&, const std::vector<int>&,
                          bool, bool, bool>(&observe),
        "state"_a, "player_idx"_a,
        "shuffle_color"_a, "color_permute"_a, "inv_color_permute"_a,
        "hide_action"_a, "trinary"_a, "sad"_a);

  m.def("observe_sad", &observeSAD, "state"_a, "player_idx"_a);

  py::class_<HanabiThreadLoop, rela::ThreadLoop, std::shared_ptr<HanabiThreadLoop>>(
      m, "HanabiThreadLoop")
      .def(py::init<std::vector<std::shared_ptr<HanabiEnv>>,
                    std::vector<std::vector<std::shared_ptr<R2D2Actor>>>, bool>(),
           "envs"_a, "actors"_a, "eval_mode"_a);

  // bind some hanabi util classes from hanabi-learning-environment
  py::class_<HanabiCard>(m, "HanabiCard")
      .def(py::init<int, int, int>(), "color"_a, "rank"_a, "id"_a)
      .def("color", &HanabiCard::Color)
      .def("rank", &HanabiCard::Rank)
      .def("id", &HanabiCard::Id)
      .def("is_valid", &HanabiCard::IsValid)
      .def("to_string", &HanabiCard::ToString)
      .def(py::pickle(
          [](const HanabiCard& c) {
            // __getstate__
            return py::make_tuple(c.Color(), c.Rank(), c.Id());
          },
          [](py::tuple t) {
            // __setstate__
            if (t.size() != 3) {
              throw std::runtime_error("Invalid state!");
            }
            HanabiCard c(t[0].cast<int>(), t[1].cast<int>(), t[2].cast<int>());
            return c;
          }));

  py::class_<HanabiCardValue>(m, "HanabiCardValue")
      .def(py::init<int, int>())
      .def("color", &HanabiCardValue::Color)
      .def("rank", &HanabiCardValue::Rank)
      .def("is_valid", &HanabiCardValue::IsValid)
      .def("to_string", &HanabiCardValue::ToString)
      .def(py::pickle(
          [](const HanabiCardValue& c) {
            // __getstate__
            return py::make_tuple(c.Color(), c.Rank());
          },
          [](py::tuple t) {
            // __setstate__
            if (t.size() != 2) {
              throw std::runtime_error("Invalid state!");
            }
            HanabiCardValue c(t[0].cast<int>(), t[1].cast<int>());
            return c;
          }));

  py::class_<HanabiHistoryItem>(m, "HanabiHistoryItem")
      .def(py::init<HanabiMove>(), "move"_a)
      .def_readwrite("move", &HanabiHistoryItem::move)
      .def_readwrite("player", &HanabiHistoryItem::player)
      .def_readwrite("scored", &HanabiHistoryItem::scored)
      .def_readwrite("information_token", &HanabiHistoryItem::information_token)
      .def_readwrite("color", &HanabiHistoryItem::color)
      .def_readwrite("rank", &HanabiHistoryItem::rank)
      .def_readwrite("reveal_bitmask", &HanabiHistoryItem::reveal_bitmask)
      .def_readwrite(
          "newly_revealed_bitmask", &HanabiHistoryItem::newly_revealed_bitmask);

  auto hanabiHand = py::class_<HanabiHand>(m, "HanabiHand")
      .def(py::init<>())
      .def("cards", &HanabiHand::Cards)
      .def("knowledge_", &HanabiHand::Knowledge_, py::return_value_policy::reference)
      .def("knowledge", &HanabiHand::Knowledge)
      .def("add_card", &HanabiHand::AddCard, "card"_a, "initial_knowledge"_a)
      .def("remove_from_hand", &HanabiHand::RemoveFromHand, "card_index"_a, "discard_pile"_a)
      .def("to_string", &HanabiHand::ToString);

  py::class_<HanabiHand::CardKnowledge>(hanabiHand, "CardKnowledge")
      .def(py::init<int, int>(), "num_colors"_a, "num_ranks"_a)
      .def("num_colors", &HanabiHand::CardKnowledge::NumColors)
      .def("color_hinted", &HanabiHand::CardKnowledge::ColorHinted)
      .def("color", &HanabiHand::CardKnowledge::Color)
      .def("color_plausible", &HanabiHand::CardKnowledge::ColorPlausible, "color"_a)
      .def("apply_is_color_hint", &HanabiHand::CardKnowledge::ApplyIsColorHint, "color"_a)
      .def("apply_is_not_color_hint", &HanabiHand::CardKnowledge::ApplyIsNotColorHint, "color"_a)
      .def("num_ranks", &HanabiHand::CardKnowledge::NumRanks)
      .def("rank_hinted", &HanabiHand::CardKnowledge::RankHinted)
      .def("rank", &HanabiHand::CardKnowledge::Rank)
      .def("rank_plausible", &HanabiHand::CardKnowledge::RankPlausible, "rank"_a)
      .def("apply_is_rank_hint", &HanabiHand::CardKnowledge::ApplyIsRankHint, "rank"_a)
      .def("apply_is_not_rank_hint", &HanabiHand::CardKnowledge::ApplyIsNotRankHint, "rank"_a)
      .def("is_card_plausible", &HanabiHand::CardKnowledge::IsCardPlausible, "color"_a, "rank"_a)
      .def("to_string", &HanabiHand::CardKnowledge::ToString);

  py::class_<HanabiGame>(m, "HanabiGame")
      .def(py::init<const std::unordered_map<std::string, std::string>&>(), "params"_a)
      .def("max_moves", &HanabiGame::MaxMoves)
      .def("get_move_uid",
           (int (HanabiGame::*)(HanabiMove) const) &HanabiGame::GetMoveUid, "move"_a)
      .def("get_move", &HanabiGame::GetMove, "uid"_a)
      .def("num_colors", &HanabiGame::NumColors)
      .def("num_ranks", &HanabiGame::NumRanks)
      .def("hand_size", &HanabiGame::HandSize)
      .def("max_information_tokens", &HanabiGame::MaxInformationTokens)
      .def("max_life_tokens", &HanabiGame::MaxLifeTokens)
      .def("max_deck_size", &HanabiGame::MaxDeckSize);

  py::class_<HanabiState>(m, "HanabiState")
      .def(py::init<const HanabiGame*, int>(), "game"_a, "start_player"_a = -1)
      .def("hands", py::overload_cast<>(&HanabiState::Hands, py::const_))
      .def("apply_move", &HanabiState::ApplyMove, "move"_a)
      .def("cur_player", &HanabiState::CurPlayer)
      .def("score", &HanabiState::Score)
      .def("max_possible_score", &HanabiState::MaxPossibleScore)
      .def("info_tokens", &HanabiState::InformationTokens)
      .def("life_tokens", &HanabiState::LifeTokens)
      .def("to_string", &HanabiState::ToString)
      .def("is_terminal", &HanabiState::IsTerminal);

  auto hanabiMove = py::class_<HanabiMove>(m, "HanabiMove")
      .def(py::init<HanabiMove::Type, int8_t, int8_t, int8_t, int8_t>(),
           "move_type"_a, "card_index"_a, "target_offset"_a, "color"_a, "rank"_a)
      .def("move_type", &HanabiMove::MoveType)
      .def("target_offset", &HanabiMove::TargetOffset)
      .def("card_index", &HanabiMove::CardIndex)
      .def("color", &HanabiMove::Color)
      .def("rank", &HanabiMove::Rank)
      .def("to_string", &HanabiMove::ToString)
      .def("set_color", &HanabiMove::SetColor, "color"_a)
      .def(py::pickle(
          [](const HanabiMove& m) {
            // __getstate__
            return py::make_tuple(
                m.MoveType(), m.CardIndex(), m.TargetOffset(), m.Color(), m.Rank());
          },
          [](py::tuple t) {
            // __setstate__
            if (t.size() != 5) {
              throw std::runtime_error("Invalid state!");
            }
            HanabiMove m(
                t[0].cast<HanabiMove::Type>(),
                t[1].cast<int8_t>(),
                t[2].cast<int8_t>(),
                t[3].cast<int8_t>(),
                t[4].cast<int8_t>());
            return m;
          }));

  py::enum_<HanabiMove::Type>(hanabiMove, "Type")
      .value("Invalid", HanabiMove::Type::kInvalid)
      .value("Play", HanabiMove::Type::kPlay)
      .value("Discard", HanabiMove::Type::kDiscard)
      .value("RevealColor", HanabiMove::Type::kRevealColor)
      .value("RevealRank", HanabiMove::Type::kRevealRank)
      .value("Deal", HanabiMove::Type::kDeal);

  py::class_<HanabiObservation>(m, "HanabiObservation")
      .def(py::init<const HanabiState&, int, bool>(),
           "state"_a, "observing_player"_a, "show_cards"_a = false)
      .def(py::init<int, int,
                    const std::vector<HanabiHand>&,
                    const std::vector<HanabiCard>&,
                    const std::vector<int>&,
                    const std::vector<HanabiHistoryItem>&,
                    int, int, int,
                    const std::vector<HanabiMove>&,
                    const HanabiGame*>(),
           "cur_player"_a, "observing_player"_a,
           "hands"_a, "discard_pile"_a, "fireworks"_a, "last_moves"_a,
           "deck_size"_a, "information_tokens"_a, "life_tokens"_a,
           "legal_moves"_a, "parent_game"_a)
      .def("legal_moves", &HanabiObservation::LegalMoves)
      .def("last_moves", &HanabiObservation::LastMoves)
      .def("life_tokens", &HanabiObservation::LifeTokens)
      .def("information_tokens", &HanabiObservation::InformationTokens)
      .def("deck_size", &HanabiObservation::DeckSize)
      .def("fireworks", &HanabiObservation::Fireworks)
      .def("card_playable_on_fireworks",
           [](HanabiObservation& obs, int color, int rank) {
             return obs.CardPlayableOnFireworks(color, rank);
           })
      .def("discard_pile", &HanabiObservation::DiscardPile)
      .def("hands", &HanabiObservation::Hands);

  py::class_<CanonicalObservationEncoder>(m, "ObservationEncoder")
      .def(py::init<const HanabiGame*>(), "parent_game"_a)
      .def("shape", &CanonicalObservationEncoder::Shape)
      .def("encode", &CanonicalObservationEncoder::Encode,
           "obs"_a, "show_own_cards"_a, "order"_a,
           "shuffle_color"_a, "color_permute"_a, "inv_color_permute"_a, "hide_action"_a);
}
