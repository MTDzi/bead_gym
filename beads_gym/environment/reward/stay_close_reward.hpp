#pragma once

#include <iostream>
#include <memory>
#include <vector>
#include <map>
#include <cassert>

#include <Eigen/Dense>
#include "beads_gym/beads/bead.hpp"

namespace beads_gym::environment::reward {

template <typename Eigen2or3dVector>
class StayCloseReward {

  using BeadType = beads_gym::beads::Bead<Eigen2or3dVector>;

  public:
      StayCloseReward(std::map<size_t, std::vector<double>>& reference_positions, std::vector<double>& weights) {
        for (auto& reference_position : reference_positions) {
          reference_positions_[reference_position.first] = Eigen2or3dVector{reference_position.second.data()};
        }
        assert(
          (weights.size() == reference_positions.size())
          && "Should be as many weights as reference positions"
        );
        weights_ = weights;
      };
      StayCloseReward(std::vector<std::shared_ptr<BeadType>>& reference_beads, std::vector<double> weights) {
        size_t index = 0;
        for (auto& reference_bead : reference_beads) {
          reference_positions_[index] = reference_bead->get_position();
          ++index;
        }
        assert(
          (weights.size() == reference_beads.size())
          && "Should be as many weights as reference beads"
        );
        weights_ = weights;
      };
      ~StayCloseReward() = default;

      double calculate_reward(std::vector<std::shared_ptr<BeadType>>& environment_beads) {
        double reward_{0.0};
        size_t weight_idx = 0;
        for (auto& bead : environment_beads) {
          if (reference_positions_.find(bead->get_id()) != reference_positions_.end()) {
            reward_ -= weights_[weight_idx] * (bead->get_position() - reference_positions_[bead->get_id()]).norm();
          }
          ++weight_idx;
        }
        return reward_;
      };

  private:
    std::map<size_t, Eigen::Vector3d> reference_positions_;
    std::vector<double> weights_;
};

} // namespace beads_gym.environment