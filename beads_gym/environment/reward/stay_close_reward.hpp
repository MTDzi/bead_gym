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
      StayCloseReward(std::map<size_t, std::vector<double>>& reference_positions) {
        for (auto& reference_position : reference_positions) {
          reference_positions_[reference_position.first] = Eigen2or3dVector{reference_position.second.data()};
        }
      };
      ~StayCloseReward() = default;

      double calculate_reward(std::vector<std::shared_ptr<BeadType>>& environment_beads) {
        double reward_{0.0};
        for (auto& bead : environment_beads) {
          if (reference_positions_.find(bead->get_id()) != reference_positions_.end()) {
            reward_ -= (bead->get_position() - reference_positions_[bead->get_id()]).norm();
          }
        }
        return reward_;
      };

  private:
    std::map<size_t, Eigen::Vector3d> reference_positions_;
};

} // namespace beads_gym.environment