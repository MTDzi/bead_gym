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
class Reward {

  using BeadType = beads_gym::beads::Bead<Eigen2or3dVector>;

  public:
      Reward(std::vector<std::shared_ptr<BeadType>>& reference_beads) : reference_beads_{reference_beads} {};
      ~Reward() = default;

      double calculate_reward(std::vector<std::shared_ptr<BeadType>>& environment_beads) {
        // double sum_{0.0};
        // for (auto& bead : environment_beads) {
        //   sum_ += bead->get_velocity().norm();
        // }
        // return sum_;
        double reward_ = 1.0;
        double preferable_z_positions[] = {0, 1};
        for (size_t i = 0; i < reference_beads_.size(); i++) {
            double bead_z = environment_beads[i]->get_position().z();
            reward_ -= std::min(std::abs(bead_z - preferable_z_positions[i]), 0.5);
        }
        return reward_;
      };

  private:
    std::vector<std::shared_ptr<BeadType>> reference_beads_;
};

} // namespace beads_gym.environment