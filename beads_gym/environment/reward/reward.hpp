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
        return 0.0;
      };

  private:
    std::vector<std::shared_ptr<BeadType>> reference_beads_;
};

} // namespace beads_gym.environment