#pragma once

#include <iostream>
#include <memory>
#include <vector>
#include <map>

#include "beads_gym/beads/bead.hpp"

#include "beads_gym/bonds/bond.hpp"

namespace beads_gym::environment::integrator {

template <typename Eigen2or3dVector>
class Integrator {

  using BeadType = beads_gym::beads::Bead<Eigen2or3dVector>;

  public:
      Integrator(double dt) : dt_{dt} {}
      ~Integrator() = default;

      void step(std::vector<std::shared_ptr<BeadType>>& beads) {
        // TODO: Ask copilot if this is the best way to do this, or should we pass this vector by
        // reference?
        for (auto &bead : beads) {
          if (bead->is_mobile()) {
            // Verlet algorithm, https://en.wikipedia.org/wiki/Verlet_integration
            auto curr_position = bead->get_position();
            auto curr_acceleration = bead->get_acceleration();
            auto prev_positon = bead->get_prev_position();
            auto new_position = 2 * curr_position - prev_positon + curr_acceleration * dt_ * dt_;
            auto new_velocity = (new_position - prev_positon) / (2 * dt_);
            bead->set_position(new_position);
            bead->set_velocity(new_velocity);
            bead->zero_out_force();
          }
        }
      }

    private:
        double dt_;
};
} // namespace beads_gym.environment.integrator