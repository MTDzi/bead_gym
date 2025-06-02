#pragma once

#include <iostream>
#include <memory>
#include <vector>
#include <map>

#include "beads_gym/beads/bead.hpp"
#include "beads_gym/beads/bead_group.hpp"

#include "beads_gym/bonds/bond.hpp"

namespace beads_gym::environment::integrator {

template <typename Eigen2or3dVector>
class Integrator {

  using BeadType = beads_gym::beads::Bead<Eigen2or3dVector>;
  using BeadGroupType = beads_gym::beads::BeadGroup<Eigen2or3dVector>;

  public:
      Integrator(double dt) : dt_{dt} {}
      ~Integrator() = default;

      void step(std::vector<std::shared_ptr<BeadType>>& beads) {
        // TODO: Ask copilot if this is the best way to do this, or should we pass this vector by
        // reference?
        for (auto &bead : beads) {
          if (bead->is_mobile()) {
            // Verlet algorithm, https://en.wikipedia.org/wiki/Verlet_integration
            auto new_position_and_velocity = _step(
              bead->get_position(),
              bead->get_prev_position(),
              bead->get_acceleration()
            );
            bead->set_position(new_position_and_velocity.first);
            bead->set_velocity(new_position_and_velocity.second);
          }
        }
      }

      void group_step(std::vector<BeadGroupType>& bead_groups) {
        for (auto& bead_group : bead_groups) {
          // Update internal state
          bead_group.update_com_and_inertia_mat();
          bead_group.update_net_force_and_torque();

          // The COM evolution is actually the same as in a "normal" step
          auto new_com_position_and_velocity = _step(
            bead_group.get_com(),
            bead_group.get_com_prev(),
            bead_group.get_com_acceleration()
          );
          bead_group.set_com(new_com_position_and_velocity.first);
          bead_group.set_com_velocity(new_com_position_and_velocity.second);

          // But the rotational part is different
          auto curr_torque_world = bead_group.get_torque_world();
          auto orientation = bead_group.get_orientation();
          auto curr_angular_mom_body = bead_group.get_angular_mom_body();
          auto curr_torque_body = orientation.inverse() * curr_torque_world;
          auto new_angular_mom_body = curr_angular_mom_body + curr_torque_body * dt_; // A basic Euler approach
          bead_group.set_angular_mom_body(new_angular_mom_body);

          auto intertia_mat = bead_group.get_inertia_mat();
          auto new_angular_velocity_body = intertia_mat.inverse() * new_angular_mom_body;
          auto new_angular_velocity_world = orientation * new_angular_velocity_body;
          bead_group.set_angular_vel_world(new_angular_velocity_world);
          auto pure_angular_velocity_world = Eigen::Quaterniond(
            0,
            new_angular_velocity_world.x(),
            new_angular_velocity_world.y(),
            new_angular_velocity_world.z()
          );
          auto new_orientation = pure_angular_velocity_world * orientation;
          new_orientation.w() = orientation.w() + new_orientation.w() * 0.5 * dt_;
          new_orientation.vec() = orientation.vec() + new_orientation.vec() * 0.5 * dt_;
          new_orientation.normalize();
          bead_group.set_orientation(new_orientation);

          // With the new orientation and new_angular_velocity_world, we can add a correction
          // from the rotational part 
          bead_group.update_beads_world_positions();
        }      
      }

      std::pair<Eigen2or3dVector, Eigen2or3dVector> _step(const Eigen2or3dVector& curr_position, const Eigen2or3dVector& prev_position, const Eigen2or3dVector& curr_acceleration) {
          auto new_position = 2 * curr_position - prev_position + curr_acceleration * dt_ * dt_;
          auto new_velocity = (new_position - prev_position) / (2 * dt_);
          return {new_position, new_velocity};
      }

      double get_dt() {
        return dt_;
      }

    private:
        double dt_;
};
} // namespace beads_gym.environment.integrator