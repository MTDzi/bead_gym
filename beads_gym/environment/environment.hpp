#pragma once

#include <iostream>
#include <memory>
#include <vector>
#include <map>
#include <cassert>

#include <Eigen/Dense>
#include "beads_gym/beads/bead.hpp"
#include "beads_gym/bonds/bond.hpp"
#include "beads_gym/environment/integrator/integrator.hpp"
#include "beads_gym/environment/reward/reward.hpp"

namespace beads_gym::environment {

template <typename Eigen2or3dVector>
class Environment {

  using BeadType = beads_gym::beads::Bead<Eigen2or3dVector>;
  using BondType = beads_gym::bonds::Bond<Eigen2or3dVector>;
  using RewardType = beads_gym::environment::reward::Reward<Eigen2or3dVector>;

  public:
      Environment() : integrator_{integrator::Integrator<Eigen2or3dVector>(0.01)} {};
      ~Environment() = default;

      void add_bead(std::shared_ptr<BeadType> bead) {
        assert(beads_map_.find(bead->get_id()) == beads_map_.end() && "Bead with this ID already exists in the map!");
        beads_.push_back(bead);
        beads_map_[bead->get_id()] = bead;
      }

      void add_bond(std::shared_ptr<BondType> bond) {
        bond->set_bead_1(beads_map_[bond->bead_1_id()]);
        bond->set_bead_2(beads_map_[bond->bead_2_id()]);
        bonds_.push_back(bond);
      }

      void add_reward(std::shared_ptr<RewardType> reward) {
        rewards_.push_back(reward);
      }

      void step(std::map<size_t, std::vector<double>>& action) {
        for (auto& bond : bonds_) {
          bond->apply_forces();
        }

        // Now for the actuators
        for (auto& bead_id : action) {
          auto bead = beads_map_[bead_id.first];
          bead->add_force(
            bead->get_mass() * Eigen2or3dVector{bead_id.second[0], bead_id.second[1], bead_id.second[2]}
          );
        }

        // Now for the gravity
        for (auto& bead : beads_) {
          bead->add_force(Eigen2or3dVector{0.0, 0.0, -gravity_ * bead->get_mass()});
        }

        // Finally, we can step the integrator
        integrator_.step(beads_);
      }
      
      void reset() { std::cout << "Environment reset" << std::endl; }

      std::vector<std::shared_ptr<BeadType>> get_beads() const { return beads_; }

    private:
        std::vector<std::shared_ptr<BeadType>> beads_;
        std::map<size_t, std::shared_ptr<BeadType>> beads_map_;
        std::vector<std::shared_ptr<BondType>> bonds_;
        integrator::Integrator<Eigen2or3dVector> integrator_;
        std::vector<std::shared_ptr<RewardType>> rewards_;
        constexpr static double gravity_ = 9.81;
};

} // namespace beads_gym.environment