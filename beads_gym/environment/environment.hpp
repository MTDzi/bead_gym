#pragma once

#include <iostream>
#include <cmath>
#include <random>
#include <memory>
#include <vector>
#include <map>
#include <cassert>

#include <Eigen/Dense>

#include "beads_gym/beads/bead.hpp"
#include "beads_gym/bonds/bond.hpp"
#include "beads_gym/environment/integrator/integrator.hpp"
#include "beads_gym/environment/reward/stay_close_reward.hpp"

namespace beads_gym::environment {

template <typename Eigen2or3dVector>
class Environment {

  using BeadType = beads_gym::beads::Bead<Eigen2or3dVector>;
  using BondType = beads_gym::bonds::Bond<Eigen2or3dVector>;
  using RewardType = beads_gym::environment::reward::StayCloseReward<Eigen2or3dVector>;

  public:
      Environment(double internal_timestep, int num_internal_steps = 1, double theta = 0.0, double sigma=0.0) 
      : integrator_{integrator::Integrator<Eigen2or3dVector>(internal_timestep)},
        dt_{internal_timestep},
        num_internal_steps_{num_internal_steps}, 
        theta_{theta},
        sigma_{sigma} {
          std::random_device rd;
          gen = std::mt19937(rd());
          sqrt_dt_ = std::sqrt(dt_);
          prev_wind = Eigen2or3dVector::Zero();
        };
      ~Environment() = default;

      void add_bead(std::shared_ptr<BeadType> bead) {
        assert(beads_map_.find(bead->get_id()) == beads_map_.end() && "Bead with this ID already exists in the map!");

        beads_.push_back(bead);
        beads_map_[bead->get_id()] = bead;

        auto bead_copy = *bead;
        initial_beads_.push_back(std::make_shared<BeadType>(bead_copy));
        initial_beads_map_[bead->get_id()] = initial_beads_.back();
      }

      void add_bond(std::shared_ptr<BondType> bond) {
        bond->set_bead_1(beads_map_[bond->bead_1_id()]);
        bond->set_bead_2(beads_map_[bond->bead_2_id()]);
        bonds_.push_back(bond);
      }

      void add_reward_calculator(std::shared_ptr<RewardType> reward_calculator) {
        reward_calculators_.push_back(reward_calculator);
      }

      std::vector<double> step(std::map<size_t, std::vector<double>>& action) {
        for (int internal_step=0; internal_step<num_internal_steps_; ++internal_step) {
          // First, calculate the forces coming from the bonds themselves
          for (auto& bond : bonds_) {
            // TODO: worth parallelizing if number of bonds is large
            bond->apply_forces();
          }

          // Now for the actuators
          for (auto& bead_id : action) {
            auto bead = beads_map_[bead_id.first];
            bead->add_force(
              bead->get_mass() * Eigen2or3dVector{bead_id.second[0], bead_id.second[1], bead_id.second[2]}
            );
          }

          // Add gravity
          for (auto& bead : beads_) {
            bead->add_force(Eigen2or3dVector{0.0, 0.0, -gravity_ * bead->get_mass()});
          }

          // Add wind (aka Ohrstein-Uhlenbeck process)
          auto curr_wind = calc_ohrstein_uhlenbeck_wind();
          prev_wind = curr_wind;
          for (auto& bead : beads_) {
            bead->add_force(curr_wind);
          }

          // And with all forces in place, we can step the integrator
          integrator_.step(beads_);
        }

        // Finally, we calculate the rewards
        std::vector<double> rewards;
        for (auto& reward_calculator : reward_calculators_) {
          rewards.push_back(reward_calculator->calculate_reward(beads_));
        }
        return rewards;
      }
      
      void reset() {
          beads_.clear();
          beads_map_.clear();

          // Repopulate beads_ and beads_map_ from initial_beads_ and initial_beads_map_
          for (const auto& bead : initial_beads_) {
              auto bead_copy = std::make_shared<BeadType>(*bead);
              beads_.push_back(bead_copy);
              beads_map_[bead->get_id()] = bead_copy;
          }

          for (auto& bond : bonds_) {
              bond->set_bead_1(beads_map_[bond->bead_1_id()]);
              bond->set_bead_2(beads_map_[bond->bead_2_id()]);
          }
      }

      std::vector<std::shared_ptr<BeadType>> get_beads() const { return beads_; }

      double calc_bond_potential() {
        double potential_{0.0};
        for (auto& bond : bonds_) {
          potential_ += bond->potential();
        }
        return potential_;
      }

      Eigen2or3dVector calc_ohrstein_uhlenbeck_wind() {
        return prev_wind * (1 - theta_) * dt_ + sigma_ * random_normal_vector() * sqrt_dt_;
      }

      Eigen2or3dVector random_normal_vector() {
        Eigen2or3dVector norm_vec = Eigen2or3dVector::Zero();
        return norm_vec;
      }

    private:
        constexpr static double gravity_ = 9.81;
        std::vector<std::shared_ptr<BeadType>> beads_;
        std::map<size_t, std::shared_ptr<BeadType>> beads_map_;
        std::vector<std::shared_ptr<BeadType>> initial_beads_;
        std::map<size_t, std::shared_ptr<BeadType>> initial_beads_map_;
        std::vector<std::shared_ptr<BondType>> bonds_;
        integrator::Integrator<Eigen2or3dVector> integrator_;
        double dt_;
        double sqrt_dt_;
        int num_internal_steps_;
        double theta_;
        double sigma_;
        Eigen2or3dVector prev_wind;
        std::mt19937 gen;
        std::vector<std::shared_ptr<RewardType>> reward_calculators_;
};

} // namespace beads_gym.environment