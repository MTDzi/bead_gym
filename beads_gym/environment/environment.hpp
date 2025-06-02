#pragma once

#include <iostream>
#include <sstream>
#include <cmath>
#include <random>
#include <memory>
#include <vector>
#include <map>
#include <cassert>

#include <Eigen/Dense>

#include "beads_gym/beads/bead.hpp"
#include "beads_gym/beads/bead_group.hpp"
#include "beads_gym/bonds/bond.hpp"
#include "beads_gym/environment/integrator/integrator.hpp"
#include "beads_gym/environment/reward/stay_close_reward.hpp"

namespace beads_gym::environment {

template <typename Eigen2or3dVector>
class Environment {

  using BeadType = beads_gym::beads::Bead<Eigen2or3dVector>;
  using BondType = beads_gym::bonds::Bond<Eigen2or3dVector>;
  using RewardType = beads_gym::environment::reward::StayCloseReward<Eigen2or3dVector>;
  using BeadGroupType = beads_gym::beads::BeadGroup<Eigen2or3dVector>;

  public:
      Environment(double internal_timestep, int num_internal_steps = 1, double theta = 0.0, double sigma=0.0) 
      : integrator_{integrator::Integrator<Eigen2or3dVector>(internal_timestep)},
        dt_{internal_timestep},
        num_internal_steps_{num_internal_steps}, 
        theta_{theta},
        sigma_{sigma} {
          std::random_device rd;
          gen = std::mt19937(rd());
          dist = std::normal_distribution<>{0.0, 1.0};
          sqrt_dt_ = std::sqrt(dt_);
          prev_wind_ = Eigen2or3dVector::Zero();
          bonds_within_bead_group_resolved_ = false;
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

      void add_bead_group(std::vector<size_t> bead_ids) {
        pre_bead_groups_.push_back(bead_ids);
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
        if (bonds_within_bead_group_resolved_ == false) {
          resolve_bonds_within_bead_groups();
          resolve_bead_groups();

          bonds_within_bead_group_resolved_ = true;
        }

        for (int internal_step=0; internal_step<num_internal_steps_; ++internal_step) {
          // First, calculate the forces coming from the bonds themselves
          for (auto& bond : bonds_) {
            // TODO: worth parallelizing if number of bonds is large
            bond->apply_forces();
          }

          // Add gravity
          for (auto& bead : beads_) {
            bead->add_force(Eigen2or3dVector{0.0, 0.0, -gravity_ * bead->get_mass()});
          }

          // Add wind (aka Ohrstein-Uhlenbeck process)
          auto curr_wind = calc_ohrstein_uhlenbeck_wind();
          for (auto& bead : beads_) {
            bead->add_force(curr_wind);
          }

          // TODO: can be done just once
          for (auto& bead : beads_) {
            bead->save_external_force();
          }

          // Now for the actuators
          for (auto& bead_id : action) {
            auto bead = beads_map_[bead_id.first];
            bead->add_force(
              bead->get_mass() * Eigen2or3dVector{bead_id.second[0], bead_id.second[1], bead_id.second[2]}
            );
          }

          // And with all forces in place, we can step the integrator
          integrator_.step(beads_);
          integrator_.group_step(bead_groups_);

          // Which we followup with zeroing out of the forces to start from scratch in the
          // next iteration
          for (auto& bead : beads_) {
            bead->zero_out_force();
          }
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
      std::vector<std::shared_ptr<BondType>> get_bonds() const { return bonds_; }

      double calc_bond_potential() {
        double potential_{0.0};
        for (auto& bond : bonds_) {
          potential_ += bond->potential();
        }
        return potential_;
      }

      Eigen2or3dVector calc_ohrstein_uhlenbeck_wind() {
        auto curr_wind = prev_wind_ * (1.0 - theta_) * dt_ + sigma_ * random_normal_vector() * sqrt_dt_;
        prev_wind_ = curr_wind;
        return curr_wind;
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
      Eigen2or3dVector prev_wind_;
      std::mt19937 gen;
      std::normal_distribution<> dist;
      std::vector<std::shared_ptr<RewardType>> reward_calculators_;
      bool bonds_within_bead_group_resolved_;
      std::vector<std::vector<size_t>> pre_bead_groups_;
      std::vector<BeadGroupType> bead_groups_;

      Eigen2or3dVector random_normal_vector() {
        Eigen2or3dVector norm_vec = Eigen2or3dVector::Zero();
        for (int i=0; i<norm_vec.size(); ++i) {
          norm_vec(i) = dist(gen);
        }
        return norm_vec;
      }

      void resolve_bead_groups() {
        for (auto& bead_ids : pre_bead_groups_) {
          std::vector<std::shared_ptr<BeadType>> local_beads;
          for (size_t bead_id : bead_ids) {
            if (beads_map_.find(bead_id) == beads_map_.end()) {
              std::stringstream msg;
              msg << "Bead " << bead_id << " from a bead group not present in the environment";
              throw std::runtime_error(msg.str());
            } 
            local_beads.push_back(beads_map_[bead_id]);
          }
          bead_groups_.push_back(std::move(local_beads));  
        }
      }

      void resolve_bonds_within_bead_groups() {
        std::vector<size_t> bond_indices_to_remove;
        for (size_t bond_id=0; bond_id<bonds_.size(); bond_id++) {
          std::shared_ptr<BondType> bond = bonds_[bond_id];
          size_t bead_1_id = bond->bead_1_id();
          size_t bead_2_id = bond->bead_2_id();
          for (auto& bead_group : pre_bead_groups_) {
            bool bead_1_in_group = (bead_group.end() != std::find(bead_group.begin(), bead_group.end(), bead_1_id));
            bool bead_2_in_group = (bead_group.end() != std::find(bead_group.begin(), bead_group.end(), bead_2_id));
            if (bead_1_in_group && bead_2_in_group) {
              bond_indices_to_remove.push_back(bond_id);
            }
          }
        }
        std::sort(bond_indices_to_remove.rbegin(), bond_indices_to_remove.rend());
        for (size_t index_to_remove : bond_indices_to_remove) {
          bonds_.erase(bonds_.begin() + index_to_remove);
        }
      }
};

} // namespace beads_gym.environment