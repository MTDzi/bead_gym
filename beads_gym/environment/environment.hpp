#pragma once

#include <iostream>
#include <memory>
#include <vector>
#include <map>

#include "beads_gym/beads/bead.hpp"
#include "beads_gym/bonds/bond.hpp"
#include "beads_gym/environment/integrator/integrator.hpp"

namespace beads_gym::environment {

template <typename Eigen2or3dVector>
class Environment {

  using BeadType = beads_gym::beads::Bead<Eigen2or3dVector>;
  using BondType = beads_gym::bonds::Bond<Eigen2or3dVector>;

  public:
      Environment() : integrator_{integrator::Integrator<Eigen2or3dVector>(0.01)} {};
      Environment(std::vector<std::shared_ptr<BeadType>> &beads)
      : beads_{beads}, integrator_{integrator::Integrator<Eigen2or3dVector>(0.01)} {}
      ~Environment() = default;
    

      void add_bead(const std::shared_ptr<BeadType> &bead) {
        beads_.push_back(bead);
        beads_map_[bead->get_id()] = bead;
      }

      void add_bond(const std::shared_ptr<BondType> &bond) {
        bond->set_bead_1(beads_map_[bond->bead_1_id()]);
        bond->set_bead_2(beads_map_[bond->bead_2_id()]);
        bonds_.push_back(bond);
      }
      
      void step() { std::cout << "Environment step" << std::endl; }
      void reset() { std::cout << "Environment reset" << std::endl; }

      std::vector<std::shared_ptr<BeadType>> get_beads() const { return beads_; }

    private:
        std::vector<std::shared_ptr<BeadType>> beads_;
        std::map<size_t, std::shared_ptr<BeadType>> beads_map_;
        std::vector<std::shared_ptr<BondType>> bonds_;
        integrator::Integrator<Eigen2or3dVector> integrator_;
};

} // namespace beads_gym.environment