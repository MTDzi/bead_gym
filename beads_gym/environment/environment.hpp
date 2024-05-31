#pragma once

#include <iostream>
#include <memory>
#include <vector>

#include "beads_gym/beads/bead.hpp"
#include "beads_gym/beads/three_degrees_of_freedom_bead.hpp"

namespace beads_gym::environment {

template <typename Eigen2or3dVector>
class Environment {

  using BeadType = beads_gym::beads::Bead<Eigen2or3dVector>;

  public:
      Environment() = default;
      Environment(std::vector<std::shared_ptr<BeadType>> &beads) : beads_{beads} {}
      Environment(const Environment&) = default;
      Environment(Environment&&) = default;
      Environment& operator=(const Environment&) = default;
      Environment& operator=(Environment&&) = default;
      ~Environment() = default;
    

      void add_bead(const std::shared_ptr<BeadType> &bead) { beads_.push_back(bead); }
      
      void step() { std::cout << "Environment step" << std::endl; }
      void reset() { std::cout << "Environment reset" << std::endl; }

      std::vector<std::shared_ptr<BeadType>> get_beads() const { return beads_; }

    private:
        std::vector<std::shared_ptr<BeadType>> beads_;
};

} // namespace beads_gym.environment