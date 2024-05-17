#pragma once

#include <iostream>
#include <vector>

#include "beads_gym/beads/bead.hpp"

namespace beads_gym::environment {

template <typename Eigen2or3dVector>
class Environment {

  using BeadType = beads_gym::beads::Bead<Eigen2or3dVector>;

  public:
      Environment() = default;
      Environment(std::vector<BeadType> &beads) : beads_{beads} {}
      Environment(const Environment&) = default;
      Environment(Environment&&) = default;
      Environment& operator=(const Environment&) = default;
      Environment& operator=(Environment&&) = default;
      ~Environment() = default;
    
      void step() { std::cout << "Environment step" << std::endl; }
      void reset() { std::cout << "Environment reset" << std::endl; }

      std::vector<BeadType> get_beads() const { return beads_; }

    private:
        std::vector<BeadType> beads_;
};

} // namespace beads_gym.environment