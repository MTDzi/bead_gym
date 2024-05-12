#pragma once

#include <iostream>
#include <string>
#include <utility>
#include <Eigen/Dense>

namespace beads_gym::beads {

template <typename Eigen2or3dVector>
class Bead {
  public:
      Bead(Eigen2or3dVector position) : position_{position} {}
      Bead(const Bead&) = default;
      Bead(Bead&&) = default;
      Bead& operator=(const Bead&) = default;
      Bead& operator=(Bead&&) = default;
      ~Bead() = default;
    
      void set_position(const Eigen2or3dVector& position) { position_ = position; }
      Eigen2or3dVector get_position() const { return position_; }
        
  protected:
      Eigen2or3dVector position_;
  };
} // namespace beads_gym.bead
