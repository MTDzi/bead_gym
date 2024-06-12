#pragma once

#include <iostream>
#include <string>
#include <utility>
#include <Eigen/Dense>
#include <vector>

namespace beads_gym::beads {

template <typename Eigen2or3dVector>
class Bead {
  public:
      Bead() = delete;
      Bead(size_t id, std::vector<double> &position) : id_{id}, position_{Eigen2or3dVector{position.data()}} {}
      Bead(size_t id, Eigen2or3dVector position) : id_{id}, position_{position} {}
      Bead(const Bead&) = default;
      Bead(Bead&&) = default;
      Bead& operator=(const Bead&) = default;
      Bead& operator=(Bead&&) = default;
      ~Bead() = default;
    
      void set_position(const Eigen2or3dVector& position) { position_ = position; }
      Eigen2or3dVector get_position() const { return position_; }

      size_t get_id() const { return id_; }
        
  protected:
      size_t id_;
      Eigen2or3dVector position_;
  };
} // namespace beads_gym.bead
