#pragma once

#include <iostream>
#include <string>
#include <utility>
#include <Eigen/Dense>
#include <vector>

namespace beads_gym::beads {

template <typename Eigen2or3dVector>
class Bead {
  // TODO: make sure id is unique
  // TODO: re-use constructor 2
  public:
      Bead() = delete;
      Bead(size_t id, std::vector<double> &position, double mass, bool is_mobile, const std::vector<int> &constrained_axes = {})
        : id_(id), mass_(mass), velocity_(Eigen2or3dVector::Zero()), 
          acceleration_(Eigen2or3dVector::Zero()), force_(Eigen2or3dVector::Zero()), external_force_(Eigen2or3dVector::Zero()),
          is_mobile_(is_mobile), constrained_axes_(constrained_axes) {
        // FIXME: re-use the other constructor 
        assert(mass > 0);
        position_ = Eigen2or3dVector(position.data());
        prev_position_ = Eigen2or3dVector(position.data());
        // Initialization logic is centralized here.
      }

      Bead(size_t id, Eigen2or3dVector &position, double mass, bool is_mobile, const std::vector<int> &constrained_axes = {})
        : id_(id), mass_(mass), position_{position}, prev_position_{position},
          velocity_(Eigen2or3dVector::Zero()),
          acceleration_(Eigen2or3dVector::Zero()), force_(Eigen2or3dVector::Zero()), external_force_(Eigen2or3dVector::Zero()),
          is_mobile_(is_mobile), constrained_axes_(constrained_axes) {
        assert(mass > 0);
        // Initialization logic is centralized here.
      }
      Bead(const Bead&) = default;
      Bead(Bead&&) = default;
      Bead& operator=(const Bead&) = default;
      Bead& operator=(Bead&&) = default;
      ~Bead() = default;

      Eigen2or3dVector get_position() const { return position_; }
      void set_position(const Eigen2or3dVector &position) {
        prev_position_ = position_;
        position_ = position;
        for (size_t i = 0; i < constrained_axes_.size(); i++) {
          position_(constrained_axes_[i]) = prev_position_(constrained_axes_[i]);
        }
      }
      Eigen2or3dVector get_prev_position() const { return prev_position_; }
  
      Eigen2or3dVector get_velocity() const { return velocity_; }
      void set_velocity(const Eigen2or3dVector& velocity) { velocity_ = velocity; }
  
      Eigen2or3dVector get_acceleration() const { return force_ / mass_; }
      void add_acceleration(const Eigen2or3dVector& acceleration) { acceleration_ += acceleration; }

      Eigen2or3dVector get_external_acceleration() const { return external_force_ / mass_; }

      Eigen2or3dVector get_force() const { return force_; }
      void add_force(const Eigen2or3dVector& force) { force_ += force; }
      void zero_out_force() { force_ = Eigen2or3dVector::Zero(); }
      void save_external_force() {
        external_force_ = Eigen2or3dVector(force_);
      }
    
      size_t get_id() const { return id_; }
      double get_mass() const { return mass_; }

      bool is_mobile() const { return is_mobile_; }
        
  protected:
      size_t id_;
      double mass_;
      Eigen2or3dVector position_;
      Eigen2or3dVector prev_position_;
      Eigen2or3dVector velocity_;
      Eigen2or3dVector acceleration_;
      Eigen2or3dVector force_;
      Eigen2or3dVector external_force_;
      bool is_mobile_;
      const std::vector<int> constrained_axes_;
  };
} // namespace beads_gym.bead
