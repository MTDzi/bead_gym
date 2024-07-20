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
      Bead(size_t id, std::vector<double> &position, double mass, bool is_mobile)
        : id_(id), mass_(mass), velocity_(Eigen2or3dVector::Zero()), 
          acceleration_(Eigen2or3dVector::Zero()), force_(Eigen2or3dVector::Zero()),
          is_mobile_(is_mobile) {
        assert(mass > 0);
        position_ = Eigen2or3dVector(position.data());
        // Initialization logic is centralized here.
      }

      Bead(size_t id, Eigen2or3dVector &position, double mass, bool is_mobile)
        : id_(id), mass_(mass), velocity_(Eigen2or3dVector::Zero()),
          acceleration_(Eigen2or3dVector::Zero()), force_(Eigen2or3dVector::Zero()),
          is_mobile_(is_mobile) {
        assert(mass > 0);
        // Initialization logic is centralized here.
      }
      // TODO: Not sure if we need these:
      // Bead(const Bead&) = default;
      // Bead(Bead&&) = default;
      Bead& operator=(const Bead&) = default;
      Bead& operator=(Bead&&) = default;
      ~Bead() = default;

      double get_mass() const { return mass_; }

      void set_position(const Eigen2or3dVector& position) { position_ = position; }
      Eigen2or3dVector get_position() const { return position_; }
  
      void set_velocity(const Eigen2or3dVector& velocity) { velocity_ = velocity; }
      Eigen2or3dVector get_velocity() const { return velocity_; }
  
      void set_acceleration(const Eigen2or3dVector& acceleration) { acceleration_ = acceleration; }
      Eigen2or3dVector get_acceleration() const { return acceleration_; }

      void add_force(const Eigen2or3dVector& force) { force_ += force; }
      void zero_out_force() { force_ = Eigen2or3dVector::Zero(); }
      Eigen2or3dVector get_force() const { return force_; }
    
      size_t get_id() const { return id_; }

      bool is_mobile() const { return is_mobile_; }
        
  protected:
      size_t id_;
      double mass_;
      Eigen2or3dVector position_;
      Eigen2or3dVector velocity_;
      Eigen2or3dVector acceleration_;
      Eigen2or3dVector force_;
      bool is_mobile_;
  };
} // namespace beads_gym.bead
