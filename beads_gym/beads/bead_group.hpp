#pragma once

#include <iostream>
#include <utility>
#include <memory>
#include <cassert>
#include <type_traits>

#include <Eigen/Geometry> // For Quaterniond

#include "beads_gym/beads/bead.hpp"


namespace beads_gym::beads {

template <typename Eigen2or3dVector>
class BeadGroup {

  using BeadType = beads_gym::beads::Bead<Eigen2or3dVector>;

  // Needed to calculate the intertia tensor as it can be either
  // a 2D or 3D matrix
  using Scalar = typename Eigen2or3dVector::Scalar;
  static constexpr int Dim = Eigen2or3dVector::RowsAtCompileTime;
  using Eigen2or3dMatrix = Eigen::Matrix<Scalar, Dim, Dim>;

  public:
    BeadGroup() = delete;
    BeadGroup(std::vector<std::shared_ptr<BeadType>> beads) : beads_{beads} {
      assert(
        beads.size() >= 2
        && "At least two beads are needed to define a BeadGroup"
      );

      total_mass_ = 0;
      for (const auto& bead : beads_) {
        total_mass_ += bead->get_mass();
      }

      com_prev_.setZero();
      update_com_and_inertia_mat();

      for (auto& bead : beads_) {
        local_positions_.push_back(bead->get_position() - com_);
      }

      // Calculate orientation
      orientation_ = _calc_initial_orientation();

      angular_mom_body_.setZero();
      angular_vel_body_.setZero();
    }

  void update_com_and_inertia_mat() {
    com_ = _calc_center_of_mass();
    intertia_mat_ = _calc_intertia_mat();
  }

  void update_net_force_and_torque() {
    net_force.setZero();
    net_torque_world.setZero();
    for (auto& bead : beads_) {
      Eigen2or3dVector force = bead->get_force();
      Eigen2or3dVector position = bead->get_position();
      net_force += force;
      net_torque_world += (position - com_).cross(force);
    }
  }

  // COM-related getters and setters
  Eigen2or3dVector get_com() const { return com_; }
  Eigen2or3dVector get_com_prev() const { return com_prev_; }
  void set_com(const Eigen2or3dVector& new_com) {
    com_prev_ = com_;
    com_ = new_com;
  }
  void set_com_velocity(const Eigen2or3dVector& new_com_velocity) { com_velocity_ = new_com_velocity; }
  Eigen2or3dVector get_com_acceleration() const { return net_force / total_mass_; }

  // Rotational-related getters and setters
  Eigen2or3dVector get_torque_world() const { return net_torque_world; }
  Eigen::Quaterniond get_orientation() const { return orientation_; }
  Eigen2or3dVector get_angular_mom_body() const { return angular_mom_body_; }
  Eigen2or3dMatrix get_inertia_mat() const { return intertia_mat_; }
  void set_angular_mom_body(const Eigen2or3dVector& new_angular_mom_body) { angular_mom_body_ = new_angular_mom_body; }
  void set_angular_vel_world(const Eigen2or3dVector& new_angular_vel_world) { angular_vel_world_ = new_angular_vel_world; }
  void set_orientation(const Eigen::Quaterniond new_orientation) { orientation_ = new_orientation; }

  void update_beads_world_positions() {
    for (size_t bead_id=0; bead_id<beads_.size(); bead_id++) {
      beads_[bead_id]->set_position(
        com_ +
        orientation_ * local_positions_[bead_id] 
      );

      beads_[bead_id]->set_velocity(
        com_velocity_ +
        angular_vel_body_.cross(beads_[bead_id]->get_position() - com_)
      );
    }
  }

  private:
    Eigen2or3dVector _calc_center_of_mass() {
      com_.setZero();
      for (const auto& bead : beads_) {
        com_ += bead->get_mass() * bead->get_position();
      }
      return com_ / total_mass_;
    }

    Eigen2or3dMatrix _calc_intertia_mat() {
      intertia_mat_ = Eigen2or3dMatrix::Zero();
      Eigen2or3dMatrix identity = Eigen2or3dMatrix::Identity();
      for (const auto& bead : beads_) {
        Eigen2or3dVector r_prime = bead->get_position() - com_; // Position relative to COM

        // I = m * ( (r^2)I - r*r^T )
        // TODO: This requires inspection for the 2D case, for 3D case it will
        // work
        intertia_mat_ += bead->get_mass() * (
          r_prime.squaredNorm() * identity - r_prime * r_prime.transpose()
        );
      }
      return intertia_mat_;
    }

    Eigen::Quaterniond _calc_initial_orientation() {
      Eigen2or3dVector first_diff = (beads_[1]->get_position() - beads_[0]->get_position()).stableNormalized();
      if (beads_.size() == 2) {
        // We need to come up with two more vectors that are perpendicular to the first_diff
        orientation_ = _calc_initial_orientation_for_linear_system(first_diff);
      } else {
        // We have one more bead at our disposal, BUT we need to make sure the three beads lie on a line
        Eigen2or3dVector second_diff = (beads_[2]->get_position() - beads_[1]->get_position()).stableNormalized();
        // TODO: We could explore all beads to check if we can find a set that defines a plane instead
        // of just stopping after looking at just three first beads
        
        // Check if first_diff and second_diff are basically the same
        if ((first_diff - second_diff).stableNorm() < 0.01) {
          orientation_ = _calc_initial_orientation_for_linear_system(first_diff);
        } else {
          orientation_ = _calc_initial_orientation_for_non_linear_system(first_diff, second_diff);
        }
      }
      return orientation_;
    }

    Eigen::Quaterniond _calc_initial_orientation_for_linear_system(Eigen2or3dVector first_diff) {
      Eigen2or3dVector non_aligned_proposal = first_diff;
      non_aligned_proposal(0) += 1.;
      non_aligned_proposal(1) -= 0.5;  // Basically, make sure the vector is different
      non_aligned_proposal = non_aligned_proposal.stableNormalized();
  
      Eigen::Vector3d x_axis;
      Eigen::Vector3d y_axis;
      Eigen::Vector3d z_axis;

      // TODO: This is a hack for the 2D case to work such that a quaternion describes an orientation of
      // of a 2D group of beads. However, the proper way would be to NOT use quaternions.
      constexpr int Dim = Eigen2or3dVector::RowsAtCompileTime;
      if constexpr (Dim == 2) {
        x_axis = Eigen::Vector3d{first_diff(0), first_diff(1), 0.0};
        Eigen::Vector2d perpendicular = Eigen::Vector2d{first_diff(1), -first_diff(0)};
        y_axis = Eigen::Vector3d{perpendicular(0), perpendicular(1), 0.0};
        z_axis = Eigen::Vector3d{0.0, 0.0, 1.0};
      } else if constexpr (Dim == 3) {
        x_axis = first_diff;
        Eigen::Vector3d perpendicular = first_diff.cross(non_aligned_proposal).stableNormalized();
        y_axis = perpendicular;
        z_axis = first_diff.cross(perpendicular).stableNormalized();
      }

      return _create_quaternion_from_axes(x_axis, y_axis, z_axis);
    }

    Eigen::Quaterniond _calc_initial_orientation_for_non_linear_system(Eigen2or3dVector first_diff, Eigen2or3dVector second_diff) {
      Eigen::Vector3d x_axis;
      Eigen::Vector3d y_axis;
      Eigen::Vector3d z_axis;

      // TODO: This is a hack for the 2D case to work such that a quaternion describes an orientation of
      // of a 2D group of beads. However, the proper way would be to NOT use quaternions.
      constexpr int Dim = Eigen2or3dVector::RowsAtCompileTime;
      if constexpr (Dim == 2) {
        x_axis = Eigen::Vector3d{first_diff(0), first_diff(1), 0.0};
        Eigen::Vector2d perpendicular = Eigen::Vector2d{first_diff(1), -first_diff(0)};
        y_axis = Eigen::Vector3d{perpendicular(0), perpendicular(1), 0.0};
        z_axis = Eigen::Vector3d{0.0, 0.0, 1.0};
      } else if constexpr (Dim == 3) {
        x_axis = first_diff;
        Eigen::Vector3d perpendicular = first_diff.cross(second_diff).stableNormalized();
        y_axis = perpendicular;
        z_axis = first_diff.cross(perpendicular).stableNormalized();
      }

      return _create_quaternion_from_axes(x_axis, y_axis, z_axis);
    }

    Eigen::Quaterniond _create_quaternion_from_axes(
        const Eigen::Vector3d& x_axis,
        const Eigen::Vector3d& y_axis,
        const Eigen::Vector3d& z_axis
    ) {
        assert(x_axis.dot(y_axis) < 1e-9); // Check for orthogonality
        assert(x_axis.cross(y_axis).isApprox(z_axis)); // Check for right-handedness

        // Step 1: Construct the 3x3 rotation matrix
        Eigen::Matrix3d rotation_matrix;
        rotation_matrix.col(0) = x_axis;
        rotation_matrix.col(1) = y_axis;
        rotation_matrix.col(2) = z_axis;

        // Step 2: Create a Quaternion from the rotation matrix
        Eigen::Quaterniond q(rotation_matrix);

        q.normalize(); // Just to make sure 

        return q;
    }

    std::vector<std::shared_ptr<BeadType>> beads_;
    std::vector<Eigen2or3dVector> local_positions_;
    double total_mass_;

    Eigen2or3dVector com_; // COM = Center Of Mass
    Eigen2or3dVector com_prev_;
    Eigen2or3dVector com_velocity_;

    Eigen2or3dMatrix intertia_mat_; // Intertia tensor
    Eigen::Quaterniond orientation_; // TODO: What about 2D?
    Eigen2or3dVector net_force;
    Eigen2or3dVector net_torque_world;
    Eigen2or3dVector angular_vel_body_;
    Eigen2or3dVector angular_vel_world_;
    Eigen2or3dVector angular_mom_body_;
  };
} // namespace beads_gym.bead
