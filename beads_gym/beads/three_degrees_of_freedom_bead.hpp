#pragma once

#include <Eigen/Dense>

#include "bead.hpp"

namespace beads_gym::beads {


template <typename Eigen2or3dVector>
class ThreeDegreesOfFreedomBead : public Bead<Eigen2or3dVector> {
    public:
        ThreeDegreesOfFreedomBead(Eigen2or3dVector position, double mass) : Bead<Eigen2or3dVector>{position}, mass_{mass} {
            velocity_ = Eigen2or3dVector::Zero();
            acceleration_ = Eigen2or3dVector::Zero();
        }
        ThreeDegreesOfFreedomBead(std::vector<double> &position, double mass) : Bead<Eigen2or3dVector>{position}, mass_{mass} {
            velocity_ = Eigen2or3dVector::Zero();
            acceleration_ = Eigen2or3dVector::Zero();
        }

        ThreeDegreesOfFreedomBead(const ThreeDegreesOfFreedomBead&) = default;
        ThreeDegreesOfFreedomBead(ThreeDegreesOfFreedomBead&&) = default;
        ThreeDegreesOfFreedomBead& operator=(const ThreeDegreesOfFreedomBead&) = default;
        ThreeDegreesOfFreedomBead& operator=(ThreeDegreesOfFreedomBead&&) = default;
        ~ThreeDegreesOfFreedomBead() = default;
    
        double get_mass() const { return mass_; }
    
        void set_velocity(const Eigen2or3dVector& velocity) { velocity_ = velocity; }
        Eigen2or3dVector get_velocity() const { return velocity_; }
    
        void set_acceleration(const Eigen2or3dVector& acceleration) { acceleration_ = acceleration; }
        Eigen2or3dVector get_acceleration() const { return acceleration_; }

    private:
        double mass_;
        Eigen2or3dVector velocity_;
        Eigen2or3dVector acceleration_;
};

} // namespace beads_gym::beads