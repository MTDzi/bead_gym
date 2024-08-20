#pragma once

#include <memory>
#include <type_traits>

#include <Eigen/Dense>

#include "beads_gym/bonds/bond.hpp"
#include "beads_gym/beads/bead.hpp"


namespace beads_gym::bonds {

template <typename Eigen2or3dVector>
class DistanceBond : public Bond<Eigen2or3dVector> {

    using Bead = beads_gym::beads::Bead<Eigen2or3dVector>;

    public:
        DistanceBond(size_t bead_id_1, size_t bead_id_2, double k = 100.0
        , double r0 = 1.0) : Bond<Eigen2or3dVector>{bead_id_1, bead_id_2}, k_{k}, r0_{r0} {}

        double potential() override {
            auto pos_diff = this->beads_position_diff();
            return k_ * std::pow(pos_diff.norm() - r0_, 2.0);
        }

        void apply_forces() override {
            bool is_mobile_1 = this->bead_1_->is_mobile();
            bool is_mobile_2 = this->bead_2_->is_mobile();
            if (!is_mobile_1 && !is_mobile_2) {
                return;
            }

            auto pos_diff = this->beads_position_diff();
            auto force = 2.0 * k_ * (pos_diff.norm() - r0_) * pos_diff.normalized();
            
            if (is_mobile_1) {
                this->bead_1_->add_force(-force);
            }
            if (is_mobile_2) {
                this->bead_2_->add_force(force);
            }
        }

        void apply_torques() override {
            return;
        }

    private:
        double k_ = 100.0;
        double r0_ = 1.0;

        Eigen2or3dVector beads_position_diff() {
            return this->bead_1_->get_position() - this->bead_2_->get_position();
        }
};

} // namespace beads_gym.bonds