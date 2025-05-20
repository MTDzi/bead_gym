#pragma once

#include <memory>
#include <type_traits>
#include <optional>

#include <Eigen/Dense>

#include "beads_gym/bonds/bond.hpp"
#include "beads_gym/beads/bead.hpp"


namespace beads_gym::bonds {

template <typename Eigen2or3dVector>
class DistanceBond : public Bond<Eigen2or3dVector> {

    using Bead = beads_gym::beads::Bead<Eigen2or3dVector>;

    public:
        DistanceBond(size_t bead_id_1, size_t bead_id_2, double k = 100.0
        , std::optional<double> r0 = std::nullopt) : Bond<Eigen2or3dVector>{bead_id_1, bead_id_2}, k_{k} {
            if (r0.has_value()) {
                r0_ = *r0;
            } else {
                r0_ = std::nullopt;
            }
        }

        double potential() override {
            check_r0();
            auto pos_diff = this->beads_position_diff();
            return k_ * std::pow(pos_diff.norm() - *r0_, 2.0);
        }

        void apply_forces() override {
            check_r0();
            bool is_mobile_1 = this->bead_1_->is_mobile();
            bool is_mobile_2 = this->bead_2_->is_mobile();
            if (!is_mobile_1 && !is_mobile_2) {
                return;
            }

            auto pos_diff = this->beads_position_diff();
            auto force = 2.0 * k_ * (pos_diff.norm() - *r0_) * pos_diff.normalized();
            
            if (is_mobile_1) {
                this->bead_1_->add_force(-force);
            }
            if (is_mobile_2) {
                this->bead_2_->add_force(force);
            }
        }

        void apply_torques() override {
            check_r0();
            return;
        }

        std::pair<size_t, size_t> get_bead_ids() {
            return {this->bead_1_id_, this->bead_2_id_};
        }

    private:

        double k_;
        std::optional<double> r0_;

        Eigen2or3dVector beads_position_diff() {
            return this->bead_1_->get_position() - this->bead_2_->get_position();
        }

        void check_r0() {
            if (!r0_.has_value()) {
                r0_ = beads_position_diff().norm();  // lazy init
            }
        }
};

} // namespace beads_gym.bonds