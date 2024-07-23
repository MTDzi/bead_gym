#pragma once

#include <cassert>
#include <memory>
#include <utility>
#include <Eigen/Dense>
#include <vector>

#include "beads_gym/beads/bead.hpp"

namespace beads_gym::bonds {

template <typename Eigen2or3dVector>
class Bond {

    using BeadType = beads_gym::beads::Bead<Eigen2or3dVector>;

    public:
        Bond(size_t bead_1_id, size_t bead_2_id) : bead_1_id_{bead_1_id}, bead_2_id_{bead_2_id} {}

        void set_bead_1(std::shared_ptr<BeadType> bead_1) {
            bead_1_ = bead_1;
            assert(bead_1->get_id() == bead_1_id_);
        }

        void set_bead_2(std::shared_ptr<BeadType> bead_2) {
            bead_2_ = bead_2;
            assert(bead_2->get_id() == bead_2_id_);
        }

        size_t bead_1_id() {
            return bead_1_id_;
        }

        size_t bead_2_id() {
            return bead_2_id_;
        }       

        virtual double potential() {
            return 0.0d;
        }

        virtual void apply_forces() {
            return;
        }

        virtual void apply_torques() {
            return;
        }

        
    protected:
        size_t bead_1_id_;
        size_t bead_2_id_;
        std::shared_ptr<BeadType> bead_1_;
        std::shared_ptr<BeadType> bead_2_;
};
} // namespace beads_gym.bonds
