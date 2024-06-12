#pragma once

#include <memory>
#include <Eigen/Dense>

#include "beads_gym/bonds/bond.hpp"
#include "beads_gym/beads/bead.hpp"


namespace beads_gym::bonds {
    
template <typename Eigen2or3dVector>
class DistanceBond : public Bond<Eigen2or3dVector> {

    using Bead = beads_gym::beads::Bead<Eigen2or3dVector>;

    public:
        DistanceBond(size_t bead_id_1, size_t bead_id_2) : Bond<Eigen2or3dVector>{bead_id_1, bead_id_2} {}

        double potential() {
            return 0.0d;
        }

        void apply_forces() {
            return;
        }

        void apply_torques() {
            return;
        }
};

} // namespace beads_gym.bonds