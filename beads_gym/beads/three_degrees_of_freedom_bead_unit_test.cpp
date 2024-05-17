#include "beads_gym/beads/three_degrees_of_freedom_bead.hpp"

#include <gtest/gtest.h>
#include <iostream>
#include <numeric>
#include <string>
#include <Eigen/Dense>

namespace beads_gym::beads {

TEST(ThreeDegreesOfFreedomBeadTest, ThreeDegreesOfFreedomBeadConstructor) {
  ThreeDegreesOfFreedomBead<Eigen::Vector3d> tdof_bead_3d{{0, 0, 0}, 1.0d};
  ThreeDegreesOfFreedomBead<Eigen::Vector2d> tdof_bead_2d{{0, 0}, 1.0d};
}

} // namespace beads_gym.beads