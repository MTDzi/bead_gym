#include "beads_gym/bonds/distance_bond.hpp"
#include "beads_gym/beads/bead.hpp"
#include "beads_gym/beads/three_degrees_of_freedom_bead.hpp"

#include <gtest/gtest.h>
#include <iostream>
#include <numeric>
#include <string>
#include <Eigen/Dense>

namespace beads_gym::bonds {

TEST(DistanceBondTest, DistanceBondConstructor) {

  auto bead_1 = std::make_shared<beads_gym::beads::Bead<Eigen::Vector3d>>(0, Eigen::Vector3d{1.0, 1.0, 1.0});
  auto tdf_bead_2 = std::make_shared<beads_gym::beads::ThreeDegreesOfFreedomBead<Eigen::Vector3d>>(1, Eigen::Vector3d::Zero(), 1.0d);

  DistanceBond<Eigen::Vector3d> dist_bond{0, 1};

  dist_bond.set_bead_1(bead_1);
  dist_bond.set_bead_2(tdf_bead_2);

  // Check if the pote ntial is equal to 0
  EXPECT_EQ(dist_bond.potential(), 0.0);
}

} // namespace beads_gym.beads