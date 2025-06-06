#include "beads_gym/bonds/distance_bond.hpp"
#include "beads_gym/beads/bead.hpp"

#include <gtest/gtest.h>
#include <iostream>
#include <numeric>
#include <string>
#include <Eigen/Dense>

namespace beads_gym::bonds {

TEST(DistanceBondTest, DistanceBondConstructor) {

  Eigen::Vector3d position_111 = Eigen::Vector3d{1.0, 1.0, 1.0};
  Eigen::Vector3d position_000 = Eigen::Vector3d::Zero();
  auto bead_1 = std::make_shared<beads_gym::beads::Bead<Eigen::Vector3d>>(0, position_111, 1.0, true);
  auto bead_2 = std::make_shared<beads_gym::beads::Bead<Eigen::Vector3d>>(1, position_000, 1.0, false);

  DistanceBond<Eigen::Vector3d> dist_bond{0, 1};

  dist_bond.set_bead_1(bead_1);
  dist_bond.set_bead_2(bead_2);

  // Check if the potential is equal to 0
  EXPECT_EQ(dist_bond.potential(), 0.0);

  std::cout << "Before:" << bead_1->get_force() << std::endl;

  dist_bond.apply_forces();

  std::cout << "After: " << bead_1->get_force() << std::endl;

  EXPECT_EQ(dist_bond.potential(), 0.0);
}

} // namespace beads_gym.beads
