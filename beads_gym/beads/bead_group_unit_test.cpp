#include "beads_gym/beads/bead_group.hpp"

#include <gtest/gtest.h>
#include <iostream>
#include <numeric>
#include <string>
#include <memory>
#include <Eigen/Dense>


namespace beads_gym::beads {


BeadGroup<Eigen::Vector3d> get_10_bead_group() {
  std::vector<std::shared_ptr<Bead<Eigen::Vector3d>>> beads;
  for (size_t i=0; i<10; i++) {
    Eigen::Vector3d position_3d{(double)i, (double)i, 0.0};
    auto bead_3d = std::make_shared<Bead<Eigen::Vector3d>>(0, position_3d, 1.0, true);
    beads.push_back(bead_3d);
  }
  return BeadGroup<Eigen::Vector3d>{beads};
}

BeadGroup<Eigen::Vector3d> get_square_bead_group() {
  std::vector<std::shared_ptr<Bead<Eigen::Vector3d>>> beads;
  std::vector<std::vector<double>> positions = {
    {0, 0, 0},
    {1, 0, 0},
    {1, 1, 0},
    {0, 1, 0}
  };
  for (size_t i=0; i<positions.size(); i++) {
    auto bead_3d = std::make_shared<Bead<Eigen::Vector3d>>(0, positions[i], 1.0, true);
    beads.push_back(bead_3d);
  }
  return BeadGroup<Eigen::Vector3d>{beads};
}
  
TEST(BeadGroupTest, BeadGroupConstructor) {
  auto ten_bead_group = get_10_bead_group();
  auto square_bead_group = get_square_bead_group();
}

TEST(BeadGroupTest, BeadGroupUpdates) {
  auto bead_group = get_10_bead_group();
  bead_group.update_com_and_inertia_mat();
  bead_group.update_net_force_and_torque();

  // Check the COM first
  Eigen::Vector3d com = bead_group.get_com();
  std::vector<double> positions{0, 1, 2, 3, 4, 5, 6, 7, 8, 9};
  double sum = std::accumulate(positions.begin(), positions.end(), 0);
  Eigen::Vector3d expected_com = Eigen::Vector3d{
    sum / positions.size(), sum / positions.size(), 0.0
  };
  EXPECT_EQ(com, expected_com);

  // The torque should be 0
  EXPECT_EQ(bead_group.get_torque_world(), Eigen::Vector3d::Zero());
}

TEST(BeadGroupTest, BeadGroupOrientation) {
  auto bead_group = get_square_bead_group();

  // Now the orientation
  Eigen::Quaterniond orientation = bead_group.get_orientation();
  Eigen::Matrix3d xyz;
  xyz.col(0) = Eigen::Vector3d{1, 0, 0};
  xyz.col(1) = Eigen::Vector3d{0, 0, 1};
  xyz.col(2) = Eigen::Vector3d{0, -1, 0};
  Eigen::Quaterniond expected_orientation = Eigen::Quaterniond{xyz}; 
  EXPECT_EQ(orientation, expected_orientation);
}

}
