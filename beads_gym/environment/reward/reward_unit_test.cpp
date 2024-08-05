#include "beads_gym/beads/bead.hpp"
#include "beads_gym/environment/reward/reward.hpp"

#include <gtest/gtest.h>
#include <iostream>
#include <memory>
#include <numeric>
#include <string>
#include <Eigen/Dense>


namespace beads_gym::environment::reward {
  
TEST(RewardTest, RewardConstructor) {

  auto beads = std::vector<std::shared_ptr<beads_gym::beads::Bead<Eigen::Vector3d>>>();
  for (int i=1; i<10; ++i) {
    std::vector<double> position{static_cast<double>(i), static_cast<double>(-i), 1.0d / static_cast<double>(i)};
    Eigen::Vector3d position_3d{position.data()};

    auto bead = std::make_shared<beads_gym::beads::Bead<Eigen::Vector3d>>(i, position_3d, 1.0d, true);
    beads.push_back(bead);
  }
  Reward<Eigen::Vector3d> reward{beads};
  reward.calculate_reward(beads);
}

} // namespace beads_gym.environment.reward