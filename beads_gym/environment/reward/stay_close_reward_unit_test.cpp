#include "beads_gym/beads/bead.hpp"
#include "beads_gym/environment/reward/stay_close_reward.hpp"

#include <gtest/gtest.h>
#include <iostream>
#include <memory>
#include <numeric>
#include <string>
#include <Eigen/Dense>
#include <vector>


namespace beads_gym::environment::reward {
  
TEST(RewardTest, RewardConstructor) {

  std::map<size_t, std::vector<double>> reference_positions = {{0, {0, 0, 0}}, {1, {0, 0, 1}}};
  std::vector<double> weights = {1.0, 2.0};
  StayCloseReward<Eigen::Vector3d> reward{reference_positions, weights};
  
  std::vector<std::shared_ptr<beads_gym::beads::Bead<Eigen::Vector3d>>> beads;
  for (int i=1; i<10; ++i) {
    std::vector<double> position{static_cast<double>(i), static_cast<double>(-i), 1.0d / static_cast<double>(i)};
    Eigen::Vector3d position_3d{position.data()};

    auto bead = std::make_shared<beads_gym::beads::Bead<Eigen::Vector3d>>(i, position_3d, 1.0d, true);
    beads.push_back(bead);
  }
  
  reward.calculate_reward(beads);
}

} // namespace beads_gym.environment.reward