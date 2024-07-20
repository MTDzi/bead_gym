#include "beads_gym/environment/environment.hpp"
#include "beads_gym/beads/bead.hpp"

#include <gtest/gtest.h>
#include <iostream>
#include <memory>
#include <numeric>
#include <string>
#include <Eigen/Dense>


namespace beads_gym::environment {
  
TEST(EnvironmentTest, EnvironmentConstructor) {

  std::vector<std::shared_ptr<beads_gym::beads::Bead<Eigen::Vector3d>>> beads;
  for (int i = 1; i < 10; ++i) {
    std::vector<double> position{static_cast<double>(i), static_cast<double>(-i), 1.0d / static_cast<double>(i)};
    Eigen::Vector3d position_3d{position.data()};

    auto bead = std::make_shared<beads_gym::beads::Bead<Eigen::Vector3d>>(0, position_3d, 1.0d, true);

    beads.push_back(bead);
  }
  Environment<Eigen::Vector3d> environment{beads};

  auto beads_from_env = environment.get_beads();
  auto first_bead = beads_from_env.at(0)->get_position();
  std::cout << first_bead << std::endl;
  environment.reset();
}

} // namespace beads_gym.environment