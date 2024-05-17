#include "beads_gym/environment/environment.hpp"
#include "beads_gym/beads/bead.hpp"

#include <gtest/gtest.h>
#include <iostream>
#include <numeric>
#include <string>
#include <Eigen/Dense>


namespace beads_gym::environment {
  
TEST(BeadTest, EnvironmentConstructor) {

  std::vector<beads_gym::beads::Bead<Eigen::Vector3d>> beads;
  for (int i = 1; i < 10; ++i) {
    std::vector<double> position{static_cast<double>(i), static_cast<double>(-i), 1.0d / static_cast<double>(i)};
    Eigen::Vector3d position_3d{position.data()};

    beads.push_back(beads_gym::beads::Bead<Eigen::Vector3d>{position_3d});
  }
  Environment<Eigen::Vector3d> environment{beads};

  auto dupa = environment.get_beads();
  auto first_dupa = dupa[0].get_position();
  std::cout << first_dupa << std::endl;
  environment.reset();

}

} // namespace beads_gym.environment