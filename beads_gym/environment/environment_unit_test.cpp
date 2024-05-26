#include "beads_gym/environment/environment.hpp"
#include "beads_gym/beads/bead.hpp"
#include "beads_gym/beads/three_degrees_of_freedom_bead.hpp"

#include <gtest/gtest.h>
#include <iostream>
#include <memory>
#include <numeric>
#include <string>
#include <Eigen/Dense>


namespace beads_gym::environment {
  
TEST(BeadTest, EnvironmentConstructor) {

  std::vector<std::shared_ptr<beads_gym::beads::Bead<Eigen::Vector3d>>> beads;
  for (int i = 1; i < 10; ++i) {
    std::vector<double> position{static_cast<double>(i), static_cast<double>(-i), 1.0d / static_cast<double>(i)};
    Eigen::Vector3d position_3d{position.data()};

    auto bead = std::make_shared<beads_gym::beads::Bead<Eigen::Vector3d>>(Eigen::Vector3d{1.0, 1.0, 1.0});

    beads.push_back(bead);
  }
  auto tdf_bead = std::make_shared<beads_gym::beads::ThreeDegreesOfFreedomBead<Eigen::Vector3d>>(Eigen::Vector3d::Zero(), 1.0d);
  beads.push_back(tdf_bead);
  Environment<Eigen::Vector3d> environment{beads};

  auto dupa = environment.get_beads();
  auto first_dupa = dupa.at(0)->get_position();
  std::cout << first_dupa << std::endl;
  environment.reset();

}

} // namespace beads_gym.environment