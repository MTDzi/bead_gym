#include "beads_gym/environment/environment.hpp"
#include "beads_gym/beads/bead.hpp"
#include "beads_gym/bonds/distance_bond.hpp"

#include <gtest/gtest.h>
#include <iostream>
#include <memory>
#include <numeric>
#include <string>
#include <Eigen/Dense>


namespace beads_gym::environment {
  
TEST(EnvironmentTest, EnvironmentConstructor) {

  Environment<Eigen::Vector3d> environment;
  for (int i = 1; i < 10; ++i) {
    std::vector<double> position{static_cast<double>(i), static_cast<double>(-i), 1.0d / static_cast<double>(i)};
    Eigen::Vector3d position_3d{position.data()};

    auto bead = std::make_shared<beads_gym::beads::Bead<Eigen::Vector3d>>(i, position_3d, 1.0d, true);
    environment.add_bead(bead);
  }

  auto dist_bond = std::make_shared<beads_gym::bonds::DistanceBond<Eigen::Vector3d>>(1, 2);
  environment.add_bond(dist_bond);

  std::cout << "Bead 0 position BEFORE: " << environment.get_beads().at(1).get()->get_position() << std::endl;

  std::map<size_t, std::vector<double>> action = {{1, {0.0, 0.0, 0.0}}}; 
  environment.step(action);
  std::cout << "Bead 0 position AFTER zero action: " << environment.get_beads().at(1).get()->get_position() << std::endl;

  action = {{1, {1.0, 0.0, 0.0}}}; 
  environment.step(action);
  std::cout << "Bead 0 position AFTER non-zero action: " << environment.get_beads().at(1).get()->get_position() << std::endl;

  environment.reset();
}

} // namespace beads_gym.environment