#include "beads_gym/environment/integrator/integrator.hpp"
#include "beads_gym/beads/bead.hpp"

#include <gtest/gtest.h>
#include <iostream>
#include <memory>
#include <numeric>
#include <string>
#include <Eigen/Dense>


namespace beads_gym::environment::integrator {
  
TEST(IntegratorTest, IntegratorConstructor) {

  std::vector<std::shared_ptr<beads_gym::beads::Bead<Eigen::Vector3d>>> beads;
  for (int i = 1; i < 10; ++i) {
    std::vector<double> position{static_cast<double>(i), static_cast<double>(-i), 1.0 / static_cast<double>(i)};
    Eigen::Vector3d position_3d{position.data()};

    auto bead = std::make_shared<beads_gym::beads::Bead<Eigen::Vector3d>>(i, position_3d, 1.0, true);

    beads.push_back(bead);
  }

  double dt = 0.01;
  Integrator<Eigen::Vector3d> integ{dt};
  integ.step(beads);
  for (auto &bead : beads) {
    std::cout << "Position: " << bead->get_position() << std::endl;
  }
  std::cout << "Integrator step" << std::endl;
}

} // namespace beads_gym.environment