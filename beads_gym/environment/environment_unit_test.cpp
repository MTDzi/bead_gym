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

  using DistanceBondType = beads_gym::bonds::DistanceBond<Eigen::Vector3d>;
  using BeadType = beads_gym::beads::Bead<Eigen::Vector3d>;

  Environment<Eigen::Vector3d> produce_environment(std::vector<std::pair<size_t, size_t>> bead_pairs_for_bonds) {
    Environment<Eigen::Vector3d> environment{0.01};
    for (int i = 1; i < 10; ++i) {
      Eigen::Vector3d position_3d{static_cast<double>(i), static_cast<double>(-i), 1.0d / static_cast<double>(i)};

      auto bead = std::make_shared<BeadType>(i, position_3d, 1.0d, true);
      environment.add_bead(bead);
    }

    for (auto& pair : bead_pairs_for_bonds) {
      environment.add_bond(std::make_shared<DistanceBondType>(pair.first, pair.second));
    }

    return environment;
  }

  TEST(EnvironmentTest, EnvironmentConstructor) {

    Environment<Eigen::Vector3d> environment{0.01};
    for (int i = 1; i < 10; ++i) {
      std::vector<double> position{static_cast<double>(i), static_cast<double>(-i), 1.0d / static_cast<double>(i)};
      Eigen::Vector3d position_3d{position.data()};

      auto bead = std::make_shared<BeadType>(i, position_3d, 1.0d, true);
      environment.add_bead(bead);
    }

    auto dist_bond = std::make_shared<DistanceBondType>(1, 2);
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

  TEST(EnvironmentTest, EnvironmentWithBeadGroups) {

    std::vector<std::pair<size_t, size_t>> bead_pairs_for_bonds = {
      {1, 2},
      {2, 3},
      {3, 4},
      {4, 1},
      {4, 5}
    };
    auto environment = produce_environment(bead_pairs_for_bonds);

    EXPECT_EQ(environment.get_bonds().size(), bead_pairs_for_bonds.size());

    // Add a bead group comprising beads {1, 2}
    environment.add_bead_group({1, 2});

    // Bead groups and bonds are resolved when making the first step
    std::map<size_t, std::vector<double>> action = {{1, {0.0, 0.0, 0.0}}};
    environment.step(action);

    EXPECT_EQ(environment.get_bonds().size(), bead_pairs_for_bonds.size() - 1);
  }

  TEST(EnvironmentTest, EnvironmentWithBeadGroups2) {

    std::vector<std::pair<size_t, size_t>> bead_pairs_for_bonds = {
      {1, 2},
      {2, 3},
      {3, 1},
      {3, 4},
      {4, 5},
      {5, 6}
    };
    auto environment = produce_environment(bead_pairs_for_bonds);

    EXPECT_EQ(environment.get_bonds().size(), bead_pairs_for_bonds.size());

    // Add a bead group
    environment.add_bead_group({1, 2, 3});

    // Bead groups and bonds are resolved when making the first step
    std::map<size_t, std::vector<double>> action = {{1, {0.0, 0.0, 0.0}}};
    environment.step(action);

    EXPECT_EQ(environment.get_bonds().size(), bead_pairs_for_bonds.size() - 3);
  }

  TEST(EnvironmentTest, EnvironmentWithBeadGroups3) {

    std::vector<std::pair<size_t, size_t>> bead_pairs_for_bonds = {
      {1, 2},
      {2, 3},
      {3, 1},
      {3, 4},
      {4, 5},
      {5, 6}
    };
    auto environment = produce_environment(bead_pairs_for_bonds);

    EXPECT_EQ(environment.get_bonds().size(), bead_pairs_for_bonds.size());

    // Add a bead group
    environment.add_bead_group({1, 2, 3, 4});

    // Bead groups and bonds are resolved when making the first step
    std::map<size_t, std::vector<double>> action = {{1, {0.0, 0.0, 0.0}}};
    environment.step(action);

    EXPECT_EQ(environment.get_bonds().size(), bead_pairs_for_bonds.size() - 4);
  }

} // namespace beads_gym.environment