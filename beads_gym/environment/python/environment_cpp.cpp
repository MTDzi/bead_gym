
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <pybind11/stl_bind.h>
#include <pybind11/eigen.h>

#include <vector>

#include "beads_gym/environment/environment.hpp"
#include "beads_gym/environment/reward/reward.hpp"

namespace py = pybind11;

//PYBIND11_MAKE_OPAQUE(std::vector<std::string>);
//PYBIND11_MAKE_OPAQUE(std::vector<std::vector<std::string>>);
//PYBIND11_MAKE_OPAQUE(std::vector<std::pair<int, int>>);
//PYBIND11_MAKE_OPAQUE(std::vector<std::vector<std::pair<int, int>>>);

PYBIND11_MODULE(environment_cpp, m) {
    m.doc() = "environment module"; // optional module docstring

    // Class Environment
    py::class_<::beads_gym::environment::Environment<Eigen::Vector3d>, std::shared_ptr<beads_gym::environment::Environment<Eigen::Vector3d>>>(m, "EnvironmentCpp")
      .def(py::init<double>(), py::arg("timestep"))
      .def("add_bead", &::beads_gym::environment::Environment<Eigen::Vector3d>::add_bead)
      .def("add_bond", &::beads_gym::environment::Environment<Eigen::Vector3d>::add_bond)
      .def("add_reward_calculator", &::beads_gym::environment::Environment<Eigen::Vector3d>::add_reward_calculator)
      .def("get_beads", &::beads_gym::environment::Environment<Eigen::Vector3d>::get_beads)
      .def("step", &::beads_gym::environment::Environment<Eigen::Vector3d>::step)
      .def("calc_bond_potential", &::beads_gym::environment::Environment<Eigen::Vector3d>::calc_bond_potential)
      .def("reset", &::beads_gym::environment::Environment<Eigen::Vector3d>::reset);
}
