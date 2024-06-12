#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <pybind11/stl_bind.h>
#include <pybind11/eigen.h>
#include <vector>
#include "beads_gym/environment/environment.hpp"

namespace py = pybind11;

//PYBIND11_MAKE_OPAQUE(std::vector<std::string>);
//PYBIND11_MAKE_OPAQUE(std::vector<std::vector<std::string>>);
//PYBIND11_MAKE_OPAQUE(std::vector<std::pair<int, int>>);
//PYBIND11_MAKE_OPAQUE(std::vector<std::vector<std::pair<int, int>>>);

PYBIND11_MODULE(environment, m) {
    m.doc() = "environment module"; // optional module docstring

    // Class Environment
    py::class_<::beads_gym::environment::Environment<Eigen::Vector3d>, std::shared_ptr<beads_gym::environment::Environment<Eigen::Vector3d>>>(m, "Environment")
      .def(py::init<>())
      .def("add_bead", &::beads_gym::environment::Environment<Eigen::Vector3d>::add_bead)
      .def("add_bond", &::beads_gym::environment::Environment<Eigen::Vector3d>::add_bond)
      .def("get_beads", &::beads_gym::environment::Environment<Eigen::Vector3d>::get_beads);
}
