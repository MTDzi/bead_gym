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

    // Class Bead
    py::class_<::beads_gym::environment::Environment<Eigen::Vector3d>>(m, "Environment")
      .def(py::init<>())
      .def(py::init<std::vector<::beads_gym::beads::Bead<Eigen::Vector3d>>&>()) // Fix: Pass the vector by value
      // .def("set_position", &::beads_gym::environment::Environment<Eigen::Vector3d>::set_position)
      .def("get_beads", &::beads_gym::environment::Environment<Eigen::Vector3d>::get_beads);
}
