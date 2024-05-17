#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <pybind11/stl_bind.h>
#include <pybind11/eigen.h>
#include "beads_gym/beads/bead.hpp"

namespace py = pybind11;

//PYBIND11_MAKE_OPAQUE(std::vector<std::string>);
//PYBIND11_MAKE_OPAQUE(std::vector<std::vector<std::string>>);
//PYBIND11_MAKE_OPAQUE(std::vector<std::pair<int, int>>);
//PYBIND11_MAKE_OPAQUE(std::vector<std::vector<std::pair<int, int>>>);

PYBIND11_MODULE(bead, m) {
    m.doc() = "bead module"; // optional module docstring

    // Class Bead
    py::class_<::beads_gym::beads::Bead<Eigen::Vector3d>>(m, "Bead")
      .def(py::init<>())
      .def(py::init<std::vector<double>&>())
      .def("set_position", &::beads_gym::beads::Bead<Eigen::Vector3d>::set_position)
      .def("get_position", &::beads_gym::beads::Bead<Eigen::Vector3d>::get_position);
}
