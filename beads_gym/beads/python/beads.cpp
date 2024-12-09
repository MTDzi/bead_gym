#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <pybind11/stl_bind.h>
#include <pybind11/eigen.h>
#include <vector>

#include "beads_gym/beads/bead.hpp"

namespace py = pybind11;

//PYBIND11_MAKE_OPAQUE(std::vector<std::string>);
//PYBIND11_MAKE_OPAQUE(std::vector<std::vector<std::string>>);
//PYBIND11_MAKE_OPAQUE(std::vector<std::pair<int, int>>);
//PYBIND11_MAKE_OPAQUE(std::vector<std::vector<std::pair<int, int>>>);

PYBIND11_MODULE(beads, m) {
    m.doc() = "beads module"; // optional module docstring

    // Class Bead
    py::class_<::beads_gym::beads::Bead<Eigen::Vector3d>, std::shared_ptr<::beads_gym::beads::Bead<Eigen::Vector3d>>>(m, "Bead")
      .def(py::init<size_t, std::vector<double>&, double, bool, std::vector<int>&>(), py::arg("id"), py::arg("position"), py::arg("mass"), py::arg("is_mobile"), py::arg("constrained_axes") = std::vector<int>{})
      .def("set_position", &::beads_gym::beads::Bead<Eigen::Vector3d>::set_position)
      .def("get_position", &::beads_gym::beads::Bead<Eigen::Vector3d>::get_position)
      .def("get_velocity", &::beads_gym::beads::Bead<Eigen::Vector3d>::get_velocity)
      .def("get_acceleration", &::beads_gym::beads::Bead<Eigen::Vector3d>::get_acceleration)
      .def("get_external_acceleration", &::beads_gym::beads::Bead<Eigen::Vector3d>::get_external_acceleration);
}
