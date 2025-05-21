#include <optional>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <pybind11/stl_bind.h>
#include <pybind11/eigen.h>
#include "beads_gym/bonds/bond.hpp"
#include "beads_gym/bonds/distance_bond.hpp"

namespace py = pybind11;

PYBIND11_MODULE(bonds, m) {
    m.doc() = "bonds module"; // optional module docstring

    // Class Bond
    py::class_<::beads_gym::bonds::Bond<Eigen::Vector3d>, std::shared_ptr<::beads_gym::bonds::Bond<Eigen::Vector3d>>>(m, "Bond")
      .def(
        py::init<size_t, size_t>(),
        py::arg("bead_id_1"),
        py::arg("bead_id_2")
      );
    
    py::class_<beads_gym::bonds::DistanceBond<Eigen::Vector3d>, ::beads_gym::bonds::Bond<Eigen::Vector3d>, std::shared_ptr<::beads_gym::bonds::DistanceBond<Eigen::Vector3d>>>(m, "DistanceBond")
      .def(
        py::init<size_t, size_t, double, std::optional<double>>(),
        py::arg("bead_id_1"),
        py::arg("bead_id_2"),
        py::arg("k") = 100.0,
        py::arg("r0") = py::none()
      )
      .def(
        py::init<size_t, size_t, double>(),
        py::arg("bead_id_1"),
        py::arg("bead_id_2"),
        py::arg("k") = 100.0
      )
      .def("potential", &beads_gym::bonds::DistanceBond<Eigen::Vector3d>::potential)
      .def("get_bead_ids", &beads_gym::bonds::DistanceBond<Eigen::Vector3d>::get_bead_ids);
}
