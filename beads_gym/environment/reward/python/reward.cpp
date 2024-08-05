#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <pybind11/stl_bind.h>
#include <pybind11/eigen.h>
#include <vector>

#include "beads_gym/beads/bead.hpp"
#include "beads_gym/environment/reward/reward.hpp"

namespace py = pybind11;

//PYBIND11_MAKE_OPAQUE(std::vector<std::string>);
//PYBIND11_MAKE_OPAQUE(std::vector<std::vector<std::string>>);
//PYBIND11_MAKE_OPAQUE(std::vector<std::pair<int, int>>);
//PYBIND11_MAKE_OPAQUE(std::vector<std::vector<std::pair<int, int>>>);

PYBIND11_MODULE(reward, m) {
    m.doc() = "reward module"; // optional module docstring

    // Class Reward
    py::class_<::beads_gym::environment::reward::Reward<Eigen::Vector3d>, std::shared_ptr<::beads_gym::environment::reward::Reward<Eigen::Vector3d>>>(m, "Reward")
      .def(py::init<std::vector<std::shared_ptr<::beads_gym::beads::Bead<Eigen::Vector3d>>>&>(), py::arg("reference_beads"));
}
