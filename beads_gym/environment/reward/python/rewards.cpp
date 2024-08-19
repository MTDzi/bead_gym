#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <pybind11/stl_bind.h>
#include <pybind11/eigen.h>
#include <vector>
#include <map>

#include "beads_gym/beads/bead.hpp"
#include "beads_gym/environment/reward/stay_close_reward.hpp"

namespace py = pybind11;

//PYBIND11_MAKE_OPAQUE(std::vector<std::string>);
//PYBIND11_MAKE_OPAQUE(std::vector<std::vector<std::string>>);
//PYBIND11_MAKE_OPAQUE(std::vector<std::pair<int, int>>);
//PYBIND11_MAKE_OPAQUE(std::vector<std::vector<std::pair<int, int>>>);

PYBIND11_MODULE(rewards, m) {
    m.doc() = "reward module"; // optional module docstring

    // Class Reward
    py::class_<::beads_gym::environment::reward::StayCloseReward<Eigen::Vector3d>, std::shared_ptr<::beads_gym::environment::reward::StayCloseReward<Eigen::Vector3d>>>(m, "StayCloseReward")
      .def(py::init<std::map<size_t, std::vector<double>>&>(), py::arg("reference_positions"));
}
