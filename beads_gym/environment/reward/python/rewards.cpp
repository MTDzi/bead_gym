#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <pybind11/stl_bind.h>
#include <pybind11/eigen.h>
#include <vector>
#include <map>
#include <optional>

#include "beads_gym/beads/bead.hpp"
#include "beads_gym/environment/reward/stay_close_reward.hpp"

namespace py = pybind11;


PYBIND11_MODULE(rewards, m) {
    using Bead = beads_gym::beads::Bead<Eigen::Vector3d>;
    using StayCloseReward = beads_gym::environment::reward::StayCloseReward<Eigen::Vector3d>;

    m.doc() = "reward module"; // optional module docstring

    // Class StayCloseReward
    py::class_<StayCloseReward, std::shared_ptr<StayCloseReward>>(m, "StayCloseReward")
      .def(
        py::init<std::map<size_t, std::vector<double>>&, std::vector<double>&>(), py::arg("reference_positions"), py::arg("weights")
      )
      .def_static("from_beads", [](py::iterable bead_list, std::optional<std::vector<double>> weights = std::nullopt) {
        std::vector<double> weights_;
        std::vector<std::shared_ptr<Bead>> vec;
        for (auto item : bead_list) {
            vec.push_back(item.cast<std::shared_ptr<Bead>>());
        }
        if (weights.has_value()) {
          weights_ = weights.value();
        } else {
          weights_ = std::vector<double>(vec.size(), 1.);
        }
        return std::make_shared<StayCloseReward>(vec, weights_);
      }, py::arg("reference_beads"), py::arg("weights"));
}
