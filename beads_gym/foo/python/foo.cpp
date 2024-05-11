#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <pybind11/stl_bind.h>
#include "beads_gym/foo/foo.hpp"

namespace py = pybind11;

//PYBIND11_MAKE_OPAQUE(std::vector<std::string>);
//PYBIND11_MAKE_OPAQUE(std::vector<std::vector<std::string>>);
//PYBIND11_MAKE_OPAQUE(std::vector<std::pair<int, int>>);
//PYBIND11_MAKE_OPAQUE(std::vector<std::vector<std::pair<int, int>>>);

PYBIND11_MODULE(pyfoo, m) {
    m.doc() = "pyfoo module"; // optional module docstring

    // Free function
    m.def("free_function", py::overload_cast<int>(&::beads_gym::foo::freeFunction), "A free function taking an int.");
    m.def("free_function", py::overload_cast<int64_t>(&::beads_gym::foo::freeFunction), "A free function taking an int64.");

    // Vector of String
    m.def("string_vector_output", &::beads_gym::foo::stringVectorOutput, "A function that return a vector of string.");
    m.def("string_vector_input", &::beads_gym::foo::stringVectorInput, "A function that use a vector of string.");
    m.def("string_vector_ref_input", &::beads_gym::foo::stringVectorInput, "A function that use a vector of string const ref.");

    // Vector of Vector of String
    m.def("string_jagged_array_output", &::beads_gym::foo::stringJaggedArrayOutput, "A function that return a jagged array of string.");
    m.def("string_jagged_array_input", &::beads_gym::foo::stringJaggedArrayInput, "A function that use a jagged array of string.");
    m.def("string_jagged_array_ref_input", &::beads_gym::foo::stringJaggedArrayRefInput, "A function that use a jagged array of string const ref.");

    // Vector of Pair
    m.def("pair_vector_output", &::beads_gym::foo::pairVectorOutput, "A function that return a vector of pair.");
    m.def("pair_vector_input", &::beads_gym::foo::pairVectorInput, "A function that use a vector of pair.");
    m.def("pair_vector_ref_input", &::beads_gym::foo::pairVectorRefInput, "A function that use a vector of pair const ref.");

    // Vector of Vector of Pair
    m.def("pair_jagged_array_output", &::beads_gym::foo::pairJaggedArrayOutput, "A function that return a jagged array of pair.");
    m.def("pair_jagged_array_input", &::beads_gym::foo::pairJaggedArrayInput, "A function that use a jagged array of pair.");
    m.def("pair_jagged_array_ref_input", &::beads_gym::foo::pairJaggedArrayRefInput, "A function that use a jagged array of pair const ref.");

    // Class Foo
    py::class_<::beads_gym::foo::Foo>(m, "Foo")
      .def_static("static_function", py::overload_cast<int>(&::beads_gym::foo::Foo::staticFunction))
      .def_static("static_function", py::overload_cast<int64_t>(&::beads_gym::foo::Foo::staticFunction))
      .def(py::init<>())
      .def_property("int", &::beads_gym::foo::Foo::getInt, &::beads_gym::foo::Foo::setInt, py::return_value_policy::copy)
      .def_property("int64", &::beads_gym::foo::Foo::getInt64, &::beads_gym::foo::Foo::setInt64, py::return_value_policy::copy)
      .def("__str__", &::beads_gym::foo::Foo::operator());
}
