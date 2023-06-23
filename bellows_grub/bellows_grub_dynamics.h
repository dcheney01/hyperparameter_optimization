/*--------------- MANUAL CHANGES -------------------
	need to update return type in both the function name and function body. This will likely be py::array. Ensure this matches the .c file
	make sure input vars for each function are correct type (e.g. std::vector). Ensure this matches the .c file.
	make sure code doesn't contain invalid c syntax that slipped through
*/


#include <pybind11/stl.h>
#include <pybind11/numpy.h>
#include <pybind11/eigen.h>
#include <Eigen/LU>
#include <Eigen/SVD>

namespace py = pybind11;

#ifndef TEST_PROJECT__BELLOWS_GRUB_DYNAMICS__H
#define TEST_PROJECT__BELLOWS_GRUB_DYNAMICS__H

Eigen::MatrixXd calc_state_derivs(const Eigen::MatrixXd &X, const Eigen::MatrixXd &U, double m, double stiffness, double damping, double alpha);
std::vector<double> calc_M(std::vector<double> state, double h, double m, double r);
std::vector<double> calc_C(std::vector<double> state, std::vector<double> stateDot, double h, double m, double r);
std::vector<double> calc_grav(std::vector<double> state, double h, double m, double g);
std::vector<double> fkEnd(std::vector<double> state, double h, double l);
py::array calc_regressor(std::vector<double> state, std::vector<double> stateDot, std::vector<double> qdr, std::vector<double> qddr, double h, double m, double r, double g);

#endif

