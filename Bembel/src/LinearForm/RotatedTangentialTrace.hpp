// This file is part of Bembel, the higher order C++ boundary element library.
// It was written as part of a cooperation of J. Doelz, H. Harbrecht, S. Kurz,
// M. Multerer, S. Schoeps, and F. Wolf at Technische Universitaet Darmstadt,
// Universitaet Basel, and Universita della Svizzera italiana, Lugano. This
// source code is subject to the GNU General Public License version 3 and
// provided WITHOUT ANY WARRANTY, see <http://www.bembel.eu> for further
// information.
#ifndef BEMBEL_INCLUDE_TRACEOPERATORS_ROTATEDTANGENTIALTRACE_H_
#define BEMBEL_INCLUDE_TRACEOPERATORS_ROTATEDTANGENTIALTRACE_H_

#include "LinearForm.hpp"

namespace Bembel {

template <typename Scalar, typename ptScalar>
class RotatedTangentialTrace;

template <typename ScalarT, typename ptScalar>
struct LinearFormTraits<RotatedTangentialTrace<ScalarT, ptScalar>> {
  typedef ScalarT Scalar;
};

/**
 *    \ingroup LinearForm
 *    \brief This class provides a specialization of the linear form required
 *for the solution of the electric field integral equation.
 **/
template <typename Scalar, typename ptScalar>
class RotatedTangentialTrace
    : public LinearFormBase<RotatedTangentialTrace<Scalar, ptScalar>, Scalar, ptScalar> {
 public:
  RotatedTangentialTrace() {}
  void set_function(
      const std::function<Eigen::Matrix<Scalar, 3, 1>(Eigen::Matrix<ptScalar, 3, 1>)>
          &function) {
    function_ = function;
  }
  template <class T>
  void evaluateIntegrand_impl(
      const T &super_space, const SurfacePoint<ptScalar> &p,
      Eigen::Matrix<Scalar, Eigen::Dynamic, 2> *intval) const {
    auto polynomial_degree = super_space.get_polynomial_degree();
    auto polynomial_degree_plus_one_squared =
        (polynomial_degree + 1) * (polynomial_degree + 1);

    // get evaluation points on unit square
    Eigen::Matrix<ptScalar, 2, 1> s = p.segment(0, 2);

    // get quadrature weights
    auto ws = p(2);

    // get points on geometry and tangential derivatives
    Eigen::Matrix<ptScalar, 3, 1> x_f = p.segment(3, 3);
    Eigen::Matrix<ptScalar, 3, 1> x_f_dx = p.segment(6, 3);
    Eigen::Matrix<ptScalar, 3, 1> x_f_dy = p.segment(9, 3);

    // compute surface measures from tangential derivatives
    auto x_n = x_f_dx.cross(x_f_dy).normalized();

    // tangential component + quadrature weights
    // use n x f x n = f-<f,n>n to avoid troubles with -flto flag in combination
    // of .cross()
    auto fun_x_f = function_(x_f);
    auto tangential_component = (fun_x_f - fun_x_f.dot(x_n) * x_n) * ws;

    // extract tangential component
    auto component_x = x_f_dx.dot(tangential_component);
    auto component_y = x_f_dy.dot(tangential_component);

    // evaluate shape functions
    auto phiPhiVec = super_space.basis(s);

    // multiply basis functions with integrand
    Eigen::Matrix<Scalar, Eigen::Dynamic, 2> phiPhiMat(
        polynomial_degree_plus_one_squared, 2);
    phiPhiMat.col(0) = component_x * phiPhiVec;
    phiPhiMat.col(1) = component_y * phiPhiVec;

    // compute integrals
    (*intval) += phiPhiMat;
    return;
  };

 private:
  std::function<Eigen::Matrix<Scalar, 3, 1>(Eigen::Matrix<ptScalar, 3, 1>)> function_;
};
}  // namespace Bembel

#endif
