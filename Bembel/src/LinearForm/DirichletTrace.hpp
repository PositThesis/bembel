// This file is part of Bembel, the higher order C++ boundary element library.
// It was written as part of a cooperation of J. Doelz, H. Harbrecht, S. Kurz,
// M. Multerer, S. Schoeps, and F. Wolf at Technische Universitaet Darmstadt,
// Universitaet Basel, and Universita della Svizzera italiana, Lugano. This
// source code is subject to the GNU General Public License version 3 and
// provided WITHOUT ANY WARRANTY, see <http://www.bembel.eu> for further
// information.
#ifndef BEMBEL_LINEARFORM_DIRICHLETTRACE_H_
#define BEMBEL_LINEARFORM_DIRICHLETTRACE_H_

namespace Bembel {

template <typename Scalar, typename ptScalar>
class DirichletTrace;

template <typename ScalarT, typename ptScalar>
struct LinearFormTraits<DirichletTrace<ScalarT, ptScalar>> {
  typedef ScalarT Scalar;
};

/**
 *  \ingroup LinearForm
 *  \brief This class provides an implementation of the Dirichlet trace operator
 * and a corresponding method to evaluate the linear form corresponding to the
 * right hand side of the system via quadrature.
 */
template <typename Scalar, typename ptScalar>
class DirichletTrace : public LinearFormBase<DirichletTrace<Scalar, ptScalar>, Scalar, ptScalar> {
 public:
  DirichletTrace() {}
  void set_function(const std::function<Scalar(Eigen::Matrix<ptScalar, 3, 1>)> &function) {
    function_ = function;
  }
  template <class T>
  void evaluateIntegrand_impl(
      const T &super_space, const SurfacePoint<ptScalar> &p,
      Eigen::Matrix<Scalar, Eigen::Dynamic, 1> *intval) const {
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
    auto x_kappa = x_f_dx.cross(x_f_dy).norm();

    // integrand without basis functions
    auto integrand = function_(x_f) * x_kappa * ws;

    // multiply basis functions with integrand
    super_space.addScaledBasis(intval, integrand, s);

    return;
  };

 private:
  std::function<Scalar(Eigen::Matrix<ptScalar, 3, 1>)> function_;
};
}  // namespace Bembel

#endif
