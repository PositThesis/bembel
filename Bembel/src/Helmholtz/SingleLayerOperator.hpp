// This file is part of Bembel, the higher order C++ boundary element library.
// It was written as part of a cooperation of J. Doelz, H. Harbrecht, S. Kurz,
// M. Multerer, S. Schoeps, and F. Wolf at Technische Universitaet Darmstadt,
// Universitaet Basel, and Universita della Svizzera italiana, Lugano. This
// source code is subject to the GNU General Public License version 3 and
// provided WITHOUT ANY WARRANTY, see <http://www.bembel.eu> for further
// information.
//
#ifndef BEMBEL_LINEAROPERATOR_HELMHOLTZ_HELMHOLTZSINGLELAYEROPERATOR_H_
#define BEMBEL_LINEAROPERATOR_HELMHOLTZ_HELMHOLTZSINGLELAYEROPERATOR_H_

namespace Bembel {
// forward declaration of class HelmholtzSingleLayerOperator in order to define
// traits
template <typename ptScalar>
class HelmholtzSingleLayerOperator;

template <typename ptScalar>
struct LinearOperatorTraits<HelmholtzSingleLayerOperator<ptScalar>> {
  using EigenType = Eigen::Matrix<std::complex<ptScalar>, Eigen::Dynamic, 1>;
  using Scalar = typename Eigen::Matrix<std::complex<ptScalar>, Eigen::Dynamic, 1>::Scalar;
  enum {
    OperatorOrder = -1,
    Form = DifferentialForm::Discontinuous,
    NumberOfFMMComponents = 1
  };
};

/**
 * \ingroup Helmholtz
 */
template <typename ptScalar>
class HelmholtzSingleLayerOperator
    : public LinearOperatorBase<HelmholtzSingleLayerOperator<ptScalar>, ptScalar> {
  // implementation of the kernel evaluation, which may be based on the
  // information available from the superSpace
 public:
  HelmholtzSingleLayerOperator() {}
  template <class T>
  void evaluateIntegrand_impl(
      const T &super_space, const SurfacePoint<ptScalar> &p1, const SurfacePoint<ptScalar> &p2,
      Eigen::Matrix<
          typename LinearOperatorTraits<HelmholtzSingleLayerOperator>::Scalar,
          Eigen::Dynamic, Eigen::Dynamic> *intval) const {
    auto polynomial_degree = super_space.get_polynomial_degree();
    auto polynomial_degree_plus_one_squared =
        (polynomial_degree + 1) * (polynomial_degree + 1);

    // get evaluation points on unit square
    Eigen::Matrix<ptScalar, 2, 1> s = p1.segment(0, 2);
    Eigen::Matrix<ptScalar, 2, 1> t = p2.segment(0, 2);

    // get quadrature weights
    auto ws = p1(2);
    auto wt = p2(2);

    // get points on geometry and tangential derivatives
    Eigen::Matrix<ptScalar, 3, 1> x_f = p1.segment(3, 3);
    Eigen::Matrix<ptScalar, 3, 1> x_f_dx = p1.segment(6, 3);
    Eigen::Matrix<ptScalar, 3, 1> x_f_dy = p1.segment(9, 3);
    Eigen::Matrix<ptScalar, 3, 1> y_f = p2.segment(3, 3);
    Eigen::Matrix<ptScalar, 3, 1> y_f_dx = p2.segment(6, 3);
    Eigen::Matrix<ptScalar, 3, 1> y_f_dy = p2.segment(9, 3);

    // compute surface measures from tangential derivatives
    auto x_kappa = x_f_dx.cross(x_f_dy).norm();
    auto y_kappa = y_f_dx.cross(y_f_dy).norm();

    // integrand without basis functions
    auto integrand = evaluateKernel(x_f, y_f) * x_kappa * y_kappa * ws * wt;

    // multiply basis functions with integrand and add to intval, this is an
    // efficient implementation of
    //(*intval) += super_space.BasisInteraction(s, t) * evaluateKernel(x_f, y_f)
    //* x_kappa * y_kappa * ws * wt;
    super_space.addScaledBasisInteraction(intval, integrand, s, t);

    return;
  }

  Eigen::Matrix<std::complex<ptScalar>, 1, 1> evaluateFMMInterpolation_impl(
      const SurfacePoint<ptScalar> &p1, const SurfacePoint<ptScalar> &p2) const {
    // get evaluation points on unit square
    Eigen::Matrix<ptScalar, 2, 1> s = p1.segment(0, 2);
    Eigen::Matrix<ptScalar, 2, 1> t = p2.segment(0, 2);

    // get points on geometry and tangential derivatives
    Eigen::Matrix<ptScalar, 3, 1> x_f = p1.segment(3, 3);
    Eigen::Matrix<ptScalar, 3, 1> x_f_dx = p1.segment(6, 3);
    Eigen::Matrix<ptScalar, 3, 1> x_f_dy = p1.segment(9, 3);
    Eigen::Matrix<ptScalar, 3, 1> y_f = p2.segment(3, 3);
    Eigen::Matrix<ptScalar, 3, 1> y_f_dx = p2.segment(6, 3);
    Eigen::Matrix<ptScalar, 3, 1> y_f_dy = p2.segment(9, 3);

    // compute surface measures from tangential derivatives
    auto x_kappa = x_f_dx.cross(x_f_dy).norm();
    auto y_kappa = y_f_dx.cross(y_f_dy).norm();

    // interpolation
    Eigen::Matrix<std::complex<ptScalar>, 1, 1> intval;
    intval(0) = evaluateKernel(x_f, y_f) * x_kappa * y_kappa;

    return intval;
  }

  /**
   * \brief Fundamental solution of Helmholtz problem
   */
  std::complex<ptScalar> evaluateKernel(const Eigen::Matrix<ptScalar, 3, 1> &x,
                                      const Eigen::Matrix<ptScalar, 3, 1> &y) const {
    auto r = (x - y).norm();
    return std::exp(-std::complex<ptScalar>(0., 1.) * wavenumber_ * r) / ptScalar(4.) /
           ptScalar(BEMBEL_PI) / r;
  }
  //////////////////////////////////////////////////////////////////////////////
  //    setters
  //////////////////////////////////////////////////////////////////////////////
  void set_wavenumber(std::complex<ptScalar> wavenumber) {
    wavenumber_ = wavenumber;
  }
  //////////////////////////////////////////////////////////////////////////////
  //    getters
  //////////////////////////////////////////////////////////////////////////////
  std::complex<ptScalar> get_wavenumber() { return wavenumber_; }

 private:
  std::complex<ptScalar> wavenumber_;
};

}  // namespace Bembel
#endif
