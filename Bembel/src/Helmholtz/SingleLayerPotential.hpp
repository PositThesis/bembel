// This file is part of Bembel, the higher order C++ boundary element library.
// It was written as part of a cooperation of J. Doelz, H. Harbrecht, S. Kurz,
// M. Multerer, S. Schoeps, and F. Wolf at Technische Universitaet Darmstadt,
// Universitaet Basel, and Universita della Svizzera italiana, Lugano. This
// source code is subject to the GNU General Public License version 3 and
// provided WITHOUT ANY WARRANTY, see <http://www.bembel.eu> for further
// information.
#ifndef BEMBEL_LINEAROPERATOR_HELMHOLTZ_HELMHOLTZSINGLELAYERPOTENTIAL_H_
#define BEMBEL_LINEAROPERATOR_HELMHOLTZ_HELMHOLTZSINGLELAYERPOTENTIAL_H_

namespace Bembel {
// forward declaration of class HelmholtzSingleLayerPotential in order to define
// traits
template <typename LinOp, typename ptScalar>
class HelmholtzSingleLayerPotential;

template <typename LinOp, typename ptScalar>
struct PotentialTraits<HelmholtzSingleLayerPotential<LinOp, ptScalar>> {
  typedef typename Eigen::Matrix<std::complex<ptScalar>, Eigen::Dynamic, 1>::Scalar Scalar;
  static constexpr int OutputSpaceDimension = 1;
};

/**
 * \ingroup Helmholtz
 */
template <typename LinOp, typename ptScalar>
class HelmholtzSingleLayerPotential
    : public PotentialBase<HelmholtzSingleLayerPotential<LinOp, ptScalar>, LinOp, ptScalar> {
  // implementation of the kernel evaluation, which may be based on the
  // information available from the superSpace
 public:
  HelmholtzSingleLayerPotential() {}
  Eigen::Matrix<typename PotentialReturnScalar<
                    typename LinearOperatorTraits<LinOp>::Scalar,
                    std::complex<ptScalar>>::Scalar,
                1, 1>
  evaluateIntegrand_impl(const FunctionEvaluator<LinOp, ptScalar> &fun_ev,
                         const ElementTreeNode<ptScalar> &element,
                         const Eigen::Matrix<ptScalar, 3, 1> &point,
                         const SurfacePoint<ptScalar> &p) const {
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

    // evaluate kernel
    auto kernel = evaluateKernel(point, x_f);

    // assemble Galerkin solution
    auto cauchy_value = fun_ev.evaluate(element, p);

    // integrand without basis functions
    auto integrand = kernel * cauchy_value * x_kappa * ws;

    return integrand;
  }

  /**
   * \brief Fundamental solution of Laplace problem
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
