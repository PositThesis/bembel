// This file is part of Bembel, the higher order C++ boundary element library.
// It was written as part of a cooperation of J. Doelz, H. Harbrecht, S. Kurz,
// M. Multerer, S. Schoeps, and F. Wolf at Technische Universitaet Darmstadt,
// Universitaet Basel, and Universita della Svizzera italiana, Lugano. This
// source code is subject to the GNU General Public License version 3 and
// provided WITHOUT ANY WARRANTY, see <http://www.bembel.eu> for further
// information.
#ifndef BEMBEL_LINEAROPERATOR_MAXWELL_MAXWELLSINGLELAYERPOTENTIAL_H_
#define BEMBEL_LINEAROPERATOR_MAXWELL_MAXWELLSINGLELAYERPOTENTIAL_H_

namespace Bembel {
// forward declaration of class MaxwellSingleLayerPotential in order to define
// traits
template <typename LinOp, typename ptScalar>
class MaxwellSingleLayerPotential;

template <typename LinOp, typename ptScalar>
struct PotentialTraits<MaxwellSingleLayerPotential<LinOp, ptScalar>> {
  typedef typename Eigen::Matrix<std::complex<ptScalar>, Eigen::Dynamic, 1>::Scalar Scalar;
  static constexpr int OutputSpaceDimension = 3;
};

/**
 * \ingroup Maxwell
 */
template <typename LinOp, typename ptScalar>
class MaxwellSingleLayerPotential
    : public PotentialBase<MaxwellSingleLayerPotential<LinOp, ptScalar>, LinOp, ptScalar> {
  // implementation of the kernel evaluation, which may be based on the
  // information available from the superSpace
 public:
  MaxwellSingleLayerPotential() {}
  Eigen::Matrix<typename PotentialReturnScalar<
                    typename LinearOperatorTraits<LinOp>::Scalar,
                    std::complex<ptScalar>>::Scalar,
                3, 1>
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

    // compute surface measures from tangential derivatives
    auto h = element.get_h();

    // evaluate kernel
    auto kernel = evaluateKernel(point, x_f);
    auto kernel_gradient = evaluateKernelGrad(point, x_f);

    // assemble Galerkin solution
    auto scalar_part = fun_ev.evaluate(element, p);
    auto divergence_part = fun_ev.evaluateDiv(element, p);

    // integrand without basis functions, note that the surface measure
    // disappears for the divergence
    // auto integrand = kernel * scalar_part * ws;
    auto integrand =
        (kernel * scalar_part +
         ptScalar(1.) / wavenumber2_ * kernel_gradient * divergence_part) *
        ws;

    return integrand;
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
  /**
   * \brief Gradient of fundamental solution of Helmholtz problem
   */
  Eigen::VectorX<std::complex<ptScalar>> evaluateKernelGrad(const Eigen::Matrix<ptScalar, 3, 1> &x,
                                      const Eigen::Matrix<ptScalar, 3, 1> &y) const {
    auto c = x - y;
    auto r = c.norm();
    auto r3 = r * r * r;
    auto i = std::complex<ptScalar>(0., 1.);
    return (std::exp(-i * wavenumber_ * r) * (ptScalar(-1.) - i * wavenumber_ * r) / ptScalar(4.) /
            ptScalar(BEMBEL_PI) / r3) *
           c;
  }
  //////////////////////////////////////////////////////////////////////////////
  //    setters
  //////////////////////////////////////////////////////////////////////////////
  void set_wavenumber(std::complex<ptScalar> wavenumber) {
    wavenumber_ = wavenumber;
    wavenumber2_ = wavenumber_ * wavenumber_;
  }
  //////////////////////////////////////////////////////////////////////////////
  //    getters
  //////////////////////////////////////////////////////////////////////////////
  std::complex<ptScalar> get_wavenumber() { return wavenumber_; }

 private:
  std::complex<ptScalar> wavenumber_;
  std::complex<ptScalar> wavenumber2_;
};

}  // namespace Bembel
#endif
