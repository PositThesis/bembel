// This file is part of Bembel, the higher order C++ boundary element library.
// It was written as part of a cooperation of J. Doelz, H. Harbrecht, S. Kurz,
// M. Multerer, S. Schoeps, and F. Wolf at Technische Universitaet Darmstadt,
// Universitaet Basel, and Universita della Svizzera italiana, Lugano. This
// source code is subject to the GNU General Public License version 3 and
// provided WITHOUT ANY WARRANTY, see <http://www.bembel.eu> for further
// information.
#ifndef BEMBEL_LINEAROPERATOR_LAPLACE_LAPLACESINGLELAYERPOTENTIAL_H_
#define BEMBEL_LINEAROPERATOR_LAPLACE_LAPLACESINGLELAYERPOTENTIAL_H_

namespace Bembel {
// forward declaration of class LaplaceSingleLayerPotential in order to define
// traits
template <typename LinOp, typename ptScalar>
class LaplaceSingleLayerPotential;

template <typename LinOp, typename ptScalar>
struct PotentialTraits<LaplaceSingleLayerPotential<LinOp, ptScalar>> {
  using Scalar = typename Eigen::Matrix<ptScalar, Eigen::Dynamic, 1>::Scalar;
  static constexpr int OutputSpaceDimension = 1;
};

/**
 * \ingroup Laplace
 */
template <typename LinOp, typename ptScalar>
class LaplaceSingleLayerPotential
    : public PotentialBase<LaplaceSingleLayerPotential<LinOp, ptScalar>, LinOp, ptScalar> {
  // implementation of the kernel evaluation, which may be based on the
  // information available from the superSpace
 public:
  LaplaceSingleLayerPotential() {}
  Eigen::Matrix<
      typename PotentialReturnScalar<
          typename LinearOperatorTraits<LinOp>::Scalar, ptScalar>::Scalar,
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
  ptScalar evaluateKernel(const Eigen::Matrix<ptScalar, 3, 1> &x,
                        const Eigen::Matrix<ptScalar, 3, 1> &y) const {
    return ptScalar(1.) / ptScalar(4.) / ptScalar(BEMBEL_PI) / (x - y).norm();
  }
};

}  // namespace Bembel
#endif
