// This file is part of Bembel, the higher order C++ boundary element library.
// It was written as part of a cooperation of J. Doelz, H. Harbrecht, S. Kurz,
// M. Multerer, S. Schoeps, and F. Wolf at Technische Universitaet Darmstadt,
// Universitaet Basel, and Universita della Svizzera italiana, Lugano. This
// source code is subject to the GNU General Public License version 3 and
// provided WITHOUT ANY WARRANTY, see <http://www.bembel.eu> for further
// information.
#ifndef BEMBEL_DUFFYTRICK_INTEGRATE0_H_
#define BEMBEL_DUFFYTRICK_INTEGRATE0_H_

namespace Bembel {
namespace DuffyTrick {
/**
 *  \ingroup DuffyTrick
 *    \brief far-field quadrature routine, which is based on precomputed values
 *           in order to quickly evaluate the integrand in the case that the
 *           far-field quadrature degree can be used
 **/
template <typename Derived, class T, typename ptScalar>
void integrate0(const LinearOperatorBase<Derived, ptScalar> &LinOp, const T &super_space,
                const ElementTreeNode<ptScalar> &e1, int rot1, const ElementTreeNode<ptScalar> &e2,
                int rot2, const Eigen::Matrix<ptScalar, Eigen::Dynamic, Eigen::Dynamic> &ffield_qnodes,
                const Cubature<ptScalar> &Q,
                Eigen::Matrix<typename LinearOperatorTraits<Derived>::Scalar,
                              Eigen::Dynamic, Eigen::Dynamic> *intval) {
  intval->setZero();
  for (auto i = 0; i < Q.w_.size(); ++i)
    for (auto j = 0; j < Q.w_.size(); ++j)
      LinOp.evaluateIntegrand(
          super_space, ffield_qnodes.col(e1.id_ * Q.w_.size() + i),
          ffield_qnodes.col(e2.id_ * Q.w_.size() + j), intval);
  BEMBEL_UNUSED_(rot1);
  BEMBEL_UNUSED_(rot2);
  BEMBEL_UNUSED_(Q);
  return;
}
}  // namespace Duffy
}  // namespace Bembel
#endif
