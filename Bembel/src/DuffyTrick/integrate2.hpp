// This file is part of Bembel, the higher order C++ boundary element library.
// It was written as part of a cooperation of J. Doelz, H. Harbrecht, S. Kurz,
// M. Multerer, S. Schoeps, and F. Wolf at Technische Universitaet Darmstadt,
// Universitaet Basel, and Universita della Svizzera italiana, Lugano. This
// source code is subject to the GNU General Public License version 3 and
// provided WITHOUT ANY WARRANTY, see <http://www.bembel.eu> for further
// information.
#ifndef BEMBEL_DUFFYTRICK_INTGRATE2_H_
#define BEMBEL_DUFFYTRICK_INTGRATE2_H_

namespace Bembel {
namespace DuffyTrick {
/**
 *  \ingroup DuffyTrick
 *    \brief quadrature routine for identical elements
 *    \todo  be sure that map2element computes the weight h*Q.w(i) such that
 *           the integrand may then be scaled by qp1.weight * qp2.weight
 *           here we just set one weight to the actual weight, while the
 *other one will be set to 1. This is to remain conforming to the structure
 *           of integrate0/1
 *           Information that map2element has to provide:
 *           xi; w; Chi(xi); dChidx(xi); dChidy(xi);
 **/
template <typename Derived, class T, typename ptScalar>
void integrate2(const LinearOperatorBase<Derived, ptScalar> &LinOp, const T &super_space,
                const ElementTreeNode<ptScalar> &e1, int rot1, const ElementTreeNode<ptScalar> &e2,
                int rot2, const Eigen::Matrix<ptScalar, Eigen::Dynamic, Eigen::Dynamic> &ffield_qnodes,
                const Cubature<ptScalar> &Q,
                Eigen::Matrix<typename LinearOperatorTraits<Derived>::Scalar,
                              Eigen::Dynamic, Eigen::Dynamic> *intval) {
  intval->setZero();
  ptScalar h = e1.get_h();
  ptScalar t1 = 0;
  ptScalar t2 = 0;
  ptScalar t3 = 0;
  ptScalar t4 = 0;
  Eigen::Matrix<ptScalar, 2, 8> pts;
  SurfacePoint<ptScalar> qp1, qp2;
  // llc of the element wrt [0,1]^2
  for (auto i = 0; i < Q.w_.size(); ++i) {
    ptScalar w = h * h * Q.w_(i) * Q.xi_(0, i) * (1 - Q.xi_(0, i)) *
               (1 - Q.xi_(0, i) * Q.xi_(1, i));
    for (auto j = 0; j < Q.w_.size(); ++j) {
      t1 = Q.xi_(0, j) * (1 - Q.xi_(0, i));
      t2 = Q.xi_(1, j) * (1 - Q.xi_(0, i) * Q.xi_(1, i));
      t3 = Q.xi_(0, j) * (1 - Q.xi_(0, i)) + Q.xi_(0, i);
      t4 = Q.xi_(1, j) * (1 - Q.xi_(0, i) * Q.xi_(1, i)) +
           Q.xi_(0, i) * Q.xi_(1, i);
      pts << t1, t3, t1, t3, t2, t4, t2, t4, t2, t4, t4, t2, t1, t3, t3, t1;
      for (auto k = 0; k < 4; ++k) {
        super_space.map2surface(e1, pts.col(2 * k), w, &qp1);
        super_space.map2surface(e1, pts.col(2 * k + 1), Q.w_(j), &qp2);
        LinOp.evaluateIntegrand(super_space, qp1, qp2, intval);
        LinOp.evaluateIntegrand(super_space, qp2, qp1, intval);
      }
    }
  }
  BEMBEL_UNUSED_(e2);
  BEMBEL_UNUSED_(rot1);
  BEMBEL_UNUSED_(rot2);
  BEMBEL_UNUSED_(ffield_qnodes);
  return;
}
}  // namespace Duffy
}  // namespace Bembel

#endif
