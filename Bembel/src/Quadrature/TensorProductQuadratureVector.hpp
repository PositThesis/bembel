// This file is part of Bembel, the higher order C++ boundary element library.
// It was written as part of a cooperation of J. Doelz, H. Harbrecht, S. Kurz,
// M. Multerer, S. Schoeps, and F. Wolf at Technische Universitaet Darmstadt,
// Universitaet Basel, and Universita della Svizzera italiana, Lugano. This
// source code is subject to the GNU General Public License version 3 and
// provided WITHOUT ANY WARRANTY, see <http://www.bembel.eu> for further
// information.

#ifndef BEMBEL_QUADRATURE_TENSORPRODUCTQUADRATUREVECTOR_H_
#define BEMBEL_QUADRATURE_TENSORPRODUCTQUADRATUREVECTOR_H_

namespace Bembel {


/**
 *  \ingroup Quadrature
 *  \todo add a desciption
 */
template <template <unsigned int qrOrder, typename qrptScalar> class QuadratureRule,
          unsigned int Order, typename ptScalar>
struct TensorProductQuadratureVector {
  TensorProductQuadratureVector() {
    QuadratureRule<Order + 1, ptScalar> GL;
    Q_.xi_.resize(2, GL.xi_.size() * GL.xi_.size());
    Q_.w_.resize(GL.w_.size() * GL.w_.size());
    for (auto k = 0; k < Q_.xi_.cols(); ++k) {
      Q_.xi_.col(k) << GL.xi_[k / GL.xi_.size()], GL.xi_[k % GL.xi_.size()];
      Q_.w_(k) = GL.w_[k / GL.w_.size()] * GL.w_[k % GL.w_.size()];
    }
  }
  Cubature<ptScalar> Q_;
  TensorProductQuadratureVector<QuadratureRule, Order - 1, ptScalar> remainingQuadratures_;
  const Cubature<ptScalar> &operator[](unsigned int i) const {
    return (i == Order) ? Q_ : remainingQuadratures_[i];
  }
};

/**
 *  \ingroup Quadrature
 *  \todo add a desciption
 */
template <template <unsigned int qrOrder, typename qrptScalar> class QuadratureRule, typename ptScalar>
struct TensorProductQuadratureVector<QuadratureRule, 0, ptScalar> {
  TensorProductQuadratureVector() {
    QuadratureRule<1, ptScalar> GL;
    Q_.xi_.resize(2, GL.xi_.size() * GL.xi_.size());
    Q_.w_.resize(GL.w_.size() * GL.w_.size());
    for (auto k = 0; k < Q_.xi_.cols(); ++k) {
      Q_.xi_.col(k) << GL.xi_[k / GL.xi_.size()], GL.xi_[k % GL.xi_.size()];
      Q_.w_(k) = GL.w_[k / GL.w_.size()] * GL.w_[k % GL.w_.size()];
    }
  }
  Cubature<ptScalar> Q_;
  const Cubature<ptScalar> &operator[](unsigned int i) const { return Q_; }
};

}  // namespace Bembel
#endif
