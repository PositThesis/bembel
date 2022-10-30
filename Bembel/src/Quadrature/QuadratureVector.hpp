// This file is part of Bembel, the higher order C++ boundary element library.
// It was written as part of a cooperation of J. Doelz, H. Harbrecht, S. Kurz,
// M. Multerer, S. Schoeps, and F. Wolf at Technische Universitaet Darmstadt,
// Universitaet Basel, and Universita della Svizzera italiana, Lugano. This
// source code is subject to the GNU General Public License version 3 and
// provided WITHOUT ANY WARRANTY, see <http://www.bembel.eu> for further
// information.

#ifndef BEMBEL_QUADRATURE_QUADRATUREVECTOR_H_
#define BEMBEL_QUADRATURE_QUADRATUREVECTOR_H_

namespace Bembel {

/**
 *  \ingroup Quadrature
 *  \brief this struct wraps all the defined quadrature Rules in a nice
 *         structure overloading the [] operator such that they can
 *         be accessed within a loop during runtime
 **/
template <template <unsigned int qrOrder, typename qrptScalar> class QuadratureRule,
          unsigned int Order, typename ptScalar>
struct QuadratureVector {
  QuadratureVector() {
    QuadratureRule<Order + 1, ptScalar> QR;
    Q_.xi_ = Eigen::Map<Eigen::Matrix<ptScalar, Eigen::Dynamic, 1>>(QR.xi_.data(), QR.xi_.size());
    Q_.w_ = Eigen::Map<Eigen::Matrix<ptScalar, Eigen::Dynamic, 1>>(QR.w_.data(), QR.w_.size());
  }
  QuadratureVector<QuadratureRule, Order - 1, ptScalar> remainingQuadratures_;
  const Quadrature<1, ptScalar> &operator[](unsigned int i) const {
    return (i == Order) ? Q_ : remainingQuadratures_[i];
  }

  Quadrature<1, ptScalar> Q_;
};

template <template <unsigned int qrOrder, typename qrptScalar> class QuadratureRule, typename ptScalar>
struct QuadratureVector<QuadratureRule, 0, ptScalar> {
  QuadratureVector() {
    QuadratureRule<1, ptScalar> QR;
    Q_.xi_ = Eigen::Map<Eigen::Matrix<ptScalar, Eigen::Dynamic, 1>>(QR.xi_.data(), QR.xi_.size());
    Q_.w_ = Eigen::Map<Eigen::Matrix<ptScalar, Eigen::Dynamic, 1>>(QR.w_.data(), QR.w_.size());
  }
  Quadrature<1, ptScalar> Q_;
  const Quadrature<1, ptScalar> &operator[](unsigned int i) const { return Q_; }
};
}  // namespace Bembel
#endif
