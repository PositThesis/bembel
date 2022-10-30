// This file is part of Bembel, the higher order C++ boundary element library.
// It was written as part of a cooperation of J. Doelz, H. Harbrecht, S. Kurz,
// M. Multerer, S. Schoeps, and F. Wolf at Technische Universitaet Darmstadt,
// Universitaet Basel, and Universita della Svizzera italiana, Lugano. This
// source code is subject to the GNU General Public License version 3 and
// provided WITHOUT ANY WARRANTY, see <http://www.bembel.eu> for further
// information.

#ifndef BEMBEL_QUADRATURE_QUADRATURE_H_
#define BEMBEL_QUADRATURE_QUADRATURE_H_

namespace Bembel {

template <unsigned int Dimension, typename ptScalar>
struct Quadrature {
  Eigen::Matrix<ptScalar, Dimension, Eigen::Dynamic> xi_;
  Eigen::Matrix<ptScalar, Eigen::Dynamic, 1> w_;
};

}  // namespace Bembel
#endif
