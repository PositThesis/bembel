// This file is part of Bembel, the higher order C++ boundary element library.
// It was written as part of a cooperation of J. Doelz, H. Harbrecht, S. Kurz,
// M. Multerer, S. Schoeps, and F. Wolf at Technische Universitaet Darmstadt,
// Universitaet Basel, and Universita della Svizzera italiana, Lugano. This
// source code is subject to the GNU General Public License version 3 and
// provided WITHOUT ANY WARRANTY, see <http://www.bembel.eu> for further
// information.
//
#ifndef BEMBEL_TEST_DUFFYTRICK_INTEGRATE1_H_
#define BEMBEL_TEST_DUFFYTRICK_INTEGRATE1_H_

namespace Test {
namespace DuffyTrick {

template <typename Derived, unsigned int maxqdeg, typename ptScalar>
bool test_integrate1(const Bembel::AnsatzSpace<Derived, ptScalar> &ansatz_space,
                     const Bembel::LinearOperatorBase<Derived, ptScalar> &linOp) {
  Bembel::GaussSquare<maxqdeg + 1, ptScalar> GS;
  auto Q = GS[maxqdeg];
  Eigen::MatrixXd ffield_qnodes(0, 0);
  Eigen::MatrixXd intval;
  Eigen::MatrixXd axis;
  intval.resize(1, 1);
  axis.resize(2, 4);
  auto begin =
      ansatz_space.get_superspace().get_mesh().get_element_tree().cpbegin();
  auto end =
      ansatz_space.get_superspace().get_mesh().get_element_tree().cpend();
  double h = begin->get_h();
  double maxError = 0;
  double error = 0;

  for (auto it = begin; it != end; ++it)
    for (auto it2 = begin; it2 != end; ++it2) {
      intval.setZero();
      error = 0;
      Bembel::DuffyTrick::integrate1(linOp, ansatz_space.get_superspace(), *it,
                                     0, *it2, 0, ffield_qnodes, Q, &intval);
      axis.col(0) << it->llc_(0), it->llc_(0) + h;
      axis.col(1) << it->llc_(1), it->llc_(1) + h;
      axis.col(2) << it2->llc_(0), it2->llc_(0) + h;
      axis.col(3) << it2->llc_(1), it2->llc_(1) + h;
      auto exactInt =
          integrate1_test_function_integral(axis) / BEMBEL_SQUARED_(h);
      error = std::abs(exactInt - intval(0, 0)) / std::abs(exactInt);
      maxError = error > maxError ? error : maxError;
    }
  std::cout << maxError << std::endl;
  return maxError < Constants::integrate1_tolerance;
}
}  // namespace DuffyTrick
}  // namespace Test
#endif
