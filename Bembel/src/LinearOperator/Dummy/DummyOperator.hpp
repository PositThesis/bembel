// This file is part of Bembel, the higher order C++ boundary element library.
// It was written as part of a cooperation of J. Doelz, H. Harbrecht, S. Kurz,
// M. Multerer, S. Schoeps, and F. Wolf at Technische Universitaet Darmstadt,
// Universitaet Basel, and Universita della Svizzera italiana, Lugano. This
// source code is subject to the GNU General Public License version 3 and
// provided WITHOUT ANY WARRANTY, see <http://www.bembel.eu> for further
// information.
//
// This file is part of Bembel, the higher order C++ boundary element library.
// It was written as part of a cooperation of J. Doelz, H. Harbrecht, S. Kurz,
// M. Multerer, S. Schoeps, and F. Wolf at Technische Universitaet Darmstadt,
// Universitaet Basel, and Universita della Svizzera italiana, Lugano. This
// source code is subject to the GNU General Public License version 3 and
// provided WITHOUT ANY WARRANTY, see <http://www.bembel.eu> for further
// information.
//
#ifndef BEMBEL_LINEAROPERATOR_DUMMYOPERATOR_H_
#define BEMBEL_LINEAROPERATOR_DUMMYOPERATOR_H_

namespace Bembel {

template <typename ptScalar>
class DummyOperator;

template <typename ptScalar>
struct LinearOperatorTraits<DummyOperator<ptScalar>> {
  using EigenType = Eigen::Matrix<ptScalar, Eigen::Dynamic, 1>;
  using Scalar = typename Eigen::Matrix<ptScalar, Eigen::Dynamic, 1>::Scalar;
  enum { OperatorOrder = 0, Form = DifferentialForm::Discontinuous, NumberOfFMMComponents = 1 };
};

// forward declaration of class DummyOperator in order to define traits
// define some default test functiom
template <typename ptScalar>
std::function<ptScalar(const Eigen::Matrix<ptScalar, 2, 1> &, const Eigen::Matrix<ptScalar, 2, 1> &)>
    DummyOperator_test_function =
        [](const Eigen::Matrix<ptScalar, 2, 1> &x, const Eigen::Matrix<ptScalar, 2, 1> &y) { return 1.; };

/**
 *  \ingroup DummyOperator
 *  \brief This class provides a dummy specialization of the LinearOperator and
 * corresponding Traits for testing and debugging
 */
template <typename ptScalar>
class DummyOperator : public LinearOperatorBase<DummyOperator<ptScalar>, ptScalar> {
  // implementation of the kernel evaluation, which may be based on the
  // information available from the superSpace
 public:
  DummyOperator() { test_func_ = DummyOperator_test_function<ptScalar>; }
  DummyOperator(
      std::function<ptScalar(const Eigen::Matrix<ptScalar, 2, 1> &, const Eigen::Matrix<ptScalar, 2, 1> &)>
          test_func) {
    test_func_ = test_func;
  }
  template <class T>
  void evaluateIntegrand_impl(
      const T &super_space, const SurfacePoint<ptScalar> &p1, const SurfacePoint<ptScalar> &p2,
      Eigen::Matrix<typename LinearOperatorTraits<DummyOperator<ptScalar>>::Scalar,
                    Eigen::Dynamic, Eigen::Dynamic> *intval) const {
    (*intval)(0, 0) +=
        test_func_(p1.segment(3, 2), p2.segment(3, 2)) * p1(2) * p2(2);
    return;
  }

 private:
  std::function<ptScalar(const Eigen::Matrix<ptScalar, 2, 1> &, const Eigen::Matrix<ptScalar, 2, 1> &)>
      test_func_;
};

}  // namespace Bembel
#endif
