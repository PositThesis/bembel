// This file is part of Bembel, the higher order C++ boundary element library.
// It was written as part of a cooperation of J. Doelz, H. Harbrecht, S. Kurz,
// M. Multerer, S. Schoeps, and F. Wolf at Technische Universitaet Darmstadt,
// Universitaet Basel, and Universita della Svizzera italiana, Lugano. This
// source code is subject to the GNU General Public License version 3 and
// provided WITHOUT ANY WARRANTY, see <http://www.bembel.eu> for further
// information.
//
#ifndef BEMBEL_LINEAROPERATOR_MAXWELL_MAXWELLSINGLELAYEROPERATOR_H_
#define BEMBEL_LINEAROPERATOR_MAXWELL_MAXWELLSINGLELAYEROPERATOR_H_

namespace Bembel {
// forward declaration of class MaxwellSingleLayerOperator in order to define
// traits
template <typename ptScalar>
class MaxwellSingleLayerOperator;

template <typename ptScalar>
struct LinearOperatorTraits<MaxwellSingleLayerOperator<ptScalar>> {
  typedef Eigen::Matrix<std::complex<ptScalar>, Eigen::Dynamic, 1> EigenType;
  typedef typename Eigen::Matrix<std::complex<ptScalar>, Eigen::Dynamic, 1>::Scalar Scalar;
  enum {
    OperatorOrder = -1,
    Form = DifferentialForm::DivConforming,
    NumberOfFMMComponents = 2
  };
};

/**
 * \ingroup Maxwell
 */
template <typename ptScalar>
class MaxwellSingleLayerOperator
    : public LinearOperatorBase<MaxwellSingleLayerOperator<ptScalar>, ptScalar> {
  // implementation of the kernel evaluation, which may be based on the
  // information available from the superSpace
 public:
  MaxwellSingleLayerOperator() {}
  template <class T>
  void evaluateIntegrand_impl(
      const T &super_space, const SurfacePoint<ptScalar> &p1, const SurfacePoint<ptScalar> &p2,
      Eigen::Matrix<
          typename LinearOperatorTraits<MaxwellSingleLayerOperator>::Scalar,
          Eigen::Dynamic, Eigen::Dynamic> *intval) const {
    auto polynomial_degree = super_space.get_polynomial_degree();
    auto polynomial_degree_plus_one_squared =
        (polynomial_degree + 1) * (polynomial_degree + 1);

    // get evaluation points on unit square
    Eigen::Matrix<ptScalar, 2, 1> s = p1.segment(0, 2);
    Eigen::Matrix<ptScalar, 2, 1> t = p2.segment(0, 2);

    // get quadrature weights
    auto ws = p1(2);
    auto wt = p2(2);

    // get points on geometry and tangential derivatives
    Eigen::Matrix<ptScalar, 3, 1> x_f = p1.segment(3, 3);
    Eigen::Matrix<ptScalar, 3, 1> x_f_dx = p1.segment(6, 3);
    Eigen::Matrix<ptScalar, 3, 1> x_f_dy = p1.segment(9, 3);
    Eigen::Matrix<ptScalar, 3, 1> y_f = p2.segment(3, 3);
    Eigen::Matrix<ptScalar, 3, 1> y_f_dx = p2.segment(6, 3);
    Eigen::Matrix<ptScalar, 3, 1> y_f_dy = p2.segment(9, 3);

    // compute surface measures from tangential derivatives
    ptScalar h = ptScalar(1.) / (1 << super_space.get_refinement_level());  // h = 1 ./ (2^M)

    // integrand without basis functions
    auto kernel_evaluation = evaluateKernel(x_f, y_f) * ws * wt;
    auto integrand_vector = kernel_evaluation;
    auto integrand_divergence = -kernel_evaluation / wavenumber2_ / h / h;

    // vector part: mulitply shape functions with integrand and add to buffer
    super_space.addScaledVectorBasisInteraction(intval, integrand_vector, s, t,
                                                x_f_dx, x_f_dy, y_f_dx, y_f_dy);

    // divergence part: multiply shape functions with integrand and add to
    // buffer
    super_space.addScaledVectorBasisDivergenceInteraction(
        intval, integrand_divergence, s, t);

    return;
  }

  Eigen::Matrix<std::complex<ptScalar>, 4, 4> evaluateFMMInterpolation_impl(
      const SurfacePoint<ptScalar> &p1, const SurfacePoint<ptScalar> &p2) const {
    // get evaluation points on unit square
    Eigen::Matrix<ptScalar, 2, 1> s = p1.segment(0, 2);
    Eigen::Matrix<ptScalar, 2, 1> t = p2.segment(0, 2);

    // get points on geometry and tangential derivatives
    Eigen::Matrix<ptScalar, 3, 1> x_f = p1.segment(3, 3);
    Eigen::Matrix<ptScalar, 3, 1> x_f_dx = p1.segment(6, 3);
    Eigen::Matrix<ptScalar, 3, 1> x_f_dy = p1.segment(9, 3);
    Eigen::Matrix<ptScalar, 3, 1> y_f = p2.segment(3, 3);
    Eigen::Matrix<ptScalar, 3, 1> y_f_dx = p2.segment(6, 3);
    Eigen::Matrix<ptScalar, 3, 1> y_f_dy = p2.segment(9, 3);

    // evaluate kernel
    auto kernel = evaluateKernel(x_f, y_f);

    // interpolation
    Eigen::Matrix<std::complex<ptScalar>, 4, 4> intval;
    intval.setZero();
    intval(0, 0) = kernel * x_f_dx.dot(y_f_dx);
    intval(0, 2) = kernel * x_f_dx.dot(y_f_dy);
    intval(2, 0) = kernel * x_f_dy.dot(y_f_dx);
    intval(2, 2) = kernel * x_f_dy.dot(y_f_dy);
    intval(1, 1) = -kernel / wavenumber2_;
    intval(1, 3) = -kernel / wavenumber2_;
    intval(3, 1) = -kernel / wavenumber2_;
    intval(3, 3) = -kernel / wavenumber2_;

    return intval;
  }

  /**
   * \brief Fundamental solution of Helmholtz/Maxwell problem
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

/**
 * \brief The Maxwell single layer operator requires a special treatment of the
 * moment matrices of the FMM due to the involved derivatives on the ansatz
 * functions.
 */
template <typename InterpolationPoints, typename ptScalar>
struct H2Multipole::Moment2D<InterpolationPoints, MaxwellSingleLayerOperator<ptScalar>, ptScalar> {
  static std::vector<Eigen::Matrix<ptScalar, Eigen::Dynamic, Eigen::Dynamic>> compute2DMoment(
      const SuperSpace<MaxwellSingleLayerOperator<ptScalar>, ptScalar> &super_space,
      const int cluster_level, const int cluster_refinements,
      const int number_of_points) {
    Eigen::Matrix<ptScalar, Eigen::Dynamic, Eigen::Dynamic> moment = moment2DComputer<
        Moment1D<InterpolationPoints, MaxwellSingleLayerOperator<ptScalar>, ptScalar>,
        Moment1D<InterpolationPoints, MaxwellSingleLayerOperator<ptScalar>, ptScalar>, MaxwellSingleLayerOperator<ptScalar>, ptScalar>(
        super_space, cluster_level, cluster_refinements, number_of_points);
    Eigen::Matrix<ptScalar, Eigen::Dynamic, Eigen::Dynamic> moment_dx = moment2DComputer<
        Moment1DDerivative<InterpolationPoints, MaxwellSingleLayerOperator<ptScalar>, ptScalar>,
        Moment1D<InterpolationPoints, MaxwellSingleLayerOperator<ptScalar>, ptScalar>, MaxwellSingleLayerOperator<ptScalar>, ptScalar>(
        super_space, cluster_level, cluster_refinements, number_of_points);
    Eigen::Matrix<ptScalar, Eigen::Dynamic, Eigen::Dynamic> moment_dy = moment2DComputer<
        Moment1D<InterpolationPoints, MaxwellSingleLayerOperator<ptScalar>, ptScalar>,
        Moment1DDerivative<InterpolationPoints, MaxwellSingleLayerOperator<ptScalar>, ptScalar>, MaxwellSingleLayerOperator<ptScalar>, ptScalar>(
        super_space, cluster_level, cluster_refinements, number_of_points);

    Eigen::Matrix<ptScalar, Eigen::Dynamic, Eigen::Dynamic> moment1(moment.rows() + moment_dx.rows(), moment.cols());
    moment1 << moment, moment_dx;
    Eigen::Matrix<ptScalar, Eigen::Dynamic, Eigen::Dynamic> moment2(moment.rows() + moment_dy.rows(), moment.cols());
    moment2 << moment, moment_dy;

    std::vector<Eigen::Matrix<ptScalar, Eigen::Dynamic, Eigen::Dynamic>> vector_of_moments;
    vector_of_moments.push_back(moment1);
    vector_of_moments.push_back(moment2);

    return vector_of_moments;
  }
};

}  // namespace Bembel
#endif
