// This file is part of Bembel, the higher order C++ boundary element library.
// It was written as part of a cooperation of J. Doelz, H. Harbrecht, S. Kurz,
// M. Multerer, S. Schoeps, and F. Wolf at Technische Universitaet Darmstadt,
// Universitaet Basel, and Universita della Svizzera italiana, Lugano. This
// source code is subject to the GNU General Public License version 3 and
// provided WITHOUT ANY WARRANTY, see <http://www.bembel.eu> for further
// information.
#ifndef BEMBEL_INCLUDE_SUPERSPACE_H_
#define BEMBEL_INCLUDE_SUPERSPACE_H_
namespace Bembel {
/**
 *  \ingroup AnsatzSpace
 *  \brief The superspace manages local polynomial bases on each element of the
 * mesh and provides an itnerface to evaluate them.
 */
template <typename Derived, typename ptScalar>
struct SuperSpace {
  typedef typename LinearOperatorTraits<Derived>::Scalar Scalar;
  //////////////////////////////////////////////////////////////////////////////
  //    constructors
  //////////////////////////////////////////////////////////////////////////////
  SuperSpace(){};
  SuperSpace(Geometry<ptScalar>& geom, int M, int P) { init_SuperSpace(geom, M, P); }
  SuperSpace(const SuperSpace& other) {
    mesh_ = other.mesh_;
    phi = other.phi;
    phiDx = other.phiDx;
    phiPhi = other.phiPhi;
    phiPhiDx = other.phiPhiDx;
    phiPhiDy = other.phiPhiDy;
    phiTimesPhi = other.phiTimesPhi;
    // vPhiScalVPhi = other.vPhiScalVPhi;
    divPhiTimesDivPhi = other.divPhiTimesDivPhi;
    polynomial_degree = other.polynomial_degree;
    polynomial_degree_plus_one_squared =
        other.polynomial_degree_plus_one_squared;
  }
  SuperSpace(SuperSpace&& other) {
    mesh_ = other.mesh_;
    phi = other.phi;
    phiDx = other.phiDx;
    phiPhi = other.phiPhi;
    phiPhiDx = other.phiPhiDx;
    phiPhiDy = other.phiPhiDy;
    phiTimesPhi = other.phiTimesPhi;
    // vPhiScalVPhi = other.vPhiScalVPhi;
    divPhiTimesDivPhi = other.divPhiTimesDivPhi;
    polynomial_degree = other.polynomial_degree;
    polynomial_degree_plus_one_squared =
        other.polynomial_degree_plus_one_squared;
  };
  SuperSpace& operator=(SuperSpace other) {
    mesh_ = other.mesh_;
    phi = other.phi;
    phiDx = other.phiDx;
    phiPhi = other.phiPhi;
    phiPhiDx = other.phiPhiDx;
    phiPhiDy = other.phiPhiDy;
    phiTimesPhi = other.phiTimesPhi;
    // vPhiScalVPhi = other.vPhiScalVPhi;
    divPhiTimesDivPhi = other.divPhiTimesDivPhi;
    polynomial_degree = other.polynomial_degree;
    polynomial_degree_plus_one_squared =
        other.polynomial_degree_plus_one_squared;
    return *this;
  }
  //////////////////////////////////////////////////////////////////////////////
  //    getters
  //////////////////////////////////////////////////////////////////////////////
  int get_polynomial_degree() const { return polynomial_degree; }
  int get_polynomial_degree_plus_one_squared() const {
    return polynomial_degree_plus_one_squared;
  }
  int get_refinement_level() const { return mesh_->get_max_level(); }
  int get_number_of_elements() const { return mesh_->get_number_of_elements(); }
  int get_number_of_patches() const { return mesh_->get_geometry().size(); }
  const PatchVector<ptScalar>& get_geometry() const { return mesh_->get_geometry(); }
  const ClusterTree<ptScalar>& get_mesh() const { return *mesh_; };
  //////////////////////////////////////////////////////////////////////////////
  //    init_SuperSpace
  //////////////////////////////////////////////////////////////////////////////
  void init_SuperSpace(const Geometry<ptScalar>& geom, int M, int P) {
    polynomial_degree = P;
    polynomial_degree_plus_one_squared =
        (polynomial_degree + 1) * (polynomial_degree + 1);
    phi = (Basis::BasisHandler<Scalar, ptScalar>::funPtrPhi(P));
    phiDx = (Basis::BasisHandler<Scalar, ptScalar>::funPtrPhiDx(P));
    phiPhi = (Basis::BasisHandler<Scalar, ptScalar>::funPtrPhiPhi(P));
    phiPhiDx = (Basis::BasisHandler<Scalar, ptScalar>::funPtrPhiPhiDx(P));
    phiPhiDy = (Basis::BasisHandler<Scalar, ptScalar>::funPtrPhiPhiDy(P));
    phiTimesPhi = (Basis::BasisHandler<Scalar, ptScalar>::funPtrPhiTimesPhi(P));
    // vPhiScalVPhi = (Basis::BasisHandler<typename
    // LinearOperatorTraits<Derived>::Scalar>::funPtrVPhiScalVPhi(P));
    divPhiTimesDivPhi =
        (Basis::BasisHandler<Scalar, ptScalar>::funPtrDivPhiTimesDivPhi(P));
    mesh_ = std::make_shared<ClusterTree<ptScalar>>();
    mesh_->init_ClusterTree(geom, M);
    mesh_->checkOrientation();
    return;
  };
  //////////////////////////////////////////////////////////////////////////////
  //    map2surface
  //////////////////////////////////////////////////////////////////////////////
  void map2surface(const ElementTreeNode<ptScalar>& e, const Eigen::Matrix<ptScalar, 2, 1>& xi,
                   ptScalar w, SurfacePoint<ptScalar>* surf_pt) const {
    Eigen::Matrix<ptScalar, 2, 1> st = e.llc_ + e.get_h() * xi;
    mesh_->get_geometry()[e.patch_].updateSurfacePoint(surf_pt, st, w, xi);
    return;
  }
  //////////////////////////////////////////////////////////////////////////////
  //    Methods
  //////////////////////////////////////////////////////////////////////////////
  /**
   * \brief Compute all products of local shape functions on the unit square at
   * coordinates s,t, scale by w and add to intval.
   */
  void addScaledBasisInteraction(
      Eigen::Matrix<Scalar, Eigen::Dynamic, Eigen::Dynamic>* intval,
      typename LinearOperatorTraits<Derived>::Scalar w,
      const Eigen::Matrix<ptScalar, 2, 1>& s, const Eigen::Matrix<ptScalar, 2, 1>& t) const {
    phiTimesPhi(intval, w, s, t);
  }
  /**
   * \brief Compute all products of local shape functions on the unit square at
   * coordinates s,t.
   */
  Eigen::Matrix<Scalar, Eigen::Dynamic, Eigen::Dynamic> basisInteraction(
      const Eigen::Matrix<ptScalar, 2, 1>& s, const Eigen::Matrix<ptScalar, 2, 1>& t) const {
    Eigen::Matrix<Scalar, Eigen::Dynamic, Eigen::Dynamic> intval(
        polynomial_degree_plus_one_squared, polynomial_degree_plus_one_squared);
    intval.setZero();
    phiTimesPhi(&intval, ptScalar(1.), s, t);
    return intval;
  }

  /**
   * \brief Compute all products of surface curls of local shape functions
   * on the unit square at coordinates s,t.
   */
  void addScaledSurfaceCurlInteraction(
      Eigen::Matrix<Scalar, Eigen::Dynamic, Eigen::Dynamic>* intval, Scalar w,
      const SurfacePoint<ptScalar>& p1, const SurfacePoint<ptScalar>& p2) const {
    // surface measures
    ptScalar kappa1 = (p1.segment<3>(6)).cross(p1.segment<3>(9)).norm();
    ptScalar kappa2 = (p2.segment<3>(6)).cross(p2.segment<3>(9)).norm();
    // compute basis functions's surface curl. Each column of s_curl is a basis
    // function's surface curl at point s.
    Eigen::Matrix<ptScalar, Eigen::Dynamic, Eigen::Dynamic> s_curl(3, polynomial_degree_plus_one_squared);
    s_curl = (1.0 / kappa1) *
             (-p1.segment<3>(6) * basisDy(p1.segment<2>(0)).transpose() +
              p1.segment<3>(9) * basisDx(p1.segment<2>(0)).transpose());
    Eigen::Matrix<ptScalar, Eigen::Dynamic, Eigen::Dynamic> t_curl(3, polynomial_degree_plus_one_squared);
    t_curl = (1.0 / kappa2) *
             (-p2.segment<3>(6) * basisDy(p2.segment<2>(0)).transpose() +
              p2.segment<3>(9) * basisDx(p2.segment<2>(0)).transpose());
    // inner product of surface curls of any two basis functions
    for (int j = 0; j < polynomial_degree_plus_one_squared; ++j)
      for (int i = 0; i < polynomial_degree_plus_one_squared; ++i)
        (*intval)(j * polynomial_degree_plus_one_squared + i) +=
            w * s_curl.col(i).dot(t_curl.col(j));
  }

  /**
   * \brief Compute all products of surface gradients of local shape functions
   * on the unit square at coordinates s,t.
   */
  void addScaledSurfaceGradientInteraction(
      Eigen::Matrix<Scalar, Eigen::Dynamic, Eigen::Dynamic>* intval, Scalar w,
      const SurfacePoint<ptScalar>& p1, const SurfacePoint<ptScalar>& p2) const {
    // inner product of surface gradients of any two basis functions equals to
    // inner product of surface curls of any two basis functions
    addScaledSurfaceCurlInteraction(intval, w, p1, p2);
  }

  /**
   * \brief Compute all scalar products of vector valued local shape functions
   * on the surface points with reference coordinates s,t, scale by w and add to
   * intval.
   */
  void addScaledVectorBasisInteraction(
      Eigen::Matrix<Scalar, Eigen::Dynamic, Eigen::Dynamic>* intval, Scalar w,
      const Eigen::Matrix<ptScalar, 2, 1>& s, const Eigen::Matrix<ptScalar, 2, 1>& t,
      const Eigen::Matrix<ptScalar, 3, 1> x_f_dx, const Eigen::Matrix<ptScalar, 3, 1> x_f_dy,
      const Eigen::Matrix<ptScalar, 3, 1> y_f_dx, const Eigen::Matrix<ptScalar, 3, 1> y_f_dy) const {
    auto basis_interaction = basisInteraction(s, t);
    intval->block(0, 0, polynomial_degree_plus_one_squared,
                  polynomial_degree_plus_one_squared) +=
        w * x_f_dx.dot(y_f_dx) * basis_interaction;
    intval->block(0, polynomial_degree_plus_one_squared,
                  polynomial_degree_plus_one_squared,
                  polynomial_degree_plus_one_squared) +=
        w * x_f_dx.dot(y_f_dy) * basis_interaction;
    intval->block(polynomial_degree_plus_one_squared, 0,
                  polynomial_degree_plus_one_squared,
                  polynomial_degree_plus_one_squared) +=
        w * x_f_dy.dot(y_f_dx) * basis_interaction;
    intval->block(polynomial_degree_plus_one_squared,
                  polynomial_degree_plus_one_squared,
                  polynomial_degree_plus_one_squared,
                  polynomial_degree_plus_one_squared) +=
        w * x_f_dy.dot(y_f_dy) * basis_interaction;
  }
  /**
   * \brief Compute all scalar products of vector valued local shape functions
   * on the surface points with reference coordinates s,t.
   */
  Eigen::Matrix<Scalar, Eigen::Dynamic, Eigen::Dynamic> vectorBasisInteraction(
      const Eigen::Matrix<ptScalar, 2, 1>& s, const Eigen::Matrix<ptScalar, 2, 1>& t,
      const Eigen::Matrix<ptScalar, 3, 1> x_f_dx, const Eigen::Matrix<ptScalar, 3, 1> x_f_dy,
      const Eigen::Matrix<ptScalar, 3, 1> y_f_dx, const Eigen::Matrix<ptScalar, 3, 1> y_f_dy) const {
    Eigen::Matrix<Scalar, Eigen::Dynamic, Eigen::Dynamic> intval(
        2 * polynomial_degree_plus_one_squared,
        2 * polynomial_degree_plus_one_squared);
    intval.setZero();
    addScaledVectorBasisInteraction(&intval, ptScalar(1.), s, t, x_f_dx, x_f_dy, y_f_dx,
                                    y_f_dy);
    return intval;
  }
  /**
   * \brief Compute all products of divergences of local shape functions on the
   * unit square at coordinates s,t, scale by w and add to intval.
   */
  void addScaledVectorBasisDivergenceInteraction(
      Eigen::Matrix<Scalar, Eigen::Dynamic, Eigen::Dynamic>* intval, Scalar w,
      const Eigen::Matrix<ptScalar, 2, 1>& s, const Eigen::Matrix<ptScalar, 2, 1>& t) const {
    divPhiTimesDivPhi(intval, w, s, t);
  }
  /**
   * \brief Compute all products of divergences of local shape functions on the
   * unit square at coordinates s,t.
   */
  Eigen::Matrix<Scalar, Eigen::Dynamic, Eigen::Dynamic>
  vectorBasisDivergenceInteraction(const Eigen::Matrix<ptScalar, 2, 1>& s,
                                   const Eigen::Matrix<ptScalar, 2, 1>& t) const {
    Eigen::Matrix<Scalar, Eigen::Dynamic, Eigen::Dynamic> intval(
        2 * polynomial_degree_plus_one_squared,
        2 * polynomial_degree_plus_one_squared);
    intval.setZero();
    divPhiTimesDivPhi(&intval, ptScalar(1.), s, t);
    return intval;
  }
  /**
   * \brief Evaluate local shape functions on the unit square at coordinate s,
   * scale by w and add to intval.
   */
  void addScaledBasis(Eigen::Matrix<Scalar, Eigen::Dynamic, 1>* intval,
                      Scalar w, const Eigen::Matrix<ptScalar, 2, 1>& s) const {
    phiPhi(intval, w, s);
  }
  /**
   * \brief Evaluate local shape functions on the unit square at coordinate s.
   */
  Eigen::Matrix<Scalar, Eigen::Dynamic, 1> basis(
      const Eigen::Matrix<ptScalar, 2, 1>& s) const {
    Eigen::Matrix<Scalar, Eigen::Dynamic, 1> intval(
        polynomial_degree_plus_one_squared);
    intval.setZero();
    phiPhi(&intval, ptScalar(1.), s);
    return intval;
  }
  /**
   * \brief Evaluate derivatives in x direction of local shape functions on the
   * unit square at coordinate s, scale by w and add to intval.
   */
  void addScaledBasisDx(Eigen::Matrix<Scalar, Eigen::Dynamic, 1>* intval,
                        typename LinearOperatorTraits<Derived>::Scalar w,
                        const Eigen::Matrix<ptScalar, 2, 1>& s) const {
    phiPhiDx(intval, w, s);
  }
  /**
   * \brief Evaluate derivatives in x direction of local shape functions on the
   * unit square at coordinate s.
   */
  Eigen::Matrix<Scalar, Eigen::Dynamic, 1> basisDx(
      const Eigen::Matrix<ptScalar, 2, 1>& s) const {
    Eigen::Matrix<Scalar, Eigen::Dynamic, 1> intval(
        polynomial_degree_plus_one_squared);
    intval.setZero();
    phiPhiDx(&intval, ptScalar(1.), s);
    return intval;
  }
  /**
   * \brief Evaluate derivatives in y direction of local shape functions on the
   * unit square at coordinate s, scale by w and add to intval.
   */
  void addScaledBasisDy(Eigen::Matrix<Scalar, Eigen::Dynamic, 1>* intval,
                        typename LinearOperatorTraits<Derived>::Scalar w,
                        const Eigen::Matrix<ptScalar, 2, 1>& s) const {
    phiPhiDy(intval, w, s);
  }
  /**
   * \brief Evaluate derivatives in y direction of local shape functions on the
   * unit square at coordinate s.
   */
  Eigen::Matrix<Scalar, Eigen::Dynamic, 1> basisDy(
      const Eigen::Matrix<ptScalar, 2, 1>& s) const {
    Eigen::Matrix<Scalar, Eigen::Dynamic, 1> intval(
        polynomial_degree_plus_one_squared);
    intval.setZero();
    phiPhiDy(&intval, ptScalar(1.), s);
    return intval;
  }
  /**
   * \brief Evaluate local shape functions on the unit interval at coordinate s,
   * scale by w and add to intval.
   */
  void addScaledBasis1D(Eigen::Matrix<Scalar, Eigen::Dynamic, 1>* intval,
                        Scalar w, ptScalar s) const {
    phi(intval, w, s);
  }
  /**
   * \brief Evaluate local shape functions on the unit interval at coordinate s.
   */
  Eigen::Matrix<Scalar, Eigen::Dynamic, 1> basis1D(ptScalar s) const {
    Eigen::Matrix<Scalar, Eigen::Dynamic, 1> intval(polynomial_degree + 1);
    intval.setZero();
    phi(&intval, ptScalar(1.), s);
    return intval;
  }
  /**
   * \brief Evaluate derivatives of local shape functions on the unit interval
   * at coordinate s, scale by w and add to intval.
   */
  void addScaledBasis1DDx(Eigen::Matrix<Scalar, Eigen::Dynamic, 1>* intval,
                          Scalar w, ptScalar s) const {
    phiDx(intval, w, s);
  }
  /**
   * \brief Evaluate derivatives of local shape functions on the unit interval
   * at coordinate s.
   */
  Eigen::Matrix<Scalar, Eigen::Dynamic, 1> basis1DDx(ptScalar s) const {
    Eigen::Matrix<Scalar, Eigen::Dynamic, 1> intval(polynomial_degree + 1);
    intval.setZero();
    phiDx(&intval, ptScalar(1.), s);
    return intval;
  }
  //////////////////////////////////////////////////////////////////////////////
  //    member variables
  //////////////////////////////////////////////////////////////////////////////
 private:
  std::shared_ptr<ClusterTree<ptScalar>> mesh_;
  Basis::funptr_phi<Scalar, ptScalar> phi;
  Basis::funptr_phidx<Scalar, ptScalar> phiDx;
  Basis::funptr_phiphi<Scalar, ptScalar> phiPhi;
  Basis::funptr_phiphidx<Scalar, ptScalar> phiPhiDx;
  Basis::funptr_phiphidy<Scalar, ptScalar> phiPhiDy;
  Basis::funptr_phitimesphi<Scalar, ptScalar> phiTimesPhi;
  // Basis::funptr_vphiscalvphi<typename LinearOperatorTraits<Derived>::Scalar>
  // vPhiScalVPhi;
  Basis::funptr_divphitimesdivphi<Scalar, ptScalar> divPhiTimesDivPhi;
  int polynomial_degree;
  int polynomial_degree_plus_one_squared;
};  // namespace Bembel
}  // namespace Bembel
#endif
