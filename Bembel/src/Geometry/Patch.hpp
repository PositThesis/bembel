// This file is part of Bembel, the higher order C++ boundary element library.
// It was written as part of a cooperation of J. Doelz, H. Harbrecht, S. Kurz,
// M. Multerer, S. Schoeps, and F. Wolf at Technische Universitaet Darmstadt,
// Universitaet Basel, and Universita della Svizzera italiana, Lugano. This
// source code is subject to the GNU General Public License version 3 and
// provided WITHOUT ANY WARRANTY, see <http://www.bembel.eu> for further
// information.
#ifndef BEMBEL_GEOMETRY_PATCH_H_
#define BEMBEL_GEOMETRY_PATCH_H_
namespace Bembel {

/**
 *  \ingroup Geometry
 *  \class Patch
 *  \brief handles a single patch
 **/
template <typename ptScalar>
class Patch {
 public:
  //////////////////////////////////////////////////////////////////////////////
  /// constructors
  //////////////////////////////////////////////////////////////////////////////
  Patch(){};
  Patch(const std::vector<Eigen::Matrix<ptScalar, -1, -1>> &control_points,
        const std::vector<ptScalar> &knots_x,
        const std::vector<ptScalar> &knots_y) {
    init_Patch(control_points, knots_x, knots_y);
  }
  //////////////////////////////////////////////////////////////////////////////
  /// init
  //////////////////////////////////////////////////////////////////////////////
  inline void init_Patch(const std::vector<Eigen::Matrix<ptScalar, -1, -1>> &xyzw,
                         const std::vector<ptScalar> &x_knots,
                         const std::vector<ptScalar> &y_knots) {
    assert(xyzw.size() == 4);
    const int xyzw_cols = xyzw[0].cols();
    const int xyzw_rows = xyzw[0].rows();
    unique_knots_x_ = Spl::ExtractUniqueKnotVector(x_knots);
    unique_knots_y_ = Spl::ExtractUniqueKnotVector(y_knots);
    const int xnumpatch = unique_knots_x_.size() - 1;
    const int ynumpatch = unique_knots_y_.size() - 1;
    polynomial_degree_x_ = x_knots.size() - xyzw_cols;
    polynomial_degree_y_ = y_knots.size() - xyzw_rows;
    data_.resize(4 * (polynomial_degree_x_ * xnumpatch * polynomial_degree_y_ *
                      ynumpatch));
    {
      // Since its only for initialization, I do not care about speed.
      // Here I look weather the data given is already in bezier form.
      if (unique_knots_x_.size() == 2 && unique_knots_y_.size() == 2) {
        for (int i = 0; i < 4; i++) {
          Eigen::Matrix<ptScalar, -1, 1> tmp = Spl::Unroll(xyzw[i]);
          for (int j = 0; j < tmp.rows(); j++) data_[j * 4 + i] = (tmp[j]);
        }
      } else {
        // If not, I construct the dynamic projection (i.e. solve
        // systems for
        // the coeffs) and project to the superspace.

        Eigen::SparseMatrix<ptScalar> phi = Spl::MakeProjection<ptScalar, ptScalar>(
            x_knots, y_knots, unique_knots_x_, unique_knots_y_,
            polynomial_degree_x_, polynomial_degree_y_);

        for (int i = 0; i < 4; i++) {
          Eigen::Matrix<ptScalar, -1, 1> tmp =
              Spl::Unroll(xyzw[i]).transpose() * phi.transpose();

          for (int j = 0; j < tmp.rows(); j++) data_[j * 4 + i] = (tmp[j]);
        }
      }
    }

    return;
  }
  /* eval() evaluates the geometry. I look up the position in the knot vector,
   * scale the input arguments, evaluate the 1D basis functions and sum over
   * them with the controll points from data. */

  Eigen::Matrix<ptScalar, 3, 1> eval(const Eigen::Matrix<ptScalar, 2, 1> &reference_point) const {
    const int x_location =
        Spl::FindLocationInKnotVector(reference_point(0), unique_knots_x_);
    const int y_location =
        Spl::FindLocationInKnotVector(reference_point(1), unique_knots_y_);
    const int numy = (unique_knots_y_.size() - 1) * polynomial_degree_y_;
    const ptScalar scaledx =
        Spl::Rescale(reference_point(0), unique_knots_x_[x_location],
                     unique_knots_x_[x_location + 1]);
    const ptScalar scaledy =
        Spl::Rescale(reference_point(1), unique_knots_y_[y_location],
                     unique_knots_y_[y_location + 1]);

    ptScalar *xbasis = new ptScalar[polynomial_degree_x_];
    ptScalar *ybasis = new ptScalar[polynomial_degree_y_];

    Bembel::Basis::ShapeFunctionHandler<ptScalar>::evalBasis(polynomial_degree_x_ - 1,
                                                   xbasis, scaledx);
    Bembel::Basis::ShapeFunctionHandler<ptScalar>::evalBasis(polynomial_degree_y_ - 1,
                                                   ybasis, scaledy);

    ptScalar tmp[4] = {0., 0., 0., 0.};

    for (int i = 0; i < polynomial_degree_x_; i++) {
      for (int j = 0; j < polynomial_degree_y_; j++) {
        const ptScalar tpbasisval = xbasis[i] * ybasis[j];
        const int accs = 4 * (numy * (polynomial_degree_x_ * x_location + i) +
                              polynomial_degree_y_ * y_location + j);
#pragma omp simd
        for (int k = 0; k < 4; k++) tmp[k] += data_[accs + k] * tpbasisval;
      }
    }

    delete[] xbasis;
    delete[] ybasis;

    Eigen::Matrix<ptScalar, 3, 1> out(tmp[0], tmp[1], tmp[2]);
    // Rescaling by the NRBS weight, i.e. projection to 3D from 4D hom

    return out / tmp[3];
  }

  Eigen::Matrix<ptScalar, 3, 2> evalJacobian(
      const Eigen::Matrix<ptScalar, 2, 1> &reference_point) const {
    const int x_location =
        Spl::FindLocationInKnotVector(reference_point(0), unique_knots_x_);
    const int y_location =
        Spl::FindLocationInKnotVector(reference_point(1), unique_knots_y_);
    const int numy = (unique_knots_y_.size() - 1) * polynomial_degree_y_;
    const ptScalar scaledx =
        Spl::Rescale(reference_point(0), unique_knots_x_[x_location],
                     unique_knots_x_[x_location + 1]);
    const ptScalar scaledy =
        Spl::Rescale(reference_point(1), unique_knots_y_[y_location],
                     unique_knots_y_[y_location + 1]);

    ptScalar *xbasis = new ptScalar[polynomial_degree_x_];
    ptScalar *ybasis = new ptScalar[polynomial_degree_y_];
    ptScalar *xbasisD = new ptScalar[polynomial_degree_x_];
    ptScalar *ybasisD = new ptScalar[polynomial_degree_y_];

    Bembel::Basis::ShapeFunctionHandler<ptScalar>::evalBasis(polynomial_degree_x_ - 1,
                                                   xbasis, scaledx);
    Bembel::Basis::ShapeFunctionHandler<ptScalar>::evalBasis(polynomial_degree_y_ - 1,
                                                   ybasis, scaledy);
    Bembel::Basis::ShapeFunctionHandler<ptScalar>::evalDerBasis(polynomial_degree_x_ - 1,
                                                      xbasisD, scaledx);
    Bembel::Basis::ShapeFunctionHandler<ptScalar>::evalDerBasis(polynomial_degree_y_ - 1,
                                                      ybasisD, scaledy);

    ptScalar tmp[4] = {0., 0., 0., 0.};
    ptScalar tmpDx[4] = {0., 0., 0., 0.};
    ptScalar tmpDy[4] = {0., 0., 0., 0.};

    for (int i = 0; i < polynomial_degree_x_; i++) {
      for (int j = 0; j < polynomial_degree_y_; j++) {
        const ptScalar tpbasisval = xbasis[i] * ybasis[j];
        const ptScalar tpbasisvalDx = xbasisD[i] * ybasis[j];
        const ptScalar tpbasisvalDy = xbasis[i] * ybasisD[j];
        const int accs = 4 * (numy * (polynomial_degree_x_ * x_location + i) +
                              polynomial_degree_y_ * y_location + j);

        // Here I add up the values of the basis functions in the dc
        // basis
#pragma omp simd
        for (int k = 0; k < 4; k++) {
          tmp[k] += data_[accs + k] * tpbasisval;
          tmpDx[k] += data_[accs + k] * tpbasisvalDx;
          tmpDy[k] += data_[accs + k] * tpbasisvalDy;
        }
      }
    }

    delete[] xbasis;
    delete[] ybasis;
    delete[] xbasisD;
    delete[] ybasisD;

    Eigen::Matrix<ptScalar, 3, 2> out;

    // Eigen::Matrix<ptScalar, 3, 1> out;

    ptScalar bot = 1. / (tmp[3] * tmp[3]);

#pragma omp simd
    for (int k = 0; k < 3; k++) {
      out(k, 0) = (tmpDx[k] * tmp[3] - tmp[k] * tmpDx[3]) * bot;
      out(k, 1) = (tmpDy[k] * tmp[3] - tmp[k] * tmpDy[3]) * bot;
    }

    return out;
  }

  inline Eigen::Matrix<ptScalar, 3, 1> evalNormal(
      const Eigen::Matrix<ptScalar, 2, 1> &reference_point) const {
    Eigen::Matrix<ptScalar, 3, 2> jac = evalJacobian(reference_point);
    return jac.col(0).cross(jac.col(1));
  }

  // Wrapper for legacy code
  inline Eigen::Matrix<ptScalar, 3, 1> eval(ptScalar x, ptScalar y) const {
    return eval(Eigen::Matrix<ptScalar, 2, 1>(x, y));
  }
  inline Eigen::Matrix<ptScalar, 3, 2> evalJacobian(ptScalar x, ptScalar y) const {
    return evalJacobian(Eigen::Matrix<ptScalar, 2, 1>(x, y));
  }
  inline Eigen::Matrix<ptScalar, 3, 1> evalNormal(ptScalar x, ptScalar y) const {
    return evalNormal(Eigen::Matrix<ptScalar, 2, 1>(x, y));
  }

  // This is a combination of eval und evalJacobian, to avoid duplication of
  // work. See SurfacePoint.hpp
  void updateSurfacePoint(Eigen::Matrix<ptScalar, 12, 1> *srf_pt,
                          const Eigen::Matrix<ptScalar, 2, 1> &ref_pt, ptScalar w,
                          const Eigen::Matrix<ptScalar, 2, 1> &xi) const {
    const int x_location =
        Spl::FindLocationInKnotVector(ref_pt(0), unique_knots_x_);
    const int y_location =
        Spl::FindLocationInKnotVector(ref_pt(1), unique_knots_y_);
    const int numy = (unique_knots_y_.size() - 1) * polynomial_degree_y_;
    const ptScalar scaledx = Spl::Rescale(ref_pt(0), unique_knots_x_[x_location],
                                        unique_knots_x_[x_location + 1]);
    const ptScalar scaledy = Spl::Rescale(ref_pt(1), unique_knots_y_[y_location],
                                        unique_knots_y_[y_location + 1]);

    ptScalar *buffer =
        new ptScalar[2 * (polynomial_degree_x_ + polynomial_degree_y_) + 12];
    for (int i = 0; i < 12; ++i) buffer[i] = 0;

    ptScalar *tmp = buffer;
    ptScalar *tmpDx = tmp + 4;
    ptScalar *tmpDy = tmpDx + 4;
    ptScalar *xbasis = tmpDy + 4;
    ptScalar *ybasis = xbasis + polynomial_degree_x_;
    ptScalar *xbasisD = ybasis + polynomial_degree_y_;
    ptScalar *ybasisD = xbasisD + polynomial_degree_x_;

    Bembel::Basis::ShapeFunctionHandler<ptScalar>::evalBasis(polynomial_degree_x_ - 1,
                                                   xbasis, scaledx);
    Bembel::Basis::ShapeFunctionHandler<ptScalar>::evalBasis(polynomial_degree_y_ - 1,
                                                   ybasis, scaledy);
    Bembel::Basis::ShapeFunctionHandler<ptScalar>::evalDerBasis(polynomial_degree_x_ - 1,
                                                      xbasisD, scaledx);
    Bembel::Basis::ShapeFunctionHandler<ptScalar>::evalDerBasis(polynomial_degree_y_ - 1,
                                                      ybasisD, scaledy);

    for (int i = 0; i < polynomial_degree_x_; ++i) {
      for (int j = 0; j < polynomial_degree_y_; ++j) {
        const ptScalar tpbasisval = xbasis[i] * ybasis[j];
        const ptScalar tpbasisvalDx = xbasisD[i] * ybasis[j];
        const ptScalar tpbasisvalDy = xbasis[i] * ybasisD[j];
        const int accs = 4 * (numy * (polynomial_degree_x_ * x_location + i) +
                              polynomial_degree_y_ * y_location + j);

        // Here I add up the values of the basis functions in the dc
        // basis
        for (int k = 0; k < 4; ++k) {
          tmp[k] += data_[accs + k] * tpbasisval;
          tmpDx[k] += data_[accs + k] * tpbasisvalDx;
          tmpDy[k] += data_[accs + k] * tpbasisvalDy;
        }
      }
    }

    const ptScalar bot = 1. / tmp[3];
    const ptScalar botsqr = bot * bot;

    (*srf_pt)(0) = xi(0);
    (*srf_pt)(1) = xi(1);
    (*srf_pt)(2) = w;
    (*srf_pt)(3) = tmp[0] * bot;
    (*srf_pt)(4) = tmp[1] * bot;
    (*srf_pt)(5) = tmp[2] * bot;
    (*srf_pt)(6) = (tmpDx[0] * tmp[3] - tmp[0] * tmpDx[3]) * botsqr;
    (*srf_pt)(7) = (tmpDx[1] * tmp[3] - tmp[1] * tmpDx[3]) * botsqr;
    (*srf_pt)(8) = (tmpDx[2] * tmp[3] - tmp[2] * tmpDx[3]) * botsqr;
    (*srf_pt)(9) = (tmpDy[0] * tmp[3] - tmp[0] * tmpDy[3]) * botsqr;
    (*srf_pt)(10) = (tmpDy[1] * tmp[3] - tmp[1] * tmpDy[3]) * botsqr;
    (*srf_pt)(11) = (tmpDy[2] * tmp[3] - tmp[2] * tmpDy[3]) * botsqr;
    delete[] buffer;
    return;
  }

  //////////////////////////////////////////////////////////////////////////////
  /// getter
  //////////////////////////////////////////////////////////////////////////////

  std::vector<ptScalar> data_;  // Controllpoints in Bezier-Extracted Format.
  int polynomial_degree_x_;   // Degree in x
  int polynomial_degree_y_;   // Degree in y
  std::vector<ptScalar>
      unique_knots_x_;  // The knot vectors, where each knot is unique
  std::vector<ptScalar>
      unique_knots_y_;  // The knot vectors, where each knot is unique
};

template <typename ptScalar>
inline std::vector<Patch<ptScalar>> PatchShredder(const Patch<ptScalar> &patch) noexcept {
  // Already a Bezier patch
  if (patch.unique_knots_y_.size() == 2 && patch.unique_knots_x_.size() == 2) {
    return {patch};
  }

  // number of subpatches in x and y directions
  const int xchips = patch.unique_knots_x_.size() - 1;
  const int ychips = patch.unique_knots_y_.size() - 1;

  const int xp = patch.polynomial_degree_x_;
  const int yp = patch.polynomial_degree_y_;
  const int numy = ychips * yp;

  std::vector<Patch<ptScalar>> out(xchips * ychips);

  for (int ix = 0; ix < xchips; ix++) {
    for (int iy = 0; iy < ychips; iy++) {
      const int index = ix * ychips + iy;

      out[index].unique_knots_x_ = {0, 1};
      out[index].unique_knots_y_ = {0, 1};
      out[index].polynomial_degree_x_ = xp;
      out[index].polynomial_degree_y_ = yp;
      out[index].data_.reserve(xp * yp * 4);
    }
  }

  for (int ix = 0; ix < xchips; ix++) {
    for (int iy = 0; iy < ychips; iy++) {
      const int index = ix * ychips + iy;
      for (int jx = 0; jx < xp; jx++) {
        for (int jy = 0; jy < yp; jy++) {
          const int accs = 4 * (numy * (xp * ix + jx) + yp * iy + jy);
          for (int k = 0; k < 4; k++) {
            out[index].data_.push_back(patch.data_[accs + k]);
          }
        }
      }
    }
  }

  return out;
}

// Shredds a whole vector of Patches
template <typename ptScalar>
inline std::vector<Patch<ptScalar>> PatchShredder(
    const std::vector<Patch<ptScalar>> &patches) noexcept {
  std::vector<Patch<ptScalar>> out;
  const int input_size = patches.size();

  for (int i = 0; i < input_size; i++) {
    std::vector<Patch<ptScalar>> tmp = PatchShredder(patches[i]);

    const int tmp_size = tmp.size();

    for (int j = 0; j < tmp_size; j++) out.push_back(tmp[j]);
  }

  out.shrink_to_fit();

  return out;
}

}  // namespace Bembel

#endif
