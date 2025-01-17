// This file is part of Bembel, the higher order C++ boundary element library.
// It was written as part of a cooperation of J. Doelz, H. Harbrecht, S. Kurz,
// M. Multerer, S. Schoeps, and F. Wolf at Technische Universitaet Darmstadt,
// Universitaet Basel, and Universita della Svizzera italiana, Lugano. This
// source code is subject to the GNU General Public License version 3 and
// provided WITHOUT ANY WARRANTY, see <http://www.bembel.eu> for further
// information.
#ifndef BEMBEL_INCLUDE_CONSTANTS_H_
#define BEMBEL_INCLUDE_CONSTANTS_H_

namespace Bembel {
namespace Constants {

////////////////////////////////////////////////////////////////////////////////
/// variables
////////////////////////////////////////////////////////////////////////////////
template <typename ptScalar> ptScalar generic_tolerance = 1e-6;
// some not further specified constant
constexpr int MaxP = 20;

// some not further specified constant
constexpr int maximum_quadrature_degree = 50;
// constants for the mesh refinement
// constexpr removed as not all numbers can be constexpr (posits)
template <typename ptScalar>
ptScalar corners[2][4] = {{0., 1., 1., 0}, {0., 0., 1., 1.}};
template <typename ptScalar>
ptScalar llcs[2][4] = {{0., .5, .5, .0}, {0., 0., .5, .5}};
template <typename ptScalar>
ptScalar edgemps[2][5] = {{.5, 1., .5, 0, .5}, {0., .5, 1., .5, .5}};
// tolerance for point comparison to determine patch topology
template <typename ptScalar> ptScalar pt_comp_tolerance = 1e-9;
// realloc size must be bigger than 4
constexpr size_t Bembel_alloc_size = 100;
// the interpolation problem solved during the assembly of the projector needs
// to filter some almost-zero coefficients that might be introduced during the
// solution of the linear system
template <typename ptScalar> ptScalar projector_tolerance = 1e-4;
////////////////////////////////////////////////////////////////////////////////
/// methods
////////////////////////////////////////////////////////////////////////////////

template <typename ptScalar> inline bool isAlmostZero(ptScalar in) {
  return in < generic_tolerance<ptScalar> && in > -generic_tolerance<ptScalar>;
}
} // namespace Constants
} // namespace Bembel

#endif
