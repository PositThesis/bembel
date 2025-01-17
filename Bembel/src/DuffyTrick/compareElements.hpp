// This file is part of Bembel, the higher order C++ boundary element library.
// It was written as part of a cooperation of J. Doelz, H. Harbrecht, S. Kurz,
// M. Multerer, S. Schoeps, and F. Wolf at Technische Universitaet Darmstadt,
// Universitaet Basel, and Universita della Svizzera italiana, Lugano. This
// source code is subject to the GNU General Public License version 3 and
// provided WITHOUT ANY WARRANTY, see <http://www.bembel.eu> for further
// information.

#ifndef BEMBEL_DUFFYTRICK_COMPAREELEMENTS_H_
#define BEMBEL_DUFFYTRICK_COMPAREELEMENTS_H_

namespace Bembel {
namespace DuffyTrick {
/**
 *  \ingroup DuffyTrick
 *  \brief compares two elements for similarities and determines, how the
 *         elements have to be rotated to move the similarity to the first
 *         vertices_ or edge
 **/
template <typename ptScalar>
Eigen::Vector3i compareElements(const ElementTreeNode<ptScalar> &e1,
                                const ElementTreeNode<ptScalar> &e2, ptScalar *dist) {
  Eigen::Vector3i retval;
  retval.setZero();
  // check if the two elements are identical and directly return;
  if (std::addressof(e1) == std::addressof(e2)) {
    retval << 4, 4, 2;
    *dist = 0;
    return retval;
  } else {
    // if they are not identical, check if they have a positive distance
    // check for common vertices
    *dist = (e1.midpoint_ - e2.midpoint_).norm() - e1.radius_ - e2.radius_;
    *dist = *dist >= 0 ? *dist : 0;
    // check if elements are distinct and return now
    if (*dist > .5 / (1 << e1.level_)) {
      retval << 4, 4, 0;
      return retval;
      // otherwise check for common edge/vertex. Note that there is a
      // short circuit: either two elements share a single edge or
      // single point. everything else will break the code
    } else {
      for (auto rot1 = 0; rot1 < 4; ++rot1)
        for (auto rot2 = 0; rot2 < 4; ++rot2)
          // check for common vertices_
          if (e1.vertices_[rot1] == e2.vertices_[rot2]) {
            // if there is a common vertices_, check for common edge
            if (e1.vertices_[3] == e2.vertices_[(rot2 + 1) % 4]) {
              retval << 3, rot2, 3;
              return retval;
            } else if (e1.vertices_[(rot1 + 1) % 4] ==
                       e2.vertices_[(rot2 + 3) % 4]) {
              retval << rot1, (rot2 + 3) % 4, 3;
              return retval;
            } else {
              retval << rot1, rot2, 4;
              return retval;
            }
          }
      retval << 4, 4, 0;
      return retval;
    }
  }
  // if you ended up here, something went terribly wrong
  retval << 4, 4, -1;
  return retval;
}
}  // namespace DuffyTrick
}  // namespace Bembel

#endif
