// This file is part of Bembel, the higher order C++ boundary element library.
// It was written as part of a cooperation of J. Doelz, H. Harbrecht, S. Kurz,
// M. Multerer, S. Schoeps, and F. Wolf at Technische Universitaet Darmstadt,
// Universitaet Basel, and Universita della Svizzera italiana, Lugano. This
// source code is subject to the GNU General Public License version 3 and
// provided WITHOUT ANY WARRANTY, see <http://www.bembel.eu> for further
// information.
#ifndef BEMBEL_CLUSTERTREE_CLUSTERTREE_H_
#define BEMBEL_CLUSTERTREE_CLUSTERTREE_H_

namespace Bembel {

/**
 *  \ingroup ClusterTree
 *  \brief The ClusterTree class introduces an element structure on a Geometry
 * object. Note that we do not introduce a mesh in the classical sense, but only
 * introduce a system of local coordinates via an ElementTree.
 */
template <typename ptScalar>
class ClusterTree {
 public:
  // we declare functionality which has not been implemented (yet)
  // to be private
  ClusterTree(const ClusterTree<ptScalar>& other) = delete;
  ClusterTree(ClusterTree<ptScalar>&& other) = delete;
  ClusterTree& operator=(const ClusterTree<ptScalar>& other) = delete;
  ClusterTree& operator=(ClusterTree<ptScalar>&& other) = delete;
  //////////////////////////////////////////////////////////////////////////////
  /// constructors
  //////////////////////////////////////////////////////////////////////////////
  ClusterTree() {}
  ClusterTree(const Geometry<ptScalar>& geom, int M) { init_ClusterTree(geom, M); }
  //////////////////////////////////////////////////////////////////////////////
  /// init
  //////////////////////////////////////////////////////////////////////////////
  void init_ClusterTree(const Geometry<ptScalar>& geom, int M) {
    element_tree_.init_ElementTree(geom, M);
    points_ = element_tree_.computeElementEnclosings();
    return;
  }
  //////////////////////////////////////////////////////////////////////////////
  /// getter
  //////////////////////////////////////////////////////////////////////////////
  ElementTree<ptScalar>& get_element_tree() { return element_tree_; }
  const ElementTree<ptScalar>& get_element_tree() const { return element_tree_; }
  const Eigen::Matrix<ptScalar, Eigen::Dynamic, Eigen::Dynamic>& get_points() const { return points_; }
  const PatchVector<ptScalar>& get_geometry() const {
    return element_tree_.get_geometry();
  }
  int get_max_level() const { return element_tree_.get_max_level(); }
  int get_number_of_elements() const {
    return element_tree_.get_number_of_elements();
  }
  //////////////////////////////////////////////////////////////////////////////
  /// member functions
  //////////////////////////////////////////////////////////////////////////////
  void checkOrientation() {
    std::vector<std::array<int, 4>> edges = element_tree_.patchTopologyInfo();
    std::vector<Eigen::Matrix<ptScalar, 2, 1>> edge_midpoints = {
        Eigen::Matrix<ptScalar, 2, 1>(Constants::edgemps<ptScalar>[0][0], Constants::edgemps<ptScalar>[1][0]),  //
        Eigen::Matrix<ptScalar, 2, 1>(Constants::edgemps<ptScalar>[0][1], Constants::edgemps<ptScalar>[1][1]),  //
        Eigen::Matrix<ptScalar, 2, 1>(Constants::edgemps<ptScalar>[0][2], Constants::edgemps<ptScalar>[1][2]),  //
        Eigen::Matrix<ptScalar, 2, 1>(Constants::edgemps<ptScalar>[0][3], Constants::edgemps<ptScalar>[1][3])   //
    };
    const PatchVector<ptScalar>& geo = element_tree_.get_geometry();
    for (auto edge : edges) {
      if (edge[2] > 0 && edge[3] > 0) {
        Eigen::Matrix<ptScalar, 3, 1> a = geo[edge[0]].eval(edge_midpoints[edge[2]]);
        Eigen::Matrix<ptScalar, 3, 1> b = geo[edge[1]].eval(edge_midpoints[edge[3]]);
        Eigen::Matrix<ptScalar, 3, 1> na = geo[edge[0]].evalNormal(edge_midpoints[edge[2]]);
        Eigen::Matrix<ptScalar, 3, 1> nb = geo[edge[1]].evalNormal(edge_midpoints[edge[3]]);
        assert((a - b).norm() < Constants::pt_comp_tolerance<ptScalar> &&
               "These points should coincide according to the element tree");
        assert(a.dot(b) > 0 &&
               "Normals across patches are oriented the same way");
      }
    }
    return;
  }
  //////////////////////////////////////////////////////////////////////////////
  /// private members
  //////////////////////////////////////////////////////////////////////////////
 private:
  ElementTree<ptScalar> element_tree_;
  Eigen::Matrix<ptScalar, Eigen::Dynamic, Eigen::Dynamic> points_;
  //////////////////////////////////////////////////////////////////////////////
};
}  // namespace Bembel
#endif
