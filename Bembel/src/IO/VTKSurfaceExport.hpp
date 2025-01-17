// This file is part of Bembel, the higher order C++ boundary element library.
// It was written as part of a cooperation of J. Doelz, H. Harbrecht, S. Kurz,
// M. Multerer, S. Schoeps, and F. Wolf at Technische Universitaet Darmstadt,
// Universitaet Basel, and Universita della Svizzera italiana, Lugano. This
// source code is subject to the GNU General Public License version 3 and
// provided WITHOUT ANY WARRANTY, see <http://www.bembel.eu> for further
// information.

#ifndef BEMBEL_IO_VTKSURFACEEXPORT_H_
#define BEMBEL_IO_VTKSURFACEEXPORT_H_

namespace Bembel {

// This class provides the possibilty to generate a vtk-visualization.
template <typename ptScalar>
class VTKSurfaceExport {
 public:
  /**
  * \ingroup IO
  * \brief Provides export routines to the VTK file format.
  *
  * The constructor wants a geometetry and a refinement level. This choice is
  * deliberately not a mesh, since the visualization will often be on a finer
  * mesh then that of a computation.
  **/
  VTKSurfaceExport(const Geometry<ptScalar>& geo, int M) { init_VTKSurfaceExport(geo, M); }
  inline void init_VTKSurfaceExport(const Geometry<ptScalar>& geo, int M) {
    msh.init_ClusterTree(geo, M);
    points = msh.get_element_tree().generatePointList().transpose();
    cells = msh.get_element_tree().generateElementList().transpose();
    normals = Eigen::Matrix<ptScalar, Eigen::Dynamic, Eigen::Dynamic>(cells.rows(), 3);
    patch_number = Eigen::VectorXi(cells.rows());
    
    for (auto e = msh.get_element_tree().cpbegin();
         e != msh.get_element_tree().cpend(); ++e) {
      normals.row(e->id_) = msh.get_geometry()[e->patch_]
                                .evalNormal(e->referenceMidpoint())
                                .normalized()
                                .transpose();
    }
    for (auto e = msh.get_element_tree().cpbegin();
         e != msh.get_element_tree().cpend(); ++e) {
      patch_number(e->id_) = e->patch_;
    }
    return;
  }
  // One can add data to visualize via the addDataSet methods. They accept a
  // std::function object of different types, and generate the data needed for
  // the vtk file. Allowed formats are:
  // std::function<ptScalar(int, Eigen::Matrix<ptScalar, 2, 1>)>& fun)
  // std::function<Eigen::Matrix<ptScalar, 3, 1>(int, Eigen::Matrix<ptScalar, 2, 1>)>
  // std::function<ptScalar(Eigen::Matrix<ptScalar, 3, 1>)>
  // std::function<Eigen::Matrix<ptScalar, 3, 1>(Eigen::Matrix<ptScalar, 3, 1>)>
  inline void addDataSet(const std::string& name,
                         std::function<ptScalar(int, const Eigen::Matrix<ptScalar, 2, 1>&)> fun) {
    Eigen::Matrix<ptScalar, Eigen::Dynamic, Eigen::Dynamic> data(cells.rows(), 1);
    for (auto e = msh.get_element_tree().cpbegin();
         e != msh.get_element_tree().cpend(); ++e) {
      data(e->id_) = fun(e->patch_, e->referenceMidpoint());
    }
    addDataSet_(name, data);
    return;
  }
  inline void addDataSet(
      const std::string& name,
      std::function<Eigen::Matrix<ptScalar, 3, 1>(int, const Eigen::Matrix<ptScalar, 2, 1>&)> fun) {
    Eigen::Matrix<ptScalar, Eigen::Dynamic, Eigen::Dynamic> data(cells.rows(), 3);
    int k = 0;
    for (auto e = msh.get_element_tree().cpbegin();
         e != msh.get_element_tree().cpend(); ++e) {
      data.row(e->id_) = fun(e->patch_, e->referenceMidpoint()).transpose();
    }
    addDataSet_(name, data);
    return;
  }
  inline void addDataSet(const std::string& name,
                         std::function<Eigen::Matrix<ptScalar, 3, 1>(const Eigen::Matrix<ptScalar, 3, 1>&)> fun) {
    Eigen::Matrix<ptScalar, Eigen::Dynamic, Eigen::Dynamic> data(cells.rows(), 3);
    for (auto e = msh.get_element_tree().cpbegin();
         e != msh.get_element_tree().cpend(); ++e) {
      data.row(e->id_) =
          fun(msh.get_geometry()[e->patch_].eval(e->referenceMidpoint()))
              .transpose();
    }
    addDataSet_(name, data);
    return;
  }
  inline void addDataSet(const std::string& name,
                         std::function<ptScalar(const Eigen::Matrix<ptScalar, 3, 1>&)> fun) {
    Eigen::Matrix<ptScalar, Eigen::Dynamic, Eigen::Dynamic> data(cells.rows(), 1);
    for (auto e = msh.get_element_tree().cpbegin();
         e != msh.get_element_tree().cpend(); ++e) {
      data(e->id_) =
          fun(msh.get_geometry()[e->patch_].eval(e->referenceMidpoint()));
    }
    addDataSet_(name, data);
    return;
  }

  inline void writeToFile(const std::string& filename) {
    std::ofstream output;
    output.open(filename);
    output << "<VTKFile type=\"PolyData\" version=\"0.1\" "
              "byte_order=\"LittleEndian\">\n"
              "<PolyData>\n"
              "<Piece NumberOfPoints=\" "
           << points.rows()
           << "\" NumberOfVerts=\"0\" NumberOfLines=\"0\" NumberOfStrips=\"0\" "
              "NumberOfPolys=\""
           << cells.rows()
           << "\">\n"
              "<Points>\n"
              "<DataArray type=\"Float32\" NumberOfComponents=\"3\" "
              "format=\"ascii\">\n";
    output << points;
    output << "</DataArray>\n</Points>";
    output << "<CellData Scalars=\"patch_number\" Normals=\"cell_normals\">\n"
              "<DataArray type=\"Int32\" Name=\"patch_number\" "
              "format=\"ascii\">\n";
    output << patch_number;
    output << "</DataArray>\n<DataArray type=\"Float32\" Name=\"cell_normals\" "
              "NumberOfComponents=\"3\" format=\"ascii\">";
    output << normals;
    output << "</DataArray>\n";
    for (auto data : additionalData) {
      output << data;
    }
    output << "</CellData>\n"
              "<Polys>\n"
              "<DataArray type=\"Int32\" Name=\"connectivity\" "
              "format=\"ascii\">\n";
    output << cells;
    output << "</DataArray>\n"
              "<DataArray type=\"Int32\" Name=\"offsets\" "
              "format=\"ascii\">\n";
    for (int i = 0; i < cells.rows(); ++i) {
      output << (i + 1) * 4;
      output << "\n";
    }
    output << "</DataArray>\n"
              "</Polys>\n"
              "</Piece>\n"
              "</PolyData>\n"
              "</VTKFile>";
    output.close();
    return;
  }
  // can be used to clear the additional Data.
  inline void clearData() {
    additionalData = {};
    additionalData.shrink_to_fit();
  }

 private:
  // This routine turns the data of the above DataSet-routines into a string and
  // stores it.
  inline void addDataSet_(const std::string& name, const Eigen::Matrix<ptScalar, Eigen::Dynamic, Eigen::Dynamic>& mat) {
    assert(mat.cols() == 1 || mat.cols() == 3);
    assert(mat.rows() == cells.rows());

    const int cols = mat.cols();
    std::string data_ascii = "<DataArray type=\"Float32\" Name=\"" + name +
                             "\" NumberOfComponents=\"" +
                             std::to_string(mat.cols()) +
                             "\" format=\"ascii\">\n";
    for (int i = 0; i < mat.rows(); ++i) {
      for (int j = 0; j < cols; ++j) {
        data_ascii.append(std::to_string(mat(i, j)) + " ");
      }
      data_ascii.append("\n");
    }
    data_ascii.append("</DataArray>\n");
    additionalData.push_back(data_ascii);
  }

  ClusterTree<ptScalar> msh;
  Eigen::Matrix<ptScalar, Eigen::Dynamic, Eigen::Dynamic> points;
  Eigen::MatrixXi cells;
  Eigen::Matrix<ptScalar, Eigen::Dynamic, Eigen::Dynamic> normals;
  Eigen::VectorXi patch_number;
  std::vector<std::string> additionalData;
};

}  // namespace Bembel

#endif
