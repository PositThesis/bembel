// This file is part of Bembel, the higher order C++ boundary element library.
// It was written as part of a cooperation of J. Doelz, H. Harbrecht, S. Kurz,
// M. Multerer, S. Schoeps, and F. Wolf at Technische Universitaet Darmstadt,
// Universitaet Basel, and Universita della Svizzera italiana, Lugano. This
// source code is subject to the GNU General Public License version 3 and
// provided WITHOUT ANY WARRANTY, see <http://www.bembel.eu> for further
// information.

#ifndef BEMBEL_IO_VTKDOMAINEXPORT_H_
#define BEMBEL_IO_VTKDOMAINEXPORT_H_

namespace Bembel {

// This class provides the possibilty to generate a vtk-visualization.
template <typename ptScalar>
class VTKDomainExport {
 public:
    /**
  * \ingroup IO
  * \brief Provides export routines to the VTK file format.
  **/
  VTKDomainExport(const Eigen::Matrix<ptScalar, Eigen::Dynamic, 1> &x_vec, const Eigen::Matrix<ptScalar, Eigen::Dynamic, 1> &y_vec,
                  const Eigen::Matrix<ptScalar, Eigen::Dynamic, 1> &z_vec) {
    init_VTKDomainExport(x_vec, y_vec, z_vec);
  }
  inline void init_VTKDomainExport(const Eigen::Matrix<ptScalar, Eigen::Dynamic, 1> &x_vec,
                                   const Eigen::Matrix<ptScalar, Eigen::Dynamic, 1> &y_vec,
                                   const Eigen::Matrix<ptScalar, Eigen::Dynamic, 1> &z_vec) {
    x_vec_ = x_vec;
    y_vec_ = y_vec;
    z_vec_ = z_vec;
    x_num = x_vec_.rows() - 1;
    y_num = y_vec_.rows() - 1;
    z_num = z_vec_.rows() - 1;
    max_size_ = x_vec_.rows() * y_vec_.rows() * z_vec_.rows();
    return;
  }
  // One can add data to visualize via the addDataSet methods. They accept a
  // std::function object of different types, and generate the data needed for
  // the vtk file. Allowed formats are:
  // std::function<ptScalar(Eigen::Matrix<ptScalar, 3, 1>)>
  // std::function<Eigen::Matrix<ptScalar, 3, 1>(Eigen::Matrix<ptScalar, 3, 1>)>
  inline void addDataSet(const std::string &name,
                         std::function<ptScalar(const Eigen::Matrix<ptScalar, 3, 1> &)> fun) {
    Eigen::Matrix<ptScalar, Eigen::Dynamic, Eigen::Dynamic> data(max_size_, 1);
    for (int z_idx = 0; z_idx < z_vec_.rows(); ++z_idx)
      for (int y_idx = 0; y_idx < y_vec_.rows(); ++y_idx)
        for (int x_idx = 0; x_idx < x_vec_.rows(); ++x_idx) {
          const unsigned long i = (x_vec_.rows() * y_vec_.rows() * z_idx) +
                                  (x_vec_.rows() * y_idx) + x_idx;
          data(i) =
              fun(Eigen::Matrix<ptScalar, 3, 1>(x_vec_(x_idx), y_vec_(y_idx), z_vec_(z_idx)));
        }
    addDataSet_(name, data);
    return;
  }
  inline void addDataSet(
      const std::string &name,
      std::function<Eigen::Matrix<ptScalar, 3, 1>(const Eigen::Matrix<ptScalar, 3, 1> &)> fun) {
    Eigen::Matrix<ptScalar, Eigen::Dynamic, Eigen::Dynamic> data(max_size_, 3);
    for (int z_idx = 0; z_idx < z_vec_.rows(); ++z_idx)
      for (int y_idx = 0; y_idx < y_vec_.rows(); ++y_idx)
        for (int x_idx = 0; x_idx < x_vec_.rows(); ++x_idx) {
          const unsigned long i = (x_vec_.rows() * y_vec_.rows() * z_idx) +
                                  (x_vec_.rows() * y_idx) + x_idx;
          data.row(i) =
              fun(Eigen::Matrix<ptScalar, 3, 1>(x_vec_(x_idx), y_vec_(y_idx), z_vec_(z_idx)))
                  .transpose();
        }
    addDataSet_(name, data);
    return;
  }
  inline void writeToFile(const std::string &filename) {
    std::ofstream output;
    output.open(filename);
    output << "<VTKFile type=\"StructuredGrid\" version=\"0.1\" "
              "byte_order=\"LittleEndian\">\n"
              "<StructuredGrid WholeExtent=\""
           << "0 " << x_num << " 0 " << y_num << " 0 " << z_num
           << "\">\n"
              "<Piece Extent= \""
           << "0 " << x_num << " 0 " << y_num << " 0 " << z_num
           << "\">\n"
              "<Points>\n"
              "<DataArray type=\"Float32\" NumberOfComponents=\"3\" "
              "format=\"ascii\">\n";

    for (int z_idx = 0; z_idx < z_vec_.rows(); ++z_idx)
      for (int y_idx = 0; y_idx < y_vec_.rows(); ++y_idx)
        for (int x_idx = 0; x_idx < x_vec_.rows(); ++x_idx)
          output << x_vec_(x_idx) << " " << y_vec_(y_idx) << " "
                 << z_vec_(z_idx) << "\n";
    output << "</DataArray>\n"
              "</Points>\n"
              "<PointData>\n";
    for (auto d : additionalData) output << d;
    output << "</PointData>\n"
              "</Piece>\n"
              "</StructuredGrid>\n"
              "</VTKFile>\n";
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
  inline void addDataSet_(const std::string &name, const Eigen::Matrix<ptScalar, Eigen::Dynamic, Eigen::Dynamic> &mat) {
    assert(mat.cols() == 1 || mat.cols() == 3);

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
  int max_size_;
  int x_num;
  int y_num;
  int z_num;
  Eigen::Matrix<ptScalar, Eigen::Dynamic, 1> x_vec_;
  Eigen::Matrix<ptScalar, Eigen::Dynamic, 1> y_vec_;
  Eigen::Matrix<ptScalar, Eigen::Dynamic, 1> z_vec_;
  std::vector<std::string> additionalData;
};

}  // namespace Bembel

#endif
