#include <Bembel/IO>

class FakeDivergenceConformingOperator;
template <>
struct Bembel::LinearOperatorTraits<FakeDivergenceConformingOperator> {
  typedef Eigen::VectorXd EigenType;
  typedef Eigen::VectorXd::Scalar Scalar;
  enum { OperatorOrder = 0, Form = DifferentialForm::DivConforming };
};

int main() {
  using namespace Bembel;
  Geometry<double> geo("sphere.dat");

  // The refinement level for the visualization is independent of that of the
  // simulation since one might consider to visualize a coarse discretisation on
  // a smooth geometry.
  const int refinement_level = 4;

  // The VTKwriter sets up initial geomety information.
  VTKSurfaceExport<double> writer(geo, refinement_level);

  // Now we can add user defined data. There are different options. Since you
  // might consider to visualize a solution of a computation, we will set up a
  // "fake" ansatz-space and showcase that.
  AnsatzSpace<FakeDivergenceConformingOperator, double> aspace(geo, 2, 1, 1);
  Eigen::VectorXd coefficients(aspace.get_number_of_dofs());
  for (int j = 0; j < aspace.get_number_of_dofs(); ++j) {
    coefficients(j) = (j % 10) * 0.1;
  }
  FunctionEvaluator<FakeDivergenceConformingOperator, double> evaluator(aspace);
  evaluator.set_function(coefficients);

  // One can use either one of the following formats:
  // std::function<double(int, Eigen::Vector2d)>& fun)
  // std::function<Eigen::Vector3d(int, Eigen::Vector2d)>
  // std::function<double(Eigen::Vector3d)>
  // std::function<Eigen::Vector3d(Eigen::Vector3d)>
  std::function<Eigen::Vector3d(int, const Eigen::Vector2d&)> fun1 =
      [&](int patch_number, const Eigen::Vector2d &reference_domain_point) {
        return evaluator.evaluateOnPatch(patch_number, reference_domain_point);
      };

  std::function<double(const Eigen::Vector3d&)> fun2 =
      [](const Eigen::Vector3d& point_in_space) { return point_in_space(0); };

  // With the help of these functions, the data sets are generated by the
  // VTKWriter
  writer.addDataSet("Vector_Field", fun1);
  writer.addDataSet("X-Value", fun2);

  // Finally, we print to file.
  writer.writeToFile("example.vtp");
  return 0;
}
