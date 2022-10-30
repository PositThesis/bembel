// This file is part of Bembel, the higher order C++ boundary element library.
// It was written as part of a cooperation of J. Doelz, H. Harbrecht, S. Kurz,
// M. Multerer, S. Schoeps, and F. Wolf at Technische Universitaet Darmstadt,
// Universitaet Basel, and Universita della Svizzera italiana, Lugano. This
// source code is subject to the GNU General Public License version 3 and
// provided WITHOUT ANY WARRANTY, see <http://www.bembel.eu> for further
// information.
//
#ifndef BEMBEL_SPLINE_SHAPEFUNCTIONS_H_
#define BEMBEL_SPLINE_SHAPEFUNCTIONS_H_

namespace Bembel {
namespace Basis {

template <typename ptScalar>
using funptr_ptScalarOut_ptScalarptrptScalarIn = ptScalar (*)(ptScalar*, ptScalar);
template <typename ptScalar>
using funptr_voidOut_ptScalarptrptScalarIn = void (*)(ptScalar*, ptScalar);

/**
 *  \ingroup Spline
 *  \brief These routines implement a template recursion that allows to choose a
 *compile time instantiation of a basis-evaluation routine with a runtime p. To
 *replace the underlying basis, only these routines should be changed.
 **/
template <int P, typename ptScalar>
class PSpecificShapeFunctionHandler {
 public:
  inline static ptScalar evalCoef(int p, ptScalar* ar, ptScalar x) {
    return p == P ? Bembel::Basis::EvalBernstein<ptScalar, P>(ar, x)
                  : PSpecificShapeFunctionHandler<P - 1, ptScalar>::evalCoef(p, ar, x);
  }
  inline static ptScalar evalDerCoef(int p, ptScalar* ar, ptScalar x) {
    return p == P ? Bembel::Basis::EvalBernsteinDer<ptScalar, P>(ar, x)
                  : PSpecificShapeFunctionHandler<P - 1, ptScalar>::evalDerCoef(p, ar, x);
  }
  inline static void evalBasis(int p, ptScalar* ar, ptScalar x) {
    return p == P ? Bembel::Basis::EvalBernsteinBasis<ptScalar, P>(ar, x)
                  : PSpecificShapeFunctionHandler<P - 1, ptScalar>::evalBasis(p, ar, x);
  }
  inline static void evalDerBasis(int p, ptScalar* ar, ptScalar x) {
    return p == P
               ? Bembel::Basis::EvalBernsteinDerBasis<ptScalar, P>(ar, x)
               : PSpecificShapeFunctionHandler<P - 1, ptScalar>::evalDerBasis(p, ar, x);
  }
  inline static constexpr funptr_ptScalarOut_ptScalarptrptScalarIn<ptScalar> ptrEvalCoef(
      int p) {
    return p == P ? &Bembel::Basis::EvalBernstein<ptScalar, P>
                  : PSpecificShapeFunctionHandler<P - 1, ptScalar>::ptrEvalCoef(p);
  }
  inline static constexpr funptr_ptScalarOut_ptScalarptrptScalarIn<ptScalar> ptrEvalDerCoef(
      int p) {
    return p == P ? &Bembel::Basis::EvalBernsteinDer<ptScalar, P>
                  : PSpecificShapeFunctionHandler<P - 1, ptScalar>::ptrEvalDerCoef(p);
  }
  inline static constexpr funptr_voidOut_ptScalarptrptScalarIn<ptScalar> ptrEvalBasis(int p) {
    return p == P ? &Bembel::Basis::EvalBernsteinBasis<ptScalar, P>
                  : PSpecificShapeFunctionHandler<P - 1, ptScalar>::ptrEvalBasis(p);
  }
  inline static constexpr funptr_voidOut_ptScalarptrptScalarIn<ptScalar> ptrEvalDerBasis(
      int p) {
    return p == P ? &Bembel::Basis::EvalBernsteinDerBasis<ptScalar, P>
                  : PSpecificShapeFunctionHandler<P - 1, ptScalar>::ptrEvalDerBasis(p);
  }
  inline static constexpr bool checkP(int p) {
    static_assert(P > 0, "Polynomial degree must be larger than zero");
    return p <= Constants::MaxP;
  }
};

template <typename ptScalar>
class PSpecificShapeFunctionHandler<0, ptScalar> {
 public:
  inline static ptScalar evalCoef(int p, ptScalar* ar, ptScalar x) {
    return Bembel::Basis::EvalBernstein<ptScalar, 0>(ar, x);
  }
  inline static ptScalar evalDerCoef(int p, ptScalar* ar, ptScalar x) {
    return Bembel::Basis::EvalBernsteinDer<ptScalar, 0>(ar, x);
  }
  inline static void evalBasis(int p, ptScalar* ar, ptScalar x) {
    return Bembel::Basis::EvalBernsteinBasis<ptScalar, 0>(ar, x);
  }
  inline static void evalDerBasis(int p, ptScalar* ar, ptScalar x) {
    return Bembel::Basis::EvalBernsteinDerBasis<ptScalar, 0>(ar, x);
  }
  inline static constexpr funptr_ptScalarOut_ptScalarptrptScalarIn<ptScalar> ptrEvalCoef(
      int p) {
    return &Bembel::Basis::EvalBernstein<ptScalar, 0>;
  }
  inline static constexpr funptr_ptScalarOut_ptScalarptrptScalarIn<ptScalar> ptrEvalDerCoef(
      int p) {
    return &Bembel::Basis::EvalBernsteinDer<ptScalar, 0>;
  }
  inline static constexpr funptr_voidOut_ptScalarptrptScalarIn<ptScalar> ptrEvalBasis(int p) {
    return &Bembel::Basis::EvalBernsteinBasis<ptScalar, 0>;
  }
  inline static constexpr funptr_voidOut_ptScalarptrptScalarIn<ptScalar> ptrEvalDerBasis(
      int p) {
    return &Bembel::Basis::EvalBernsteinDerBasis<ptScalar, 0>;
  }
  inline static constexpr bool checkP(int p) { return Constants::MaxP >= 0; }
};

template <typename ptScalar>
using ShapeFunctionHandler = PSpecificShapeFunctionHandler<Constants::MaxP, ptScalar>;

}  // namespace Basis
}  // namespace Bembel
#endif
