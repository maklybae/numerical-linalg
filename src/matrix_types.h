#ifndef MATRIX_TYPES_H
#define MATRIX_TYPES_H

#include "core_types.h"
#include "scalar_types.h"

namespace linalg {

template <detail::FloatingOrComplexType Scalar>
class Matrix;

namespace detail {

template <typename Scalar, ConstnessEnum Constness>
class BaseMatrixView;

template <typename Scalar>
using MatrixView = BaseMatrixView<Scalar, ConstnessEnum::kNonConst>;

template <typename Scalar>
using ConstMatrixView = BaseMatrixView<Scalar, ConstnessEnum::kConst>;

template <typename Scalar>
struct IsMatrix : std::false_type {};

template <typename Scalar>
struct IsMatrix<Matrix<Scalar>> : std::true_type {};

template <typename Scalar>
struct IsMatrix<MatrixView<Scalar>> : std::true_type {};

template <typename Scalar>
struct IsMatrix<ConstMatrixView<Scalar>> : std::true_type {};

template <typename Scalar>
constexpr bool kIsMatrixV = IsMatrix<Scalar>::value;

template <typename Scalar>
struct IsMutableMatrix : std::false_type {};

template <typename Scalar>
struct IsMutableMatrix<Matrix<Scalar>> : std::true_type {};

template <typename Scalar>
struct IsMutableMatrix<MatrixView<Scalar>> : std::true_type {};

template <typename Scalar>
constexpr bool kIsMutableMatrixV = IsMutableMatrix<Scalar>::value;

template <typename Scalar>
concept MatrixType = kIsMatrixV<Scalar>;

template <typename Scalar>
concept MutableMatrixType = kIsMutableMatrixV<Scalar>;

template <MatrixType LhsT, MatrixType RhsT>
using CommonValueType = std::common_type_t<typename LhsT::value_type, typename RhsT::value_type>;

struct MatrixState {
  enum class TransposedEnum { kTransposed, kNonTransposed };
  enum class ConjugatedEnum { kConjugated, kNonConjugated };

  TransposedEnum transposed{TransposedEnum::kNonTransposed};
  ConjugatedEnum conjugated{ConjugatedEnum::kNonConjugated};

  MatrixState& SwitchTransposed() {
    transposed =
        (transposed == TransposedEnum::kTransposed) ? TransposedEnum::kNonTransposed : TransposedEnum::kTransposed;
    return *this;
  }

  MatrixState& SwitchConjugated() {
    conjugated =
        (conjugated == ConjugatedEnum::kConjugated) ? ConjugatedEnum::kNonConjugated : ConjugatedEnum::kConjugated;
    return *this;
  }

  bool IsTransposed() const {
    return transposed == TransposedEnum::kTransposed;
  }

  bool IsConjugated() const {
    return conjugated == ConjugatedEnum::kConjugated;
  }
};

}  // namespace detail
}  // namespace linalg

#endif
