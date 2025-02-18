#ifndef TYPES_H
#define TYPES_H

#include <type_traits>

#include "types_details.h"

namespace linalg::view {
template <typename Scalar, types::ConstnessEnum>
class BaseMatrixView;
}

namespace linalg {
template <types::FloatingOrComplexType Scalar>
class Matrix;

template <typename Scalar>
using MatrixView = view::BaseMatrixView<Scalar, types::ConstnessEnum::kNonConst>;

template <typename Scalar>
using ConstMatrixView = view::BaseMatrixView<Scalar, types::ConstnessEnum::kConst>;
}  // namespace linalg

namespace linalg::types::details {
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
}  // namespace linalg::types::details

namespace linalg::types {
template <typename Scalar>
concept MatrixType = details::kIsMatrixV<Scalar>;

template <typename Scalar>
concept MutableMatrixType = details::kIsMutableMatrixV<Scalar>;
}  // namespace linalg::types

#endif
