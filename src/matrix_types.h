#ifndef TYPES_H
#define TYPES_H

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
template <typename T>
struct IsMatrix : std::false_type {};

template <typename T>
struct IsMatrix<Matrix<T>> : std::true_type {};

template <typename T>
struct IsMatrix<MatrixView<T>> : std::true_type {};

template <typename T>
struct IsMatrix<ConstMatrixView<T>> : std::true_type {};

template <typename T>
constexpr bool kIsMatrixV = IsMatrix<T>::value;

template <typename T>
concept MatrixType = kIsMatrixV<T>;
}  // namespace linalg::types::details

#endif
