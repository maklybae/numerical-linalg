#ifndef MATRIX_VIEW_H
#define MATRIX_VIEW_H

#include "matrix_types.h"

namespace linalg {

using detail::ConstMatrixView;
using detail::MatrixView;

template <typename Scalar>
MatrixView<Scalar> Transposed(const MatrixView<Scalar>& matrix) {
  auto view = MatrixView<Scalar>{matrix};
  view.Transpose();
  return view;
}

template <typename Scalar>
ConstMatrixView<Scalar> Transposed(const ConstMatrixView<Scalar>& matrix) {
  auto view = ConstMatrixView<Scalar>{matrix};
  view.Transpose();
  return view;
}

template <std::floating_point FPScalar>
MatrixView<FPScalar> Conjugated(const MatrixView<FPScalar>& matrix) {
  return Transposed(matrix);
}

template <std::floating_point FPScalar>
ConstMatrixView<FPScalar> Conjugated(const ConstMatrixView<FPScalar>& matrix) {
  return Transposed(matrix);
}

}  // namespace linalg

#endif
