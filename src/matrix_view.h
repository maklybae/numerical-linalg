#ifndef MATRIX_VIEW_H
#define MATRIX_VIEW_H

#include <cassert>
#include <cstddef>
#include <type_traits>
#include <utility>

#include "matrix.h"
#include "submatrix_range.h"

namespace linalg {
template <typename Scalar, bool IsConst>
class BaseMatrixView {
  using SubmatrixRange = types::SubmatrixRange;
  using Matrix         = std::conditional_t<IsConst, const Matrix<Scalar>, Matrix<Scalar>>;
  using Size           = types::Size;
  using ReturnType     = std::conditional_t<IsConst, Scalar, Scalar&>;

 public:
  BaseMatrixView() = default;

  BaseMatrixView(const BaseMatrixView&) = default;

  BaseMatrixView(BaseMatrixView&& other) noexcept
      : ptr_{std::exchange(other, nullptr)}, range_{std::exchange(other, {})} {}

  BaseMatrixView& operator=(const BaseMatrixView&) = default;

  BaseMatrixView& operator=(BaseMatrixView&& other) noexcept {
    BaseMatrixView tmp = std::move(other);
    std::swap(ptr_, tmp.ptr_);
    std::swap(range_, tmp.range_);

    assert(other.ptr_ == nullptr && "Pointer of moved matrix view should be equal nullptr");
    return *this;
  }

  BaseMatrixView(Matrix* ptr, SubmatrixRange range) : ptr_{ptr}, range_{range} {}

  // NOLINTNEXTLINE(google-explicit-constructor)
  BaseMatrixView(Matrix& matrix) : ptr_{&matrix}, range_{SubmatrixRange::FullMatrix(matrix.Rows(), matrix.Cols())} {
    assert(!matrix.empty() && "Viewed Matrix should not be empty");
  }

  ReturnType operator()(Size row, Size col) const {
    assert(ptr_ != nullptr && "Matrix view should not be empty");
    assert(row < Rows() && "Row index out of bounds");
    assert(col < Cols() && "Col index out of bounds");

    return (*ptr_)(range_.RowBegin() + row, range_.ColBegin() + col);
  }

  Size Rows() const {
    return range_.Rows();
  }

  Size Cols() const {
    return range_.Cols();
  }

 private:
  Matrix* ptr_{nullptr};
  SubmatrixRange range_{};
};

template <typename Scalar>
using MatrixView = BaseMatrixView<Scalar, false>;

template <typename Scalar>
using ConstMatrixView = BaseMatrixView<Scalar, true>;

}  // namespace linalg

#endif
