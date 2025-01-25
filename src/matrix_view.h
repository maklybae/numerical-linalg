#ifndef MATRIX_VIEW_H
#define MATRIX_VIEW_H

#include <cassert>
#include <cstddef>
#include <type_traits>
#include <utility>

#include "matrix.h"
#include "matrix_iterator.h"
#include "submatrix_range.h"
#include "types.h"

namespace linalg {
template <typename Scalar, bool IsConst>
class BaseMatrixView {
  using SubmatrixRange  = types::SubmatrixRange;
  using Matrix          = std::conditional_t<IsConst, const Matrix<Scalar>, Matrix<Scalar>>;
  using StorageIterator = std::conditional_t<IsConst, typename Matrix::const_iterator, typename Matrix::iterator>;
  using Size            = types::Size;
  using Difference      = types::Difference;
  using ReturnType      = std::conditional_t<IsConst, Scalar, Scalar&>;

  // NOLINTBEGIN(readability-identifier-naming)
  using iterator =
      std::conditional_t<IsConst, iterators::ConstMatrixBlockIterator<Scalar>, iterators::MatrixBlockIterator<Scalar>>;
  using const_iterator = iterators::ConstMatrixBlockIterator<Scalar>;
  using reverse_iterator =
      std::conditional_t<IsConst, std::reverse_iterator<iterator>, std::reverse_iterator<const_iterator>>;
  using const_reverse_iterator = std::reverse_iterator<const_iterator>;
  // NOLINTEND(readability-identifier-naming)

 public:
  BaseMatrixView() = default;

  BaseMatrixView(const BaseMatrixView&) = delete;

  BaseMatrixView(BaseMatrixView&& other) noexcept
      : begin_{std::exchange(other, nullptr)}
      , range_{std::exchange(other, {})}
      , matrix_rows_{std::exchange(other.matrix_rows_, 0)}
      , matrix_cols_{std::exchange(other.matrix_cols_, 0)} {}

  // Assignment is ambiguous
  BaseMatrixView& operator=(const BaseMatrixView&)     = delete;
  BaseMatrixView& operator=(BaseMatrixView&&) noexcept = delete;

  // NOLINTNEXTLINE(google-explicit-constructor)
  BaseMatrixView(Matrix& matrix)
      : begin_{matrix.begin()}
      , range_{SubmatrixRange::FullMatrix(matrix.Rows(), matrix.Cols())}
      , matrix_rows_{matrix.Rows()}
      , matrix_cols_{matrix.Cols()} {
    assert(!matrix.empty() && "Viewed Matrix should not be empty");
  }

  BaseMatrixView(Matrix& matrix, SubmatrixRange range)
      : begin_{matrix.begin()}, range_{range}, matrix_rows_{matrix.Rows()}, matrix_cols_{matrix.Cols()} {
    assert(!matrix.empty() && "Viewed Matrix should not be empty");
    assert(range.RowBegin() < matrix_rows_ && "Row begin index out of bounds");
    assert(range.RowEnd() <= matrix_rows_ && "Row end index out of bounds");
    assert(range.ColBegin() < matrix_cols_ && "Col begin index out of bounds");
    assert(range.ColEnd() <= matrix_cols_ && "Col end index out of bounds");
  }

  BaseMatrixView(Matrix&&) = delete;

  ReturnType operator()(Size row, Size col) const {
    assert(begin_ != nullptr && "Matrix view should not be empty");
    assert(row < Rows() && "Row index out of bounds");
    assert(col < Cols() && "Col index out of bounds");

    return begin_ + (range_.RowBegin() + row, range_.ColBegin() + col);
  }

  Size Rows() const {
    return range_.Rows();
  }

  Size Cols() const {
    return range_.Cols();
  }

  // NOLINTBEGIN(readability-identifier-naming)
  iterator begin() const {
    return iterator{begin_ + static_cast<Difference>(range_.RowBegin() * range_.Cols() + range_.ColBegin()),
                    range_.Cols(), matrix_cols_ - range_.Cols()};
  }

  const_iterator cbegin() const {
    return const_iterator{begin_ + static_cast<Difference>(range_.RowBegin() * range_.Cols() + range_.ColBegin()),
                          range_.Cols(), matrix_cols_ - range_.Cols()};
  }

  iterator end() const {
    return iterator{begin_ + static_cast<Difference>(range_.RowEnd() * range_.Cols() + range_.ColBegin()),
                    range_.Cols(), matrix_cols_ - range_.Cols()};
  }

  const_iterator cend() const {
    return const_iterator{begin_ + static_cast<Difference>(range_.RowEnd() * range_.Cols() + range_.ColBegin()),
                          range_.Cols(), matrix_cols_ - range_.Cols()};
  }

  reverse_iterator rbegin() const {
    return reverse_iterator{end()};
  }

  const_reverse_iterator crbegin() const {
    return const_reverse_iterator{cend()};
  }

  reverse_iterator rend() const {
    return reverse_iterator{begin()};
  }

  const_reverse_iterator crend() const {
    return const_reverse_iterator{cbegin()};
  }
  // NOLINTEND(readability-identifier-naming)

 private:
  StorageIterator begin_{};
  SubmatrixRange range_{};
  Size matrix_rows_{};
  Size matrix_cols_{};
};

template <typename Scalar>
using MatrixView = BaseMatrixView<Scalar, false>;

template <typename Scalar>
using ConstMatrixView = BaseMatrixView<Scalar, true>;

}  // namespace linalg

#endif
