#ifndef MATRIX_VIEW_H
#define MATRIX_VIEW_H

#include <cassert>
#include <cstddef>
#include <type_traits>
#include <utility>

#include "iterator_helper.h"
#include "matrix.h"
#include "submatrix_range.h"
#include "types.h"

namespace linalg {
template <typename Scalar, bool IsConst>
class BaseMatrixView {
 public:
  using SubmatrixRange = types::SubmatrixRange;
  using Matrix         = std::conditional_t<IsConst, const Matrix<Scalar>, Matrix<Scalar>>;
  using Size           = types::Size;
  using Difference     = types::Difference;
  using ReturnType     = std::conditional_t<IsConst, Scalar, Scalar&>;

  // NOLINTBEGIN(readability-identifier-naming)
  using iterator =
      std::conditional_t<IsConst, iterators::ConstMatrixBlockIterator<Scalar>, iterators::MatrixBlockIterator<Scalar>>;
  using const_iterator = iterators::ConstMatrixBlockIterator<Scalar>;
  using reverse_iterator =
      std::conditional_t<IsConst, std::reverse_iterator<iterator>, std::reverse_iterator<const_iterator>>;
  using const_reverse_iterator = std::reverse_iterator<const_iterator>;
  // NOLINTEND(readability-identifier-naming)

  BaseMatrixView() = default;

  BaseMatrixView(const BaseMatrixView&) = delete;

  BaseMatrixView(BaseMatrixView&& other) noexcept
      : ptr_{std::exchange(other, nullptr)}, range_{std::exchange(other, {})} {}

  // Assignment is ambiguous
  BaseMatrixView& operator=(const BaseMatrixView&)     = delete;
  BaseMatrixView& operator=(BaseMatrixView&&) noexcept = delete;

  // NOLINTNEXTLINE(google-explicit-constructor)
  BaseMatrixView(Matrix& matrix) : ptr_{&matrix}, range_{SubmatrixRange::FullMatrix(matrix.Rows(), matrix.Cols())} {
    // assert(!matrix.empty() && "Viewed Matrix should not be empty");
  }

  BaseMatrixView(Matrix& matrix, SubmatrixRange range) : ptr_{&matrix}, range_{range} {
    // assert(!matrix.empty() && "Viewed Matrix should not be empty");
    assert(range.RowBegin() < ptr_->Rows() && "Row begin index out of bounds");
    assert(range.RowEnd() <= ptr_->Rows() && "Row end index out of bounds");
    assert(range.ColBegin() < ptr_->Cols() && "Col begin index out of bounds");
    assert(range.ColEnd() <= ptr_->Cols() && "Col end index out of bounds");
  }

  BaseMatrixView(Matrix&&) = delete;

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

  // NOLINTBEGIN(readability-identifier-naming)
  iterator begin() const {
    return iterator{ptr_->begin() + static_cast<Difference>(range_.RowBegin() * ptr_->Cols() + range_.ColBegin()),
                    range_.Cols(), ptr_->Cols() - range_.Cols()};
  }

  const_iterator cbegin() const {
    return const_iterator{ptr_->begin() + static_cast<Difference>(range_.RowBegin() * ptr_->Cols() + range_.ColBegin()),
                          range_.Cols(), ptr_->Cols() - range_.Cols()};
  }

  iterator end() const {
    return iterator{
        ptr_->begin() + static_cast<Difference>((range_.RowBegin() + range_.Rows()) * ptr_->Cols() + range_.ColBegin()),
        range_.Cols(), ptr_->Cols() - range_.Cols()};
  }

  const_iterator cend() const {
    return const_iterator{
        ptr_->begin() + static_cast<Difference>((range_.RowBegin() + range_.Rows()) * ptr_->Cols() + range_.ColBegin()),
        range_.Cols(), ptr_->Cols() - range_.Cols()};
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
  Matrix* ptr_{};
  SubmatrixRange range_{};
};

template <typename Scalar>
using MatrixView = BaseMatrixView<Scalar, false>;

template <typename Scalar>
using ConstMatrixView = BaseMatrixView<Scalar, true>;

}  // namespace linalg

#endif
