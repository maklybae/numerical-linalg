#ifndef MATRIX_VIEW_H
#define MATRIX_VIEW_H

#include <cassert>
#include <cstddef>
#include <type_traits>
#include <utility>

#include "iterator_helper.h"
#include "matrix.h"
#include "matrix_types.h"
#include "submatrix_range.h"

namespace linalg::view {
using types::ConstnessEnum;

template <typename Scalar, ConstnessEnum Constness>
class BaseMatrixView {
  static constexpr bool kIsConst = Constness == ConstnessEnum::kConst;

 public:
  using BlockIterator      = iterators::BlockMovingLogic<iterators::DefaultAccessor<iterators::DefaultDefines<Scalar>>>;
  using ConstBlockIterator = iterators::BlockMovingLogic<iterators::DefaultAccessor<iterators::ConstDefines<Scalar>>>;

  using SubmatrixRange = types::SubmatrixRange;
  using MyMatrix       = std::conditional_t<kIsConst, const Matrix<Scalar>, Matrix<Scalar>>;
  using ReturnType     = std::conditional_t<kIsConst, Scalar, Scalar&>;

  // NOLINTBEGIN(readability-identifier-naming)
  using value_type      = Scalar;
  using reference       = Scalar&;
  using const_reference = const Scalar&;
  using iterator        = std::conditional_t<kIsConst, ConstBlockIterator, BlockIterator>;
  using const_iterator  = ConstBlockIterator;
  using reverse_iterator =
      std::conditional_t<kIsConst, std::reverse_iterator<iterator>, std::reverse_iterator<const_iterator>>;
  using const_reverse_iterator = std::reverse_iterator<const_iterator>;
  using difference_type        = types::Difference;
  using size_type              = types::Size;
  // NOLINTEND(readability-identifier-naming)

  // Needs to generate a ctor from non-const to const view.
  template <typename, ConstnessEnum>
  friend class BaseMatrixView;

  BaseMatrixView() = delete;

  BaseMatrixView(const BaseMatrixView&) = delete;

  BaseMatrixView(BaseMatrixView&& other) noexcept
      : ptr_{std::exchange(other, nullptr)}, range_{std::exchange(other, {})} {}

  // Assignment is ambiguous
  BaseMatrixView& operator=(const BaseMatrixView&)     = delete;
  BaseMatrixView& operator=(BaseMatrixView&&) noexcept = delete;

  // NOLINTNEXTLINE(google-explicit-constructor)
  BaseMatrixView(MyMatrix& matrix) : ptr_{&matrix}, range_{SubmatrixRange::FullMatrix(matrix.Rows(), matrix.Cols())} {
    assert(matrix.IsValidMatrix() && "Matrix data should not be empty");
  }

  // NOLINTNEXTLINE(google-explicit-constructor)
  BaseMatrixView(BaseMatrixView<Scalar, ConstnessEnum::kNonConst>& other)
    requires kIsConst
      : ptr_{other.ptr_}, range_{other.range_} {}

  // NOLINTNEXTLINE(google-explicit-constructor)
  BaseMatrixView(BaseMatrixView<Scalar, ConstnessEnum::kNonConst>&& other) noexcept
    requires kIsConst
      : ptr_{std::exchange(other.ptr_, nullptr)}, range_{std::exchange(other.range_, {})} {}

  BaseMatrixView(MyMatrix& matrix, SubmatrixRange range) : ptr_{&matrix}, range_{range} {
    assert(matrix.IsValidMatrix() && "Matrix data should not be empty");
    assert(range.RowBegin() < ptr_->Rows() && "Row begin index out of bounds");
    assert(range.RowEnd() <= ptr_->Rows() && "Row end index out of bounds");
    assert(range.ColBegin() < ptr_->Cols() && "Col begin index out of bounds");
    assert(range.ColEnd() <= ptr_->Cols() && "Col end index out of bounds");
  }

  BaseMatrixView(MyMatrix&&) = delete;

  ReturnType operator()(size_type row, size_type col) const {
    assert(IsValidMatrixView() && "Matrix view should not be empty");
    assert(row < Rows() && "Row index out of bounds");
    assert(col < Cols() && "Col index out of bounds");

    return (*ptr_)(range_.RowBegin() + row, range_.ColBegin() + col);
  }

  size_type Rows() const {
    return range_.Rows();
  }

  size_type Cols() const {
    return range_.Cols();
  }

  // NOLINTBEGIN(readability-identifier-naming)
  iterator begin() const {
    return iterator{ptr_->RawBegin() + range_.RowBegin() * ptr_->Cols() + range_.ColBegin(), range_.Cols(),
                    ptr_->Cols() - range_.Cols()};
  }

  const_iterator cbegin() const {
    return const_iterator{ptr_->RawBegin() + range_.RowBegin() * ptr_->Cols() + range_.ColBegin(), range_.Cols(),
                          ptr_->Cols() - range_.Cols()};
  }

  iterator end() const {
    return iterator{ptr_->RawBegin() + (range_.RowBegin() + range_.Rows()) * ptr_->Cols() + range_.ColBegin(),
                    range_.Cols(), ptr_->Cols() - range_.Cols()};
  }

  const_iterator cend() const {
    return const_iterator{ptr_->RawBegin() + (range_.RowBegin() + range_.Rows()) * ptr_->Cols() + range_.ColBegin(),
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

  // Functionals.

  template <typename UnaryOp>
  BaseMatrixView& Apply(UnaryOp op)
    requires(!kIsConst)
  {
    assert(IsValidMatrixView() && "Matrix view should not be empty");

    std::transform(cbegin(), cend(), begin(), std::move(op));
    return *this;
  }

  template <typename UnaryOp>
  BaseMatrixView& ApplyOnDiagonal(UnaryOp op)
    requires(!kIsConst)
  {
    assert(Rows() == Cols() && "Matrix should be square");
    assert(IsValidMatrixView() && "Matrix view should not be empty");

    for (size_type i = 0; i < Rows(); ++i) {
      (*this)(i, i) = op((*this)(i, i));
    }
    return *this;
  }

 private:
  bool IsValidMatrixView() const {
    return ptr_ != nullptr;
  }

  MyMatrix* ptr_{};
  SubmatrixRange range_;
};
}  // namespace linalg::view

#endif
