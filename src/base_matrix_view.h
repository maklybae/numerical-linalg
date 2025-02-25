#ifndef BASE_MATRIX_VIEW_H
#define BASE_MATRIX_VIEW_H

#include <type_traits>

#include "core_types.h"
#include "iterators.h"
#include "matrix.h"
#include "submatrix_range.h"

namespace linalg {
namespace detail {

template <typename Scalar, ConstnessEnum Constness>
class BaseMatrixView {
  static constexpr bool kIsConst = Constness == ConstnessEnum::kConst;

  using MyMatrix                   = std::conditional_t<kIsConst, const Matrix<Scalar>, Matrix<Scalar>>;
  using ReturnType                 = std::conditional_t<kIsConst, Scalar, Scalar&>;
  using StorageIterator            = MyMatrix::StorageIterator;
  using BasicRowBlockIterator      = iterators::RowBlockIterator<Scalar>;
  using BasicConstRowBlockIterator = iterators::ConstRowBlockIterator<Scalar>;
  using BasicColBlockIterator      = iterators::ColBlockIterator<Scalar>;
  using BasicConstColBlockIterator = iterators::ConstColBlockIterator<Scalar>;

 public:
  using RowBlockIterator      = std::conditional_t<kIsConst, BasicConstRowBlockIterator, BasicRowBlockIterator>;
  using ConstRowBlockIterator = BasicConstRowBlockIterator;
  using RRowBlockIterator     = std::conditional_t<kIsConst, std::reverse_iterator<RowBlockIterator>,
                                                   std::reverse_iterator<ConstRowBlockIterator>>;
  using CRRowBlockIterator    = std::reverse_iterator<ConstRowBlockIterator>;

  using ColBlockIterator      = std::conditional_t<kIsConst, BasicConstColBlockIterator, BasicColBlockIterator>;
  using ConstColBlockIterator = BasicConstColBlockIterator;
  using RColBlockIterator     = std::conditional_t<kIsConst, std::reverse_iterator<ColBlockIterator>,
                                                   std::reverse_iterator<ConstColBlockIterator>>;
  using CRColBlockIterator    = std::reverse_iterator<ConstColBlockIterator>;

  // NOLINTBEGIN(readability-identifier-naming)
  using value_type             = Scalar;
  using reference              = Scalar&;
  using const_reference        = const Scalar&;
  using iterator               = RowBlockIterator;
  using const_iterator         = ConstRowBlockIterator;
  using reverse_iterator       = RRowBlockIterator;
  using const_reverse_iterator = CRRowBlockIterator;
  using difference_type        = Difference;
  using size_type              = Size;
  // NOLINTEND(readability-identifier-naming)

  // Needs to generate a ctor from non-const to const view.
  template <typename, ConstnessEnum>
  friend class BaseMatrixView;

  BaseMatrixView()                                 = default;
  BaseMatrixView(const BaseMatrixView&)            = default;
  BaseMatrixView& operator=(const BaseMatrixView&) = default;

  BaseMatrixView(BaseMatrixView&& other) noexcept
      : ptr_{std::exchange(other, nullptr)}, range_{std::exchange(other, {})} {}
  BaseMatrixView& operator=(BaseMatrixView&&) noexcept {
    BaseMatrixView tmp = std::move(*this);
    std::swap(ptr_, tmp.ptr_);
    std::swap(range_, tmp.range_);
    return *this;
  }

  // NOLINTNEXTLINE(google-explicit-constructor)
  BaseMatrixView(MyMatrix& matrix)
      : ptr_{&matrix}, range_{SubmatrixRange::FullMatrix(ERows{matrix.Rows()}, ECols{matrix.Cols()})} {
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

  // Iterators

  // Row-wise iterators.
  RowBlockIterator RowWiseBegin() const {
    return RowBlockIterator{StorageIteratorBegin(), RowWiseStepSize(), RowWiseMaxStep(), RowWiseShift()};
  }

  ConstRowBlockIterator RowWiseCBegin() const {
    return ConstRowBlockIterator{StorageIteratorBegin(), RowWiseStepSize(), RowWiseMaxStep(), RowWiseShift()};
  }

  RowBlockIterator RowWiseEnd() const {
    return RowBlockIterator{StorageIteratorRowWiseEnd(), RowWiseStepSize(), RowWiseMaxStep(), RowWiseShift()};
  }

  ConstRowBlockIterator RowWiseCEnd() const {
    return ConstRowBlockIterator{StorageIteratorRowWiseEnd(), RowWiseStepSize(), RowWiseMaxStep(), RowWiseShift()};
  }

  RRowBlockIterator RowWiseRBegin() const {
    return RRowBlockIterator{RowWiseEnd()};
  }

  CRRowBlockIterator RowWiseCRBegin() const {
    return CRRowBlockIterator{RowWiseCEnd()};
  }

  RRowBlockIterator RowWiseREnd() const {
    return RRowBlockIterator{RowWiseBegin()};
  }

  CRRowBlockIterator RowWiseCREnd() const {
    return CRRowBlockIterator{RowWiseCBegin()};
  }

  // Column-wise iterators.
  ColBlockIterator ColWiseBegin() const {
    return ColBlockIterator{StorageIteratorBegin(), StorageIteratorColWiseEnd(), ColWiseStepSize(), ColWiseMaxStep(),
                            ColWiseShift()};
  }

  ConstColBlockIterator ColWiseCBegin() const {
    return ConstColBlockIterator{StorageIteratorBegin(), StorageIteratorColWiseEnd(), ColWiseStepSize(),
                                 ColWiseMaxStep(), ColWiseShift()};
  }

  ColBlockIterator ColWiseEnd() const {
    return ColBlockIterator{StorageIteratorColWiseEnd(),
                            StorageIteratorColWiseEnd(),
                            ColWiseStepSize(),
                            ColWiseMaxStep(),
                            ColWiseShift(),
                            Rows()};
  }

  ConstColBlockIterator ColWiseCEnd() const {
    return ConstColBlockIterator{StorageIteratorColWiseEnd(),
                                 StorageIteratorColWiseEnd(),
                                 ColWiseStepSize(),
                                 ColWiseMaxStep(),
                                 ColWiseShift(),
                                 Rows()};
  }

  RColBlockIterator ColWiseRBegin() const {
    return RColBlockIterator{ColWiseEnd()};
  }

  CRColBlockIterator ColWiseCRBegin() const {
    return CRColBlockIterator{ColWiseCEnd()};
  }

  RColBlockIterator ColWiseREnd() const {
    return RColBlockIterator{ColWiseBegin()};
  }

  CRColBlockIterator ColWiseCREnd() const {
    return CRColBlockIterator{ColWiseCBegin()};
  }

  // STL-like iterators.
  // WARNING: Row-wise as default.
  // NOLINTBEGIN(readability-identifier-naming)
  iterator begin() const {
    return RowWiseBegin();
  }

  const_iterator cbegin() const {
    return RowWiseCBegin();
  }

  iterator end() const {
    return RowWiseEnd();
  }

  const_iterator cend() const {
    return RowWiseCEnd();
  }

  reverse_iterator rbegin() const {
    return RowWiseRBegin();
  }

  const_reverse_iterator crbegin() const {
    return RowWiseCRBegin();
  }

  reverse_iterator rend() const {
    return RowWiseREnd();
  }

  const_reverse_iterator crend() const {
    return RowWiseCREnd();
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
    return ptr_ != nullptr && Rows() > 0 && Cols() > 0;
  }

  StorageIterator StorageIteratorBegin() const {
    return ptr_->StorageIteratorBegin() + range_.RowBegin() * ptr_->Cols() + range_.ColBegin();
  }

  StorageIterator StorageIteratorRowWiseEnd() const {
    return ptr_->StorageIteratorBegin() + (range_.RowBegin() + range_.Rows()) * ptr_->Cols() + range_.ColBegin();
  }

  StorageIterator StorageIteratorColWiseEnd() const {
    return ptr_->StorageIteratorBegin() + (range_.RowBegin() + range_.Rows()) * ptr_->Cols() + range_.ColEnd() - 1;
  }

  Size RowWiseStepSize() const {
    return 1;
  }

  Size ColWiseStepSize() const {
    return ptr_->Cols();
  }

  Size RowWiseMaxStep() const {
    return range_.Cols();
  }

  Size ColWiseMaxStep() const {
    return range_.Rows();
  }

  Difference RowWiseShift() const {
    return ptr_->Cols() - range_.Cols();
  }

  Difference ColWiseShift() const {
    return -ptr_->Cols() * range_.Rows() + 1;
  }

  MyMatrix* ptr_{};
  SubmatrixRange range_{};
};

}  // namespace detail
}  // namespace linalg

#endif  // BASE_MATRIX_VIEW_H
