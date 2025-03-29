#ifndef BASE_MATRIX_VIEW_H
#define BASE_MATRIX_VIEW_H

#include <type_traits>
#include <utility>

#include "core_types.h"
#include "iterators.h"
#include "matrix.h"
#include "matrix_types.h"
#include "submatrix_range.h"

namespace linalg {
namespace detail {

template <typename Scalar, ConstnessEnum Constness>
class BaseMatrixView {
  static constexpr bool kIsConst = Constness == ConstnessEnum::kConst;

  using MyMatrix   = std::conditional_t<kIsConst, const Matrix<Scalar>, Matrix<Scalar>>;
  using ReturnType = std::conditional_t<kIsConst, Scalar, Scalar&>;
  using StorageIterator =
      std::conditional_t<kIsConst, typename MyMatrix::ConstStorageIterator, typename MyMatrix::StorageIterator>;
  using BasicBlockIterator      = iterators::BlockIterator<Scalar>;
  using BasicConstBlockIterator = iterators::ConstBlockIterator<Scalar>;

 public:
  using BlockIterator      = std::conditional_t<kIsConst, BasicConstBlockIterator, BasicBlockIterator>;
  using ConstBlockIterator = BasicConstBlockIterator;
  using RBlockIterator =
      std::conditional_t<kIsConst, std::reverse_iterator<ConstBlockIterator>, std::reverse_iterator<BlockIterator>>;
  using CRBlockIterator = std::reverse_iterator<ConstBlockIterator>;

  // NOLINTBEGIN(readability-identifier-naming)
  using value_type             = Scalar;
  using reference              = Scalar&;
  using const_reference        = const Scalar&;
  using iterator               = BlockIterator;
  using const_iterator         = ConstBlockIterator;
  using reverse_iterator       = RBlockIterator;
  using const_reverse_iterator = CRBlockIterator;
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
      : ptr_{std::exchange(other.ptr_, nullptr)}
      , range_{std::exchange(other.range_, {})}
      , state_{std::exchange(other.state_, {})} {}
  BaseMatrixView& operator=(BaseMatrixView&& other) noexcept {
    BaseMatrixView tmp = std::move(other);
    std::swap(ptr_, tmp.ptr_);
    std::swap(range_, tmp.range_);
    std::swap(state_, tmp.state_);
    return *this;
  }

  // NOLINTNEXTLINE(google-explicit-constructor)
  BaseMatrixView(MyMatrix& matrix)
      : ptr_{&matrix}, range_{SubmatrixRange::FullMatrix(ERows{matrix.Rows()}, ECols{matrix.Cols()})} {
    assert(matrix.IsValidMatrix() && "Matrix data should not be empty");
  }

  // Non-const to const view conversions.

  // NOLINTNEXTLINE(google-explicit-constructor)
  BaseMatrixView(BaseMatrixView<Scalar, ConstnessEnum::kNonConst>& other)
    requires kIsConst
      : ptr_{other.ptr_}, range_{other.range_}, state_{other.state_} {}

  // NOLINTNEXTLINE(google-explicit-constructor)
  BaseMatrixView(BaseMatrixView<Scalar, ConstnessEnum::kNonConst>&& other) noexcept
    requires kIsConst
      : ptr_{std::exchange(other.ptr_, nullptr)}
      , range_{std::exchange(other.range_, {})}
      , state_{std::exchange(other.state_, {})} {}

  BaseMatrixView(MyMatrix& matrix, SubmatrixRange range) : ptr_{&matrix}, range_{range} {
    assert(matrix.IsValidMatrix() && "Matrix data should not be empty");
    assert(range.RowBegin() < ptr_->Rows() && "Row begin index out of bounds");
    assert(range.RowEnd() <= ptr_->Rows() && "Row end index out of bounds");
    assert(range.ColBegin() < ptr_->Cols() && "Col begin index out of bounds");
    assert(range.ColEnd() <= ptr_->Cols() && "Col end index out of bounds");
  }

  BaseMatrixView(BaseMatrixView matrix_view, SubmatrixRange range)
      : ptr_{matrix_view.ptr_}, state_{matrix_view.state_} {
    assert(matrix_view.IsValidMatrixView() && "Matrix view should not be empty");
    assert(range.RowBegin() < matrix_view.Rows() && "Row begin index out of bounds");
    assert(range.RowEnd() <= matrix_view.Rows() && "Row end index out of bounds");
    assert(range.ColBegin() < matrix_view.Cols() && "Col begin index out of bounds");
    assert(range.ColEnd() <= matrix_view.Cols() && "Col end index out of bounds");

    if (state_.IsTransposed()) {
      range_ = SubmatrixRange::FromBeginSize(
          ERowBegin{matrix_view.range_.ColBegin() + range.RowBegin()}, ERows{range.Rows()},
          EColBegin{matrix_view.range_.RowBegin() + range.ColBegin()}, ECols{range.Cols()});
    } else {
      range_ = SubmatrixRange::FromBeginSize(
          ERowBegin{matrix_view.range_.RowBegin() + range.RowBegin()}, ERows{range.Rows()},
          EColBegin{matrix_view.range_.ColBegin() + range.ColBegin()}, ECols{range.Cols()});
    }
  }

  BaseMatrixView(MyMatrix&&) = delete;

  ReturnType operator()(size_type row, size_type col) const {
    assert(IsValidMatrixView() && "Matrix view should not be empty");
    assert(row < Rows() && "Row index out of bounds");
    assert(col < Cols() && "Col index out of bounds");

    return state_.IsTransposed() ? (*ptr_)(range_.RowBegin() + col, range_.ColBegin() + row)
                                 : (*ptr_)(range_.RowBegin() + row, range_.ColBegin() + col);
  }

  size_type Rows() const {
    return state_.IsTransposed() ? range_.Cols() : range_.Rows();
  }

  size_type Cols() const {
    return state_.IsTransposed() ? range_.Rows() : range_.Cols();
  }

  // Iterators

  // Row-wise iterators.
  BlockIterator RowWiseBegin() const {
    return state_.IsTransposed() ? BaseColWiseBegin() : BaseRowWiseBegin();
  }

  ConstBlockIterator RowWiseCBegin() const {
    return state_.IsTransposed() ? BaseColWiseCBegin() : BaseRowWiseCBegin();
  }

  BlockIterator RowWiseEnd() const {
    return state_.IsTransposed() ? BaseColWiseEnd() : BaseRowWiseEnd();
  }

  ConstBlockIterator RowWiseCEnd() const {
    return state_.IsTransposed() ? BaseColWiseCEnd() : BaseRowWiseCEnd();
  }

  RBlockIterator RowWiseRBegin() const {
    return RBlockIterator{RowWiseEnd()};
  }

  CRBlockIterator RowWiseCRBegin() const {
    return CRBlockIterator{RowWiseCEnd()};
  }

  RBlockIterator RowWiseREnd() const {
    return RBlockIterator{RowWiseBegin()};
  }

  CRBlockIterator RowWiseCREnd() const {
    return CRBlockIterator{RowWiseCBegin()};
  }

  // Column-wise iterators.
  BlockIterator ColWiseBegin() const {
    return state_.IsTransposed() ? BaseRowWiseBegin() : BaseColWiseBegin();
  }

  ConstBlockIterator ColWiseCBegin() const {
    return state_.IsTransposed() ? BaseRowWiseCBegin() : BaseColWiseCBegin();
  }

  BlockIterator ColWiseEnd() const {
    return state_.IsTransposed() ? BaseRowWiseEnd() : BaseColWiseEnd();
  }

  ConstBlockIterator ColWiseCEnd() const {
    return state_.IsTransposed() ? BaseRowWiseCEnd() : BaseColWiseCEnd();
  }

  RBlockIterator ColWiseRBegin() const {
    return RBlockIterator{ColWiseEnd()};
  }

  CRBlockIterator ColWiseCRBegin() const {
    return CRBlockIterator{ColWiseCEnd()};
  }

  RBlockIterator ColWiseREnd() const {
    return RBlockIterator{ColWiseBegin()};
  }

  CRBlockIterator ColWiseCREnd() const {
    return CRBlockIterator{ColWiseCBegin()};
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

  // Submatrix getters.

  BaseMatrixView Submatrix(SubmatrixRange range) const {
    return BaseMatrixView{*this, range};
  }

  BaseMatrixView Row(Index row) const {
    return Submatrix(SubmatrixRange::FromBeginEnd(ERowBegin{row}, ERowEnd{row + 1}, EColBegin{0}, EColEnd{Cols()}));
  }

  BaseMatrixView Col(Index col) const {
    return Submatrix(SubmatrixRange::FromBeginEnd(ERowBegin{0}, ERowEnd{Rows()}, EColBegin{col}, EColEnd{col + 1}));
  }

  // MatrixView-specific functions.

  BaseMatrixView& Transpose() {
    state_.SwitchTransposed();
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

  BlockIterator BaseRowWiseBegin() const {
    return BlockIterator{StorageIteratorBegin(), RowWiseStepSize(), RowWiseMaxStep(), RowWiseShift(),
                         BlockIterator::kDefaultThreshold};
  }

  ConstBlockIterator BaseRowWiseCBegin() const {
    return ConstBlockIterator{StorageIteratorBegin(), RowWiseStepSize(), RowWiseMaxStep(), RowWiseShift(),
                              BlockIterator::kDefaultThreshold};
  }

  BlockIterator BaseRowWiseEnd() const {
    return BlockIterator{StorageIteratorRowWiseEnd(), RowWiseStepSize(), RowWiseMaxStep(), RowWiseShift(),
                         BlockIterator::kDefaultThreshold};
  }

  ConstBlockIterator BaseRowWiseCEnd() const {
    return ConstBlockIterator{StorageIteratorRowWiseEnd(), RowWiseStepSize(), RowWiseMaxStep(), RowWiseShift(),
                              ConstBlockIterator::kDefaultThreshold};
  }

  BlockIterator BaseColWiseBegin() const {
    return BlockIterator{StorageIteratorBegin(), ColWiseStepSize(), ColWiseMaxStep(), ColWiseShift(),
                         StorageIteratorColWiseEnd()};
  }

  ConstBlockIterator BaseColWiseCBegin() const {
    return ConstBlockIterator{StorageIteratorBegin(), ColWiseStepSize(), ColWiseMaxStep(), ColWiseShift(),
                              StorageIteratorColWiseEnd()};
  }

  BlockIterator BaseColWiseEnd() const {
    return BlockIterator{StorageIteratorColWiseEnd(), ColWiseStepSize(), ColWiseMaxStep(), ColWiseShift(),
                         StorageIteratorColWiseEnd(), range_.Rows()};
  }

  ConstBlockIterator BaseColWiseCEnd() const {
    return ConstBlockIterator{StorageIteratorColWiseEnd(), ColWiseStepSize(), ColWiseMaxStep(), ColWiseShift(),
                              StorageIteratorColWiseEnd(), range_.Rows()};
  }

  MyMatrix* ptr_{};
  SubmatrixRange range_{};
  MatrixState state_{};
};

}  // namespace detail
}  // namespace linalg

#endif  // BASE_MATRIX_VIEW_H
