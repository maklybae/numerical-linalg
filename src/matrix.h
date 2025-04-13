#ifndef MATRIX_H
#define MATRIX_H

#include <algorithm>
#include <cassert>
#include <concepts>
#include <initializer_list>
#include <iterator>
#include <type_traits>
#include <utility>

#include "core_types.h"
#include "iterators.h"
#include "matrix_types.h"
#include "matrix_view.h"
#include "scalar_types.h"
#include "scalar_utils.h"
#include "submatrix_range.h"

namespace linalg {
template <detail::FloatingOrComplexType Scalar>
class Matrix {
  static_assert(!std::is_const_v<Scalar> && !std::is_volatile_v<Scalar>,
                "Scalar type must not be const and not be volatile");

  using StorageType          = detail::Storage<Scalar>;
  using UnderlyingSize       = StorageType::size_type;
  using StorageIterator      = detail::StorageIterator<Scalar>;
  using ConstStorageIterator = detail::ConstStorageIterator<Scalar>;

 public:
  using RowIterator      = detail::iterators::RowIterator<Scalar>;
  using ConstRowIterator = detail::iterators::ConstRowIterator<Scalar>;
  using RRowIterator     = std::reverse_iterator<RowIterator>;
  using CRRRowIterator   = std::reverse_iterator<ConstRowIterator>;
  using ColIterator      = detail::iterators::BlockIterator<Scalar>;
  using ConstColIterator = detail::iterators::ConstBlockIterator<Scalar>;
  using RColIterator     = std::reverse_iterator<ColIterator>;
  using CRColIterator    = std::reverse_iterator<ConstColIterator>;

  // NOLINTBEGIN(readability-identifier-naming)
  using value_type             = Scalar;
  using reference              = Scalar&;
  using const_reference        = const Scalar&;
  using iterator               = RowIterator;
  using const_iterator         = ConstRowIterator;
  using reverse_iterator       = RRowIterator;
  using const_reverse_iterator = CRRRowIterator;
  using difference_type        = Difference;
  using size_type              = Size;
  // NOLINTEND(readability-identifier-naming)

  template <typename T, detail::ConstnessEnum>
  friend class detail::BaseMatrixView;

  Matrix() = default;

  Matrix(const Matrix&) = default;

  // Needs to leave other in valid state.
  Matrix(Matrix&& other) noexcept : rows_{std::exchange(other.rows_, 0)}, data_{std::exchange(other.data_, {})} {}

  // Use static cast to call private helper ctor Matrix(size_type, size_type).
  Matrix(ERows rows, ECols cols) : Matrix(static_cast<size_type>(rows), static_cast<size_type>(cols)) {}

  explicit Matrix(MatrixView<Scalar> view) : rows_{view.Rows()}, data_(view.begin(), view.end()) {
    assert(IsValidMatrix() && "Matrix data should not be empty");
  }

  explicit Matrix(ConstMatrixView<Scalar> view) : rows_{view.Rows()}, data_(view.begin(), view.end()) {
    assert(IsValidMatrix() && "Matrix data should not be empty");
  }

  Matrix(std::initializer_list<std::initializer_list<Scalar>> list) {
    rows_ = ToSizeType(list.size());
    assert(rows_ > 0 && "Matrix should have at least one row");

    auto tmp_cols = list.begin()->size();
    assert(tmp_cols > 0 && "Matrix should have at least one col");

    data_.reserve(list.size() * tmp_cols);
    for (auto row : list) {
      assert(row.size() == tmp_cols && "All rows should have the same size");
      data_.insert(data_.end(), row);
    }
  }

  Matrix& operator=(const Matrix&) = default;

  // Needs to leave other in valid state.
  Matrix& operator=(Matrix&& other) noexcept {
    Matrix tmp = std::move(other);
    std::swap(rows_, tmp.rows_);
    std::swap(data_, tmp.data_);

    assert(other.rows_ == 0 && "Moved matrix should be empty");
    assert(other.data_.empty() && "Moved matrix should be empty");
    return *this;
  }

  // NOLINTNEXTLINE(readability-identifier-naming)
  friend void swap(Matrix& lhs, Matrix& rhs) {
    std::swap(lhs.rows_, rhs.rows_);
    std::swap(lhs.data_, rhs.data_);
  }

  ~Matrix() = default;

  // Size and data access methods.

  size_type Rows() const {
    return rows_;
  }

  size_type Cols() const {
    return rows_ == 0 ? 0 : ToSizeType(data_.size()) / rows_;
  }

  Scalar& operator()(Index row, Index col) {
    assert(row >= 0 && "Row index should be non-negative");
    assert(row < Rows() && "Row index out of bounds");
    assert(col >= 0 && "Col index should be non-negative");
    assert(col < Cols() && "Col index out of bounds");

    // To avoid -Wsign-conversion.
    return data_[ToUnderlyingSize(row * Cols() + col)];
  }

  Scalar operator()(Index row, Index col) const {
    assert(row >= 0 && "Row index should be non-negative");
    assert(row < Rows() && "Row index out of bounds");
    assert(col >= 0 && "Col index should be non-negative");
    assert(col < Cols() && "Col index out of bounds");

    // To avoid -Wsign-conversion.
    return data_[ToUnderlyingSize(row * Cols() + col)];
  }

  // Iterators.

  // Row-wise iterators. Satisfy ContiguousIterator.
  RowIterator RowWiseBegin() {
    return RowIterator{data_.begin()};
  }
  ConstRowIterator RowWiseBegin() const {
    return ConstRowIterator{data_.begin()};
  }
  ConstRowIterator RowWiseCBegin() const {
    return ConstRowIterator{data_.cbegin()};
  }

  RowIterator RowWiseEnd() {
    return RowIterator{data_.end()};
  }
  ConstRowIterator RowWiseEnd() const {
    return ConstRowIterator{data_.end()};
  }
  ConstRowIterator RowWiseCEnd() const {
    return ConstRowIterator{data_.cend()};
  }

  RRowIterator RowWiseRBegin() {
    return RRowIterator(RowWiseEnd());
  }
  CRRRowIterator RowWiseRBegin() const {
    return CRRRowIterator(RowWiseEnd());
  }
  CRRRowIterator RowWiseCRBegin() const {
    return CRRRowIterator(RowWiseCEnd());
  }

  RRowIterator RowWiseREnd() {
    return RRowIterator(RowWiseBegin());
  }
  CRRRowIterator RowWiseREnd() const {
    return CRRRowIterator(RowWiseBegin());
  }
  CRRRowIterator RowWiseCREnd() const {
    return CRRRowIterator(RowWiseCBegin());
  }

  // Col-wise iterators. Satisfy BidirectionalIterator.
  ColIterator ColWiseBegin() {
    return ColIterator{StorageIteratorBegin(), ColWiseStepSize(), ColWiseMaxStep(), ColWiseShift(),
                       StorageIteratorColWiseEnd()};
  }
  ConstColIterator ColWiseBegin() const {
    return ConstColIterator{StorageIteratorBegin(), ColWiseStepSize(), ColWiseMaxStep(), ColWiseShift(),
                            StorageIteratorColWiseEnd()};
  }
  ConstColIterator ColWiseCBegin() const {
    return ConstColIterator{StorageIteratorBegin(), ColWiseStepSize(), ColWiseMaxStep(), ColWiseShift(),
                            StorageIteratorColWiseEnd()};
  }

  ColIterator ColWiseEnd() {
    return ColIterator{StorageIteratorColWiseEnd(),
                       ColWiseStepSize(),
                       ColWiseMaxStep(),
                       ColWiseShift(),
                       StorageIteratorColWiseEnd(),
                       Rows()};
  }
  ConstColIterator ColWiseEnd() const {
    return ConstColIterator{StorageIteratorColWiseEnd(),
                            ColWiseStepSize(),
                            ColWiseMaxStep(),
                            ColWiseShift(),
                            StorageIteratorColWiseEnd(),
                            Rows()};
  }
  ConstColIterator ColWiseCEnd() const {
    return ConstColIterator{StorageIteratorColWiseEnd(),
                            ColWiseStepSize(),
                            ColWiseMaxStep(),
                            ColWiseShift(),
                            StorageIteratorColWiseEnd(),
                            Rows()};
  }

  RColIterator ColWiseRBegin() {
    return RColIterator(ColWiseEnd());
  }
  CRColIterator ColWiseRBegin() const {
    return CRColIterator(ColWiseEnd());
  }
  CRColIterator ColWiseCRBegin() const {
    return CRColIterator(ColWiseCEnd());
  }

  RColIterator ColWiseREnd() {
    return RColIterator(ColWiseBegin());
  }
  CRColIterator ColWiseREnd() const {
    return CRColIterator(ColWiseBegin());
  }
  CRColIterator ColWiseCREnd() const {
    return CRColIterator(ColWiseCBegin());
  }

  // STL-like iterators.
  // NOLINTBEGIN(readability-identifier-naming)
  iterator begin() {
    return RowIterator{data_.begin()};
  }
  const_iterator begin() const {
    return ConstRowIterator{data_.begin()};
  }
  const_iterator cbegin() const {
    return ConstRowIterator{data_.cbegin()};
  }

  iterator end() {
    return RowIterator{data_.end()};
  }
  const_iterator end() const {
    return ConstRowIterator{data_.end()};
  }
  const_iterator cend() const {
    return ConstRowIterator{data_.cend()};
  }

  reverse_iterator rbegin() {
    return reverse_iterator(end());
  }
  const_reverse_iterator rbegin() const {
    return reverse_iterator(end());
  }
  const_reverse_iterator crbegin() const {
    return reverse_iterator(cend());
  }

  reverse_iterator rend() {
    return reverse_iterator(begin());
  }
  const_reverse_iterator rend() const {
    return const_reverse_iterator(begin());
  }
  const_reverse_iterator crend() const {
    return const_reverse_iterator(cbegin());
  }
  // NOLINTEND(readability-identifier-naming)

  // Functionals.

  template <typename UnaryOp>
  Matrix& Apply(UnaryOp op) {
    assert(IsValidMatrix() && "Matrix data should not be empty");

    std::transform(cbegin(), cend(), begin(), std::move(op));
    return *this;
  }

  template <typename UnaryOp>
  Matrix& ApplyOnDiagonal(UnaryOp op) {
    assert(Rows() == Cols() && "Matrix should be square");
    assert(IsValidMatrix() && "Matrix data should not be empty");

    for (size_type i = 0; i < rows_; ++i) {
      (*this)(i, i) = op((*this)(i, i));
    }
    return *this;
  }

  // Submatrix getters.
  MatrixView<Scalar> Submatrix(SubmatrixRange range) {
    return MatrixView<Scalar>(*this, range);
  }

  ConstMatrixView<Scalar> Submatrix(SubmatrixRange range) const {
    return ConstMatrixView<Scalar>(*this, range);
  }

  MatrixView<Scalar> Row(Index row) {
    return Submatrix(SubmatrixRange::FromBeginSize(ERowBegin{row}, ERows{1}, EColBegin{0}, ECols{Cols()}));
  }

  ConstMatrixView<Scalar> Row(Index row) const {
    return Submatrix(SubmatrixRange::FromBeginSize(ERowBegin{row}, ERows{1}, EColBegin{0}, ECols{Cols()}));
  }

  MatrixView<Scalar> Col(Index col) {
    return Submatrix(SubmatrixRange::FromBeginSize(ERowBegin{0}, ERows{Rows()}, EColBegin{col}, ECols{1}));
  }

  ConstMatrixView<Scalar> Col(Index col) const {
    return Submatrix(SubmatrixRange::FromBeginSize(ERowBegin{0}, ERows{Rows()}, EColBegin{col}, ECols{1}));
  }

  // Static creation methods.

  static Matrix Identity(size_type size) {
    Matrix identity(size, size);
    identity.ApplyOnDiagonal([](Scalar) { return 1; });
    return identity;
  }

  static Matrix Zero(ERows rows, ECols cols) {
    return Matrix(rows, cols);
  }

  static Matrix SingleEntry(ERows rows, ECols cols, ERow unit_row, ECol unit_col) {
    Matrix unit(rows, cols);
    unit(unit_row, unit_col) = 1;
    return unit;
  }

  static Matrix ScalarMatrix(size_type size, Scalar value) {
    Matrix diagonal(size, size);
    diagonal.ApplyOnDiagonal([value](Scalar) { return value; });
    return diagonal;
  }

  static Matrix Diagonal(std::initializer_list<Scalar> list) {
    Matrix diagonal(ToSizeType(list.size()), ToSizeType(list.size()));
    auto iter = list.begin();
    diagonal.ApplyOnDiagonal([&iter](Scalar) { return *iter++; });
    return diagonal;
  }

  template <std::input_iterator InputIterator>
  static Matrix Diagonal(InputIterator first, InputIterator last) {
    static_assert(std::is_same_v<typename std::iterator_traits<InputIterator>::value_type, Scalar>,
                  "Iterator value type should be the same as matrix scalar type");
    auto size{std::distance(first, last)};
    assert(size >= 0 && "Iterator range should be non-negative");

    Matrix diagonal(size, size);
    diagonal.ApplyOnDiagonal([&first](Scalar) { return *first++; });
    return diagonal;
  }

 private:
  Matrix(size_type rows, size_type cols) {
    assert(rows > 0 && "Matrix should have at least one row");
    assert(cols > 0 && "Matrix should have at least one col");

    rows_ = rows;
    data_.resize(ToUnderlyingSize(rows_ * cols));
  }

  // Needs to use in MatrixView to build BlockIterators.
  StorageIterator StorageIteratorBegin() {
    return data_.begin();
  }
  ConstStorageIterator StorageIteratorBegin() const {
    return data_.cbegin();
  }

  StorageIterator StorageIteratorEnd() {
    return data_.end();
  }
  ConstStorageIterator StorageIteratorEnd() const {
    return data_.cend();
  }

  bool IsValidMatrix() const {
    return !data_.empty() && rows_ > 0;
  }

  template <typename BinaryOp>
  Matrix& Apply(const Matrix& rhs, BinaryOp op) {
    assert(Rows() == rhs.Rows() && "Matrix rows should be equal");
    assert(Cols() == rhs.Cols() && "Matrix cols should be equal");

    std::transform(cbegin(), cend(), rhs.cbegin(), begin(), std::move(op));
    return *this;
  }

  StorageIterator StorageIteratorColWiseEnd() {
    return StorageIteratorEnd() + Cols() - 1;
  }
  ConstStorageIterator StorageIteratorColWiseEnd() const {
    return StorageIteratorEnd() + Cols() - 1;
  }

  Size ColWiseStepSize() const {
    return Cols();
  }

  Size ColWiseMaxStep() const {
    return Rows();
  }

  Difference ColWiseShift() const {
    return -Cols() * Rows() + 1;
  }

  static UnderlyingSize ToUnderlyingSize(size_type size) {
    assert(size >= 0 && "Size should be non-negative");
    return static_cast<UnderlyingSize>(size);
  }

  static size_type ToSizeType(UnderlyingSize size) {
    assert(size <= std::numeric_limits<size_type>::max() && "Size should fit into size_type");
    return static_cast<size_type>(size);
  }

  size_type rows_{};
  StorageType data_{};
};

namespace detail {

// Not public functionals.

template <typename BinaryOp, MutableMatrixType LhsT, MatrixType RhsT>
LhsT& Apply(LhsT& lhs, const RhsT& rhs, BinaryOp op) {
  assert(lhs.Rows() == rhs.Rows() && "Matrix rows should be equal");
  assert(lhs.Cols() == rhs.Cols() && "Matrix cols should be equal");

  std::transform(lhs.cbegin(), lhs.cend(), rhs.cbegin(), lhs.begin(), std::move(op));
  return lhs;
}

// Fix zeros.

template <detail::MutableMatrixType MatrixT>
void FixZeros(MatrixT& matrix,
              UnderlyingScalarT<typename MatrixT::value_type> epsilon = kEpsilon<typename MatrixT::value_type>) {
  using Scalar = typename MatrixT::value_type;
  matrix.Apply([epsilon](Scalar value) { return FixZeros(value, epsilon); });
}

}  // namespace detail

// Unary operators.
template <detail::MatrixType MatrixT>
Matrix<typename MatrixT::value_type> operator-(const MatrixT& matrix) {
  Matrix<typename MatrixT::value_type> result(matrix);
  result.Apply(std::negate<>{});
  return result;
}

// Compare operators.
template <detail::MatrixType LhsT, detail::MatrixType RhsT>
bool operator==(const LhsT& lhs, const RhsT& rhs) {
  if (lhs.Rows() != rhs.Rows() || lhs.Cols() != rhs.Cols()) {
    return false;
  }

  return std::equal(lhs.cbegin(), lhs.cend(), rhs.cbegin(), rhs.cend(),
                    [](typename LhsT::value_type a, typename RhsT::value_type b) { return detail::ApproxEqual(a, b); });
}

template <detail::MatrixType LhsT, detail::MatrixType RhsT>
bool operator!=(const LhsT& lhs, const RhsT& rhs) {
  return !(lhs == rhs);
}

// In-place binary operators.
template <detail::MutableMatrixType LhsT, detail::MatrixType RhsT>
LhsT& operator+=(LhsT& lhs, const RhsT& rhs) {
  return detail::Apply(lhs, rhs, std::plus<>{});
}

template <detail::MutableMatrixType LhsT, detail::MatrixType RhsT>
LhsT& operator-=(LhsT& lhs, const RhsT& rhs) {
  return detail::Apply(lhs, rhs, std::minus<>{});
}

template <detail::MutableMatrixType LhsT, detail::MatrixType RhsT>
LhsT& operator*=(LhsT& lhs, const RhsT& rhs) {
  lhs = lhs * rhs;
  return lhs;
}

template <detail::MutableMatrixType LhsT>
LhsT& operator+=(LhsT& lhs, typename LhsT::value_type scalar) {
  return lhs.ApplyOnDiagonal([scalar](typename LhsT::value_type value) { return value + scalar; });
}

template <detail::MutableMatrixType LhsT>
LhsT& operator-=(LhsT& lhs, typename LhsT::value_type scalar) {
  return lhs.ApplyOnDiagonal([scalar](typename LhsT::value_type value) { return value - scalar; });
}

template <detail::MutableMatrixType LhsT>
LhsT& operator*=(LhsT& lhs, typename LhsT::value_type scalar) {
  return lhs.Apply([scalar](typename LhsT::value_type value) { return value * scalar; });
}

template <detail::MutableMatrixType LhsT>
LhsT& operator/=(LhsT& lhs, typename LhsT::value_type scalar) {
  return lhs.Apply([scalar](typename LhsT::value_type value) { return value / scalar; });
}

// Binary operators.
template <detail::MatrixType LhsT, detail::MatrixType RhsT>
Matrix<detail::CommonValueType<LhsT, RhsT>> operator+(const LhsT& lhs, const RhsT& rhs) {
  using Scalar = detail::CommonValueType<LhsT, RhsT>;
  Matrix<Scalar> result(lhs);
  result += rhs;
  return result;
}

template <detail::MatrixType LhsT, detail::MatrixType RhsT>
Matrix<detail::CommonValueType<LhsT, RhsT>> operator-(const LhsT& lhs, const RhsT& rhs) {
  using Scalar = detail::CommonValueType<LhsT, RhsT>;
  Matrix<Scalar> result(lhs);
  result -= rhs;
  return result;
}

template <detail::MatrixType LhsT, detail::MatrixType RhsT>
Matrix<detail::CommonValueType<LhsT, RhsT>> operator*(const LhsT& lhs, const RhsT& rhs) {
  using Scalar = detail::CommonValueType<LhsT, RhsT>;

  Matrix<Scalar> result(ERows{lhs.Rows()}, ECols{rhs.Cols()});

  for (Size i = 0; i < lhs.Rows(); ++i) {
    for (Size j = 0; j < rhs.Cols(); ++j) {
      for (Size k = 0; k < lhs.Cols(); ++k) {
        result(i, j) += lhs(i, k) * rhs(k, j);
      }
    }
  }

  // detail::FixZeros(result); // Use to compare convergence

  return result;
}

template <detail::MatrixType MatrixT>
Matrix<typename MatrixT::value_type> operator+(const MatrixT& lhs, typename MatrixT::value_type scalar) {
  Matrix<typename MatrixT::value_type> result(lhs);
  result += scalar;
  return result;
}

template <detail::MatrixType MatrixT>
Matrix<typename MatrixT::value_type> operator+(typename MatrixT::value_type scalar, const MatrixT& rhs) {
  Matrix<typename MatrixT::value_type> result(rhs);
  result += scalar;
  return result;
}

template <detail::MatrixType MatrixT>
Matrix<typename MatrixT::value_type> operator-(const MatrixT& lhs, typename MatrixT::value_type scalar) {
  Matrix<typename MatrixT::value_type> result(lhs);
  result -= scalar;
  return result;
}

template <detail::MatrixType MatrixT>
Matrix<typename MatrixT::value_type> operator*(const MatrixT& lhs, typename MatrixT::value_type scalar) {
  Matrix<typename MatrixT::value_type> result(lhs);
  result *= scalar;
  return result;
}

template <detail::MatrixType MatrixT>
Matrix<typename MatrixT::value_type> operator*(typename MatrixT::value_type scalar, const MatrixT& rhs) {
  Matrix<typename MatrixT::value_type> result(rhs);
  result *= scalar;
  return result;
}

template <detail::MatrixType MatrixT>
Matrix<typename MatrixT::value_type> operator/(const MatrixT& lhs, typename MatrixT::value_type scalar) {
  Matrix<typename MatrixT::value_type> result(lhs);
  result /= scalar;
  return result;
}

// Conjugate transpose and transpose.
// Transposed functions for MatrixView and ConstMatrixView is in matrix_view.h.
// Conjugated returns view for fp matrices and matrix for complex matrices.
// Reasons for this is unable to change conjugated state in matrix view.

template <typename Scalar>
MatrixView<Scalar> Transposed(Matrix<Scalar>& matrix) {
  auto view = MatrixView<Scalar>{matrix};
  return Transposed(view);
}

template <typename Scalar>
ConstMatrixView<Scalar> Transposed(const Matrix<Scalar>& matrix) {
  auto view = ConstMatrixView<Scalar>{matrix};
  return Transposed(view);
}

template <std::floating_point FPScalar>
MatrixView<FPScalar> Conjugated(Matrix<FPScalar>& matrix) {
  return Transposed(matrix);
}

template <std::floating_point FPScalar>
ConstMatrixView<FPScalar> Conjugated(const Matrix<FPScalar>& matrix) {
  return Transposed(matrix);
}

template <detail::ComplexMatrixType CMatrixT>
Matrix<typename CMatrixT::value_type> Conjugated(const CMatrixT& matrix) {
  using Scalar = typename CMatrixT::value_type;

  auto view        = Transposed(matrix);
  auto conj_matrix = Matrix<Scalar>(view);
  conj_matrix.Apply([](Scalar value) { return std::conj(value); });
  return conj_matrix;
}

// Vector norm.

template <detail::MatrixType VectorT>
detail::UnderlyingScalarT<typename VectorT::value_type> EuclideanVectorNorm(const VectorT& vector) {
  assert((vector.Cols() == 1 || vector.Rows() == 1) && "Matrix should be a vector");

  using Scalar = detail::UnderlyingScalarT<typename VectorT::value_type>;

  Scalar result{};
  for (auto value : vector) {
    result += std::norm(value);
  }

  return std::sqrt(result);
}

template <detail::MutableMatrixType VectorT>
void NormalizeVector(VectorT& vector) {
  using Scalar = typename VectorT::value_type;

  assert((vector.Cols() == 1 || vector.Rows() == 1) && "Matrix should be a vector");

  auto norm = EuclideanVectorNorm(vector);
  if (detail::ApproxZero(norm)) {
    return;
  }

  vector.Apply([norm](Scalar value) { return value / norm; });
}

// Copy functions.

// We need to use it to change only submatrix.
// Probably it should be in operator= of MatrixView, but it is used for copying pointers.
template <detail::MatrixType MatrixFromT, detail::MutableMatrixType MatrixToT>
void CopyMatrix(const MatrixFromT& from, MatrixToT& to) {
  assert(from.Rows() == to.Rows() && "Matrix rows should be equal");
  assert(from.Cols() == to.Cols() && "Matrix cols should be equal");

  std::copy(from.cbegin(), from.cend(), to.begin());
}

namespace detail {

// Casts matrix to underlying scalar type.

// WARNING: This function returns copy of matrix with omitting imaginary part.
// Returns copy on matrix with initially floating point type.
template <detail::MatrixType MatrixT>
Matrix<UnderlyingScalarT<typename MatrixT::value_type>> CastToUnderlyingScalarMatrix(const MatrixT& matrix) {
  return matrix;
}

template <detail::ComplexMatrixType MatrixT>
Matrix<UnderlyingScalarT<typename MatrixT::value_type>> CastToUnderlyingScalarMatrix(const MatrixT& matrix) {
  using Scalar = UnderlyingScalarT<typename MatrixT::value_type>;

  Matrix<Scalar> result(ERows{matrix.Rows()}, ECols{matrix.Cols()});
  std::transform(matrix.cbegin(), matrix.cend(), result.begin(),
                 [](std::complex<Scalar> value) { return std::abs(value); });
  return result;
}

}  // namespace detail

}  // namespace linalg

#endif  // MATRIX_H
