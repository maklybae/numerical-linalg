#ifndef MATRIX_H
#define MATRIX_H

#include <algorithm>
#include <cassert>
#include <initializer_list>
#include <iterator>
#include <type_traits>
#include <utility>

#include "iterator_helper.h"
#include "scalars.h"
#include "types.h"

namespace linalg::view {
template <typename, types::ConstnessEnum>
class BaseMatrixView;
}

namespace linalg {
enum Rows : types::Size {};
enum Cols : types::Size {};

template <types::FloatingOrComplexType Scalar>
class Matrix {
  using StorageType          = types::Storage<Scalar>;
  using UnderlyingSize       = typename StorageType::size_type;
  using StorageIterator      = typename StorageType::iterator;
  using ConstStorageIterator = typename StorageType::const_iterator;

 public:
  using Index = types::Index;

  using RowIterator      = iterators::RowMovingLogic<iterators::RandomAccessor<iterators::DefaultDefines<Scalar>>>;
  using ConstRowIterator = iterators::RowMovingLogic<iterators::RandomAccessor<iterators::ConstDefines<Scalar>>>;

  // NOLINTBEGIN(readability-identifier-naming)
  using value_type             = Scalar;
  using reference              = Scalar&;
  using const_reference        = const Scalar&;
  using iterator               = RowIterator;
  using const_iterator         = ConstRowIterator;
  using reverse_iterator       = std::reverse_iterator<iterator>;
  using const_reverse_iterator = std::reverse_iterator<const_iterator>;
  using difference_type        = types::Difference;
  using size_type              = types::Size;
  // NOLINTEND(readability-identifier-naming)

  template <typename T, types::ConstnessEnum>
  friend class view::BaseMatrixView;

  Matrix() = default;

  Matrix(const Matrix&) = default;

  // Needs to leave other in valid state.
  Matrix(Matrix&& other) noexcept : rows_{std::exchange(other.rows_, 0)}, data_{std::exchange(other.data_, {})} {}

  // Use static cast to call private helper ctor Matrix(size_type, size_type).
  Matrix(Rows rows, Cols cols) : Matrix(static_cast<size_type>(rows), static_cast<size_type>(cols)) {}

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
  reverse_iterator rbegin() {
    return reverse_iterator(end());
  }
  const_reverse_iterator rbegin() const {
    return reverse_iterator(end());
  }
  const_reverse_iterator crbegin() const {
    return reverse_iterator(cend());
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

  // Comparison operators.

  friend bool operator==(const Matrix& lhs, const Matrix& rhs) {
    if (lhs.Rows() != rhs.Rows() || lhs.Cols() != rhs.Cols()) {
      return false;
    }

    return std::equal(lhs.cbegin(), lhs.cend(), rhs.cbegin(), rhs.cend(),
                      [](Scalar a, Scalar b) { return scalar_utils::ApproxEqual<Scalar>(a, b); });
  }

  friend bool operator!=(const Matrix& lhs, const Matrix& rhs) {
    return !(lhs == rhs);
  }

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

  // Arithmetic operators.

  // Unary operators.

  Matrix operator-() {
    Matrix result(*this);
    result.Apply(std::negate<>{});
    return result;
  }

  // Matrix <op> Matrix

  Matrix& operator+=(const Matrix& rhs) {
    return Apply(rhs, std::plus<>{});
  }

  Matrix& operator-=(const Matrix& rhs) {
    return Apply(rhs, std::minus<>{});
  }

  Matrix& operator*=(const Matrix& rhs) {
    *this = *this * rhs;
    return *this;
  }

  friend Matrix operator+(const Matrix& lhs, const Matrix& rhs) {
    Matrix result(lhs);
    result += rhs;
    return result;
  }
  friend Matrix operator+(Matrix&& lhs, const Matrix& rhs) {
    lhs += rhs;
    return lhs;
  }
  friend Matrix operator+(const Matrix& lhs, Matrix&& rhs) {
    rhs += lhs;
    return rhs;
  }
  friend Matrix operator+(Matrix&& lhs, Matrix&& rhs) {
    lhs += rhs;
    return lhs;
  }

  friend Matrix operator-(const Matrix& lhs, const Matrix& rhs) {
    Matrix result(lhs);
    result -= rhs;
    return result;
  }
  friend Matrix operator-(Matrix&& lhs, const Matrix& rhs) {
    lhs -= rhs;
    return lhs;
  }
  friend Matrix operator-(const Matrix& lhs, Matrix&& rhs) {
    rhs -= lhs;
    return rhs;
  }
  friend Matrix operator-(Matrix&& lhs, Matrix&& rhs) {
    lhs -= rhs;
    return lhs;
  }

  friend Matrix operator*(const Matrix& lhs, const Matrix& rhs) {
    assert(lhs.Cols() == rhs.Rows() && "Lhs cols should be equal to rhs rows");

    Matrix result(lhs.Rows(), rhs.Cols());
    for (size_type i = 0; i < lhs.Rows(); ++i) {
      for (size_type j = 0; j < rhs.Cols(); ++j) {
        for (size_type k = 0; k < lhs.Cols(); ++k) {
          result(i, j) += lhs(i, k) * rhs(k, j);
        }
      }
    }

    return result;
  }

  // Scalar <op> Matrix or vice versa

  Matrix& operator+=(Scalar scalar) {
    return ApplyOnDiagonal([scalar](Scalar value) { return value + scalar; });
  }

  Matrix& operator-=(Scalar scalar) {
    return ApplyOnDiagonal([scalar](Scalar value) { return value - scalar; });
  }

  Matrix& operator*=(Scalar scalar) {
    return Apply([scalar](Scalar value) { return value * scalar; });
  }

  Matrix& operator/=(Scalar scalar) {
    return Apply([scalar](Scalar value) { return value / scalar; });
  }

  friend Matrix operator+(Matrix lhs, Scalar scalar) {
    lhs += scalar;
    return lhs;
  }

  friend Matrix operator+(Scalar scalar, Matrix rhs) {
    rhs += scalar;
    return rhs;
  }

  friend Matrix operator-(Matrix lhs, Scalar scalar) {
    lhs -= scalar;
    return lhs;
  }

  friend Matrix operator-(Scalar scalar, Matrix rhs) {
    rhs -= scalar;
    return rhs;
  }

  friend Matrix operator*(Matrix lhs, Scalar scalar) {
    lhs *= scalar;
    return lhs;
  }

  friend Matrix operator*(Scalar scalar, Matrix rhs) {
    rhs *= scalar;
    return rhs;
  }

  friend Matrix operator/(Matrix lhs, Scalar scalar) {
    lhs /= scalar;
    return lhs;
  }

  // Static creation methods.

  static Matrix Identity(size_type size) {
    Matrix identity(size, size);
    identity.ApplyOnDiagonal([](Scalar) { return 1; });
    return identity;
  }

  static Matrix Zero(size_type rows, size_type cols) {
    return Matrix(rows, cols);
  }

  static Matrix SingleEntry(size_type rows, size_type cols, Index unit_row, Index unit_col) {
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
  StorageIterator RawBegin() {
    return data_.begin();
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
}  // namespace linalg

#endif
