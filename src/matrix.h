#ifndef MATRIX_H
#define MATRIX_H

#include <cassert>
#include <complex>
#include <concepts>
#include <initializer_list>
#include <iterator>
#include <type_traits>
#include <utility>
#include <vector>

namespace linalg::types {

template <typename T>
struct IsComplex : std::false_type {};

template <std::floating_point T>
struct IsComplex<std::complex<T>> : std::true_type {};

template <typename T>
constexpr bool kIsComplexV = IsComplex<T>::value;

template <typename T>
concept FloatingOrComplexType = std::is_floating_point_v<T> || kIsComplexV<T>;

}  // namespace linalg::types

namespace linalg {
template <types::FloatingOrComplexType Scalar>
class Matrix {
 public:
  using StorageType = std::vector<Scalar>;

  // NOLINTBEGIN(readability-identifier-naming)
  using value_type             = Scalar;
  using reference              = Scalar&;
  using const_reference        = const Scalar&;
  using iterator               = StorageType::iterator;
  using const_iterator         = StorageType::const_iterator;
  using reverse_iterator       = StorageType::reverse_iterator;
  using const_reverse_iterator = StorageType::const_reverse_iterator;
  using difference_type        = StorageType::difference_type;
  using size_type              = StorageType::size_type;
  // NOLINTEND(readability-identifier-naming)

  Matrix() = default;

  Matrix(const Matrix&) = default;

  // Needs to leave other in valid state.
  Matrix(Matrix&& other) noexcept
      : rows_{std::exchange(other.rows_, 0)}
      , cols_{std::exchange(other.cols_, 0)}
      , data_{std::exchange(other.data_, {})} {}

  Matrix(size_type rows, size_type cols) : rows_{rows}, cols_{cols}, data_(rows * cols) {}

  Matrix(std::initializer_list<std::initializer_list<Scalar>> list) {
    rows_ = list.size();
    cols_ = list.begin()->size();
    data_.reserve(rows_ * cols_);

    for (auto row : list) {
      assert(row.size() == cols_ && "All rows should have the same size");
      data_.insert(data_.end(), row);
    }
  }

  Matrix& operator=(const Matrix&) = default;

  // Needs to leave other in valid state.
  Matrix& operator=(Matrix&& other) noexcept {
    Matrix tmp = std::move(other);
    std::swap(rows_, tmp.rows_);
    std::swap(cols_, tmp.cols_);
    std::swap(data_, tmp.data_);

    assert(other.rows_ == 0 && "Rows of moved matrix should be equal 0");
    assert(other.cols_ == 0 && "Cols of moved matrix should be equal 0");
    assert(other.data_.empty() && "Data of moved matrix should be equal 0");
    return *this;
  }

  // NOLINTNEXTLINE(readability-identifier-naming)
  void swap(Matrix& other) {
    std::swap(rows_, other.rows_);
    std::swap(cols_, other.cols_);
    std::swap(data_, other.data_);
  }

  ~Matrix() = default;

  // Size and data access methods.

  size_type Rows() const {
    return rows_;
  }

  size_type Cols() const {
    return cols_;
  }

  // NOLINTBEGIN(readability-identifier-naming)
  size_type size() const {
    return data_.size();
  }
  size_type max_size() const {
    return data_.max_size();
  }
  bool empty() const {
    return data_.empty();
  }
  // NOLINTEND(readability-identifier-naming)

  Scalar& operator()(size_type row, size_type col) {
    assert(row < rows_ && "Row index out of bounds");
    assert(col < cols_ && "Col index out of bounds");
    return data_[row * cols_ + col];
  }

  Scalar operator()(size_type row, size_type col) const {
    assert(row < rows_ && "Row index out of bounds");
    assert(col < cols_ && "Col index out of bounds");
    return data_[row * cols_ + col];
  }

  // Iterators.

  // NOLINTBEGIN(readability-identifier-naming)
  iterator begin() {
    return data_.begin();
  }
  const_iterator begin() const {
    return data_.begin();
  }
  const_iterator cbegin() const {
    return data_.cbegin();
  }
  reverse_iterator rbegin() {
    return data_.rbegin();
  }
  const_reverse_iterator rbegin() const {
    return data_.rbegin();
  }
  const_reverse_iterator crbegin() const {
    return data_.crbegin();
  }

  iterator end() {
    return data_.end();
  }
  const_iterator end() const {
    return data_.end();
  }
  const_iterator cend() const {
    return data_.cend();
  }
  reverse_iterator rend() {
    return data_.rend();
  }
  const_reverse_iterator rend() const {
    return data_.rend();
  }
  const_reverse_iterator crend() const {
    return data_.crend();
  }
  // NOLINTEND(readability-identifier-naming)

  // Comparison operators.

  friend bool operator==(const Matrix& lhs, const Matrix& rhs) = default;
  friend bool operator!=(const Matrix& lhs, const Matrix& rhs) = default;

  // Static creation methods.

  static Matrix Identity(size_type size) {
    Matrix identity(size, size);
    for (size_type i = 0; i < size; ++i) {
      identity(i, i) = 1;
    }
    return identity;
  }

  static Matrix Zero(size_type rows, size_type cols) {
    return Matrix(rows, cols);
  }

  static Matrix Unit(size_type rows, size_type cols, size_type unit_row, size_type unit_col) {
    Matrix unit(rows, cols);
    unit(unit_row, unit_col) = 1;
    return unit;
  }

  static Matrix Diagonal(size_type size, Scalar value) {
    Matrix diagonal(size, size);
    for (size_type i = 0; i < size; ++i) {
      diagonal(i, i) = value;
    }
    return diagonal;
  }

  static Matrix Diagonal(size_type size, std::initializer_list<Scalar> list) {
    Matrix diagonal(size, size);
    size_type i{0};
    for (auto value : list) {
      diagonal(i, i) = value;
      ++i;
    }
    return diagonal;
  }

  template <std::input_iterator InputIterator>
  static Matrix Diagonal(InputIterator first, InputIterator last) {
    static_assert(std::is_same_v<typename std::iterator_traits<InputIterator>::value_type, Scalar>,
                  "Iterator value type should be the same as matrix scalar type");
    auto ssize{std::distance(first, last)};

    assert(ssize >= 0 && "Iterator range should be non-negative");
    size_type size{static_cast<size_type>(ssize)};

    Matrix diagonal(size, size);
    for (size_type i = 0; first != last; ++first, ++i) {
      diagonal(i, i) = *first;
    }
    return diagonal;
  }

 private:
  size_type rows_{};
  size_type cols_{};
  StorageType data_{};
};
}  // namespace linalg

#endif
