#ifndef MATRIX_H
#define MATRIX_H

#include <cassert>
#include <complex>
#include <concepts>
#include <initializer_list>
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
 private:
  using Container = std::vector<Scalar>;

 public:
  using SizeT = Container::size_type;

  Matrix() = default;

  Matrix(const Matrix&) = default;

  // Needs to leave other in valid state.
  Matrix(Matrix&& other) noexcept
      : rows_{std::exchange(other.rows_, 0)}
      , cols_{std::exchange(other.cols_, 0)}
      , data_{std::exchange(other.data_, {})} {}

  Matrix(SizeT rows, SizeT cols) : rows_{rows}, cols_{cols}, data_(rows * cols) {}

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

  ~Matrix() = default;

  SizeT Rows() const {
    return rows_;
  }

  SizeT Cols() const {
    return cols_;
  }

  bool Empty() const {
    return data_.empty();
  }

  Scalar& operator()(SizeT row, SizeT col) {
    assert(row < rows_ && "Row index out of bounds");
    assert(col < cols_ && "Col index out of bounds");
    return data_[row * cols_ + col];
  }

  const Scalar& operator()(SizeT row, SizeT col) const {
    assert(row < rows_ && "Row index out of bounds");
    assert(col < cols_ && "Col index out of bounds");
    return data_[row * cols_ + col];
  }

 private:
  SizeT rows_{};
  SizeT cols_{};
  Container data_{};
};
}  // namespace linalg

#endif
