#ifndef MATRIX_H
#define MATRIX_H

#include <complex>
#include <concepts>
#include <type_traits>
#include <utility>
#include <vector>

namespace linalg {

template <typename T>
struct IsComplex : std::false_type {};

template <std::floating_point T>
struct IsComplex<std::complex<T>> : std::true_type {};

template <typename T>
constexpr bool kIsComplexV = IsComplex<T>::value;

template <typename T>
concept FloatingOrComplexType = std::is_floating_point_v<T> || kIsComplexV<T>;

template <FloatingOrComplexType Scalar>
class Matrix {
 private:
  using Container = std::vector<Scalar>;

 public:
  using SizeT = std::size_t;

  Matrix() = default;

  Matrix(const Matrix&) = default;

  // Needs to leave other in valid state.
  Matrix(Matrix&& other) noexcept
      : rows_{std::exchange(other.rows_, 0)}
      , cols_{std::exchange(other.cols_, 0)}
      , data_{std::exchange(other.data_, {})} {}

  Matrix& operator=(const Matrix&) = default;

  // Needs to leave other in valid state.
  Matrix& operator=(Matrix&& other) noexcept {
    Matrix tmp = std::move(other);
    std::swap(rows_, other.rows_);
    std::swap(cols_, other.cols_);
    std::swap(data_, other.data_);
    return *this;
  }

  ~Matrix() = default;

 private:
  SizeT rows_{};
  SizeT cols_{};
  Container data_{};
};
}  // namespace linalg

#endif
