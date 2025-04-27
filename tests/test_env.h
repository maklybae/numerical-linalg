#ifndef TEST_ENV_H
#define TEST_ENV_H

#include <gtest/gtest.h>

#include <cstdint>
#include <random>

#include "../src/matrix_types.h"       // IWYU pragma: keep
#include "../src/base_matrix_view.h"   // IWYU pragma: keep
#include "../src/matrix_view.h"        // IWYU pragma: keep
#include "../src/bidiagonalization.h"  // IWYU pragma: keep
#include "../src/core_types.h"         // IWYU pragma: keep
#include "../src/givens.h"             // IWYU pragma: keep
#include "../src/hessenberg.h"         // IWYU pragma: keep
#include "../src/householder.h"        // IWYU pragma: keep
#include "../src/qr_algorithm.h"       // IWYU pragma: keep
#include "../src/qr_decomposition.h"   // IWYU pragma: keep
#include "../src/scalar_types.h"       // IWYU pragma: keep
#include "../src/scalar_utils.h"       // IWYU pragma: keep
#include "../src/submatrix_range.h"    // IWYU pragma: keep
#include "../src/svd.h"                // IWYU pragma: keep
#include "../src/matrix.h"             // IWYU pragma: keep

namespace linalg::test {

constexpr Size kStressTestSize = 100;

class RandGenerator {
 public:
  explicit RandGenerator(unsigned int seed, int64_t from, int64_t to) : mt_(seed), dist_(from, to) {}

  template <detail::FloatingOrComplexType Scalar>
  Scalar GetRandomScalar() {
    if constexpr (std::is_floating_point_v<Scalar>) {
      return Scalar(dist_(mt_));
    } else if constexpr (detail::kIsComplexV<Scalar>) {
      using UnderlyingScalar = detail::UnderlyingScalarT<Scalar>;

      return Scalar(GetRandomScalar<UnderlyingScalar>(), GetRandomScalar<UnderlyingScalar>());
    }
  }

  template <detail::FloatingOrComplexType Scalar>
  Matrix<Scalar> GetRandomMatrix(ERows rows, ECols cols) {
    Matrix<Scalar> matrix(rows, cols);
    for (auto& value : matrix) {
      value = GetRandomScalar<Scalar>();
    }
    return matrix;
  }

  template <detail::FloatingOrComplexType Scalar>
  Matrix<Scalar> GetRandomMatrix() {
    return GetRandomMatrix<Scalar>(ERows{size_dist(mt_)}, ECols{size_dist(mt_)});
  }

  template <detail::FloatingOrComplexType Scalar>
  Matrix<Scalar> GetRandomSquareMatrix() {
    auto size = size_dist(mt_);
    return GetRandomMatrix<Scalar>(ERows{size}, ECols{size});
  }

  template <detail::FloatingOrComplexType Scalar>
  Matrix<Scalar> GetRandomHermitianMatrix() {
    auto matrix = GetRandomSquareMatrix<Scalar>();
    for (Size i = 0; i < matrix.Rows(); ++i) {
      for (Size j = 0; j < i; ++j) {
        if constexpr (detail::kIsComplexV<Scalar>) {
          matrix(i, j) = std::conj(matrix(j, i));
        } else {
          matrix(i, j) = matrix(j, i);
        }
      }
    }
    return matrix;
  }

  template <detail::FloatingOrComplexType Scalar>
  Matrix<Scalar> GetRandomBidiagonalMatrix() {
    auto matrix = GetRandomSquareMatrix<Scalar>();
    for (Size i = 0; i < matrix.Rows(); ++i) {
      for (Size j = 0; j < matrix.Cols(); ++j) {
        if (i != j && i + 1 != j) {
          matrix(i, j) = 0;
        }
      }
    }

    return matrix;
  }

  template <detail::FloatingOrComplexType Scalar>
  Matrix<Scalar> GetRandomTallMatrix() {
    auto rows = size_dist(mt_);
    auto cols = size_dist(mt_);
    if (rows < cols) {
      std::swap(rows, cols);
    }
    return GetRandomMatrix<Scalar>(ERows{rows}, ECols{cols});
  }

 private:
  std::mt19937 mt_;
  std::uniform_int_distribution<int64_t> dist_;
  static std::uniform_int_distribution<Size> size_dist;
};

extern RandGenerator rand_generator;
extern Matrix<double> matrix_double;
extern Matrix<long double> matrix_long_double;
extern Matrix<double> sym_matrix_double;
extern Matrix<long double> sym_matrix_long_double;
extern Matrix<double> bidiagonal_matrix_double;
extern Matrix<long double> bidiagonal_matrix_long_double;
extern Matrix<std::complex<double>> matrix_complex_double;
extern Matrix<std::complex<long double>> matrix_complex_long_double;
extern Matrix<std::complex<double>> hermitian_matrix_complex_double;
extern Matrix<std::complex<long double>> hermitian_matrix_complex_long_double;
extern Matrix<std::complex<double>> bidiagonal_matrix_complex_double;
extern Matrix<std::complex<long double>> bidiagonal_matrix_complex_long_double;

}  // namespace linalg::test

#endif
