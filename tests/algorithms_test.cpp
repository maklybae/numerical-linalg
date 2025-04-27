#include <gtest/gtest.h>

#include "test_env.h"

namespace linalg::test {

template <detail::FloatingOrComplexType Scalar>
void ExpectQRDecomposition(const QRDecompositionResult<Scalar>& qr_res, const Matrix<Scalar>& matrix) {
  EXPECT_EQ(matrix.Rows(), qr_res.r.Rows());
  EXPECT_EQ(matrix.Cols(), qr_res.r.Cols());
  EXPECT_EQ(matrix.Rows(), qr_res.q.Rows());
  EXPECT_EQ(matrix.Rows(), qr_res.q.Cols());

  EXPECT_TRUE(IsUnitary(qr_res.q));
  EXPECT_TRUE(IsUpperTriangular(qr_res.r));

  auto qr = qr_res.q * qr_res.r;
  EXPECT_EQ(matrix, qr);
}

template <detail::FloatingOrComplexType Scalar>
void ExpectSpectralDecomposition(const EigenDecompositionResult<Scalar>& eig_res, const Matrix<Scalar>& matrix) {
  EXPECT_EQ(matrix.Rows(), eig_res.q.Rows());
  EXPECT_EQ(matrix.Cols(), eig_res.q.Cols());
  EXPECT_EQ(matrix.Rows(), eig_res.d.Rows());
  EXPECT_EQ(matrix.Cols(), eig_res.d.Cols());

  EXPECT_TRUE(IsUnitary(eig_res.q));

  auto prod = eig_res.q * eig_res.d * Conjugated(eig_res.q);
  EXPECT_TRUE(detail::ApproxEqual(matrix, prod, 1e-4));
}

template <detail::FloatingOrComplexType Scalar>
void ExpectQRAlgorithmImplicitShifted(const detail::ImplicitShiftedQRAlgorithmResult<Scalar>& qr_res,
                                      const Matrix<Scalar>& matrix) {
  EXPECT_TRUE(IsUnitary(qr_res.u));
  EXPECT_TRUE(IsUnitary(qr_res.vt));

  auto prod = qr_res.u * qr_res.d * qr_res.vt;
  EXPECT_TRUE(detail::ApproxEqual(matrix, prod));
}

template <detail::FloatingOrComplexType Scalar>
void ExpectSVD(const SVDResult<Scalar>& svd_res, const Matrix<Scalar>& matrix) {
  EXPECT_TRUE(IsUnitary(svd_res.u, 1e-8));
  EXPECT_TRUE(IsUnitary(svd_res.vt, 1e-8));

  auto prod = svd_res.u * svd_res.s * svd_res.vt;
  EXPECT_TRUE(detail::ApproxEqual(matrix, prod, 1e-8));
}

template <detail::FloatingOrComplexType Scalar>
void ExpectBidiagonal(const BidiagonalizationResult<Scalar>& bidiag_res, const Matrix<Scalar>& matrix) {
  EXPECT_TRUE(IsUnitary(bidiag_res.u));
  EXPECT_TRUE(IsUnitary(bidiag_res.vt));
  EXPECT_TRUE(IsBidiagonal(bidiag_res.b));

  auto prod = bidiag_res.u * bidiag_res.b * bidiag_res.vt;
  EXPECT_TRUE(detail::ApproxEqual(matrix, prod, 1e-8));
}

template <detail::FloatingOrComplexType Scalar>
void ExpectHessenberg(const HessenbergFormResult<Scalar>& hessenberg_res, const Matrix<Scalar>& matrix) {
  EXPECT_TRUE(IsUnitary(hessenberg_res.q));
  EXPECT_TRUE(IsHessenberg(hessenberg_res.h));

  auto prod = hessenberg_res.q * hessenberg_res.h * Conjugated(hessenberg_res.q);
  EXPECT_TRUE(detail::ApproxEqual(matrix, prod, 1e-8));
}

TEST(QRDecomposition, Floating) {
  ExpectQRDecomposition(QRDecomposition(matrix_double), matrix_double);
  ExpectQRDecomposition(QRDecomposition(matrix_long_double), matrix_long_double);
}

TEST(QRDecomposition, Complex) {
  ExpectQRDecomposition(QRDecomposition(matrix_complex_double), matrix_complex_double);
  ExpectQRDecomposition(QRDecomposition(matrix_complex_long_double), matrix_complex_long_double);
}

TEST(QRDecomposition, RandomFloating) {
  for (Size i = 0; i < kStressTestSize; ++i) {
    auto matrix = rand_generator.GetRandomMatrix<long double>();
    ExpectQRDecomposition(QRDecomposition(matrix), matrix);
  }
}

TEST(QRDecomposition, RandomComplex) {
  for (Size i = 0; i < kStressTestSize; ++i) {
    auto matrix = rand_generator.GetRandomMatrix<std::complex<long double>>();
    ExpectQRDecomposition(QRDecomposition(matrix), matrix);
  }
}

TEST(QRDecompositionGivens, Floating) {
  ExpectQRDecomposition(GivensQRDecomposition(matrix_double), matrix_double);
  ExpectQRDecomposition(GivensQRDecomposition(matrix_long_double), matrix_long_double);
}

TEST(QRDecompositionGivens, Complex) {
  ExpectQRDecomposition(GivensQRDecomposition(matrix_complex_double), matrix_complex_double);
  ExpectQRDecomposition(GivensQRDecomposition(matrix_complex_long_double), matrix_complex_long_double);
}

TEST(QRDecompositionGivens, RandomFloating) {
  for (Size i = 0; i < kStressTestSize; ++i) {
    auto matrix = rand_generator.GetRandomMatrix<long double>();
    ExpectQRDecomposition(GivensQRDecomposition(matrix), matrix);
  }
}

TEST(QRDecompositionGivens, RandomComplex) {
  for (Size i = 0; i < kStressTestSize; ++i) {
    auto matrix = rand_generator.GetRandomMatrix<std::complex<long double>>();
    ExpectQRDecomposition(GivensQRDecomposition(matrix), matrix);
  }
}

TEST(QRAlgorithm, Floating) {
  ExpectSpectralDecomposition(SymEigenDecomposition(sym_matrix_double), sym_matrix_double);
  ExpectSpectralDecomposition(SymEigenDecomposition(sym_matrix_long_double), sym_matrix_long_double);
}

TEST(QRAlgorithm, Complex) {
  ExpectSpectralDecomposition(SymEigenDecomposition(hermitian_matrix_complex_double), hermitian_matrix_complex_double);
  ExpectSpectralDecomposition(SymEigenDecomposition(hermitian_matrix_complex_long_double),
                              hermitian_matrix_complex_long_double);
}

// TEST(QRAlgorithm, RandomFloating) {
//   for (Size i = 0; i < 10; ++i) {
//     auto matrix = rand_generator.GetRandomHermitianMatrix<long double>();
//     ExpectSpectralDecomposition(SymEigenDecomposition(matrix), matrix);
//   }
// }

// TEST(QRAlgorithm, RandomComplex) {
//   for (Size i = 0; i < 10; ++i) {
//     auto matrix = rand_generator.GetRandomHermitianMatrix<std::complex<long double>>();
//     ExpectSpectralDecomposition(SymEigenDecomposition(matrix), matrix);
//   }
// }

TEST(QRAlgorithmImplicitShifted, Floating) {
  ExpectQRAlgorithmImplicitShifted(detail::ImplicitShiftedQRAlgorithm(bidiagonal_matrix_double),
                                   bidiagonal_matrix_double);
  ExpectQRAlgorithmImplicitShifted(detail::ImplicitShiftedQRAlgorithm(bidiagonal_matrix_long_double),
                                   bidiagonal_matrix_long_double);
}

TEST(QRAlgorithmImplicitShifted, RandomFloating) {
  for (Size i = 0; i < kStressTestSize; ++i) {
    auto matrix = rand_generator.GetRandomBidiagonalMatrix<long double>();
    ExpectQRAlgorithmImplicitShifted(detail::ImplicitShiftedQRAlgorithm(matrix), matrix);
  }
}

TEST(SVD, Floating) {
  ExpectSVD(SVD(matrix_double), matrix_double);
  ExpectSVD(SVD(matrix_long_double), matrix_long_double);
}

TEST(SVD, Complex) {
  ExpectSVD(SVD(matrix_complex_double), matrix_complex_double);
  ExpectSVD(SVD(matrix_complex_long_double), matrix_complex_long_double);
}

TEST(SVD, RandomFloating) {
  for (Size i = 0; i < kStressTestSize; ++i) {
    auto matrix = rand_generator.GetRandomMatrix<long double>();
    ExpectSVD(SVD(matrix), matrix);
  }
}

TEST(SVD, RandomComplex) {
  for (Size i = 0; i < kStressTestSize; ++i) {
    auto matrix = rand_generator.GetRandomMatrix<std::complex<long double>>();
    ExpectSVD(SVD(matrix), matrix);
  }
}

TEST(Bidiagonalization, Floating) {
  ExpectBidiagonal(GetBidiagonalization(matrix_double), matrix_double);
  ExpectBidiagonal(GetBidiagonalization(matrix_long_double), matrix_long_double);
}

TEST(Bidiagonalization, Complex) {
  ExpectBidiagonal(GetBidiagonalization(matrix_complex_double), matrix_complex_double);
  ExpectBidiagonal(GetBidiagonalization(matrix_complex_long_double), matrix_complex_long_double);
}

TEST(Bidiagonalization, RandomFloating) {
  for (Size i = 0; i < kStressTestSize; ++i) {
    auto matrix = rand_generator.GetRandomTallMatrix<long double>();
    ExpectBidiagonal(GetBidiagonalization(matrix), matrix);
  }
}

TEST(Bidiagonalization, RandomComplex) {
  for (Size i = 0; i < kStressTestSize; ++i) {
    auto matrix = rand_generator.GetRandomTallMatrix<std::complex<long double>>();
    ExpectBidiagonal(GetBidiagonalization(matrix), matrix);
  }
}

TEST(HessenbergForm, Floating) {
  // Test on symmetric matrix (random on any square matrix)
  ExpectHessenberg(GetHessenbergForm(sym_matrix_double), sym_matrix_double);
  ExpectHessenberg(GetHessenbergForm(sym_matrix_long_double), sym_matrix_long_double);
}

TEST(HessenbergForm, Complex) {
  // Test on hermitian matrix (random on any square matrix)
  ExpectHessenberg(GetHessenbergForm(hermitian_matrix_complex_double), hermitian_matrix_complex_double);
  ExpectHessenberg(GetHessenbergForm(hermitian_matrix_complex_long_double), hermitian_matrix_complex_long_double);
}

TEST(HessenbergForm, RandomFloating) {
  for (Size i = 0; i < kStressTestSize; ++i) {
    auto matrix = rand_generator.GetRandomSquareMatrix<long double>();
    ExpectHessenberg(GetHessenbergForm(matrix), matrix);
  }
}

TEST(HessenbergForm, RandomComplex) {
  for (Size i = 0; i < kStressTestSize; ++i) {
    auto matrix = rand_generator.GetRandomSquareMatrix<std::complex<long double>>();
    ExpectHessenberg(GetHessenbergForm(matrix), matrix);
  }
}

TEST(GivensRotation, Floating) {
  for (Size i = 0; i < kStressTestSize; ++i) {
    auto matrix = rand_generator.GetRandomMatrix<long double>();
    auto params = GetZeroingGivensRotationParams(matrix(0, 0), matrix(1, 0));
    ApplyGivensRotationLeft(matrix, params, 0, 1);
    EXPECT_TRUE(detail::ApproxZero(matrix(1, 0)));
  }
}

TEST(GivensRotation, Complex) {
  for (Size i = 0; i < kStressTestSize; ++i) {
    auto matrix = rand_generator.GetRandomMatrix<std::complex<long double>>();
    auto params = GetZeroingGivensRotationParams(matrix(0, 0), matrix(1, 0));
    ApplyGivensRotationLeft(matrix, params, 0, 1);
    EXPECT_TRUE(detail::ApproxZero(matrix(1, 0)));
  }
}

TEST(HouseholderReflection, Floating) {
  for (Size i = 0; i < kStressTestSize; ++i) {
    auto matrix = rand_generator.GetRandomMatrix<long double>();

    auto reflect_vector = Matrix<long double>{matrix.Col(0)};
    HouseholderReflectVectorReduce(reflect_vector);

    ApplyHouseholderLeft(matrix, reflect_vector);
    auto zero_submatrix =
        matrix.Submatrix(SubmatrixRange::FromBeginEnd(ERowBegin{1}, ERowEnd{matrix.Rows()}, EColBegin{0}, EColEnd{1}));
    EXPECT_TRUE(IsZero(zero_submatrix));
  }
}

TEST(HouseholderReflection, Complex) {
  for (Size i = 0; i < kStressTestSize; ++i) {
    auto matrix = rand_generator.GetRandomMatrix<std::complex<long double>>();

    auto reflect_vector = Matrix<std::complex<long double>>{matrix.Col(0)};
    HouseholderReflectVectorReduce(reflect_vector);

    ApplyHouseholderLeft(matrix, reflect_vector);
    auto zero_submatrix =
        matrix.Submatrix(SubmatrixRange::FromBeginEnd(ERowBegin{1}, ERowEnd{matrix.Rows()}, EColBegin{0}, EColEnd{1}));
    EXPECT_TRUE(IsZero(zero_submatrix));
  }
}

}  // namespace linalg::test
