#include "../src/matrix.h"

#include <gtest/gtest.h>

#include <complex>
#include <iomanip>
#include <ios>
#include <type_traits>

#include "test_env.h"  // IWYU pragma: keep

namespace linalg::test {

using linalg::Matrix;
using linalg::Size;

TEST(MatrixCtors, DefaultCtor) {
  Matrix<double> matrix;
  EXPECT_EQ(matrix.Rows(), 0);
  EXPECT_EQ(matrix.Cols(), 0);
}

TEST(MatrixCtors, CopyCtor_EmptyMatrix) {
  Matrix<double> matrix;
  Matrix<double> copy(matrix);
  EXPECT_EQ(copy.Rows(), 0);
  EXPECT_EQ(copy.Cols(), 0);
}

TEST(MatrixCtors, MoveCtor_EmptyMatrix) {
  Matrix<double> matrix;
  Matrix moved(std::move(matrix));
  EXPECT_EQ(moved.Rows(), 0);
  EXPECT_EQ(moved.Cols(), 0);
}

TEST(MatrixCtors, SizeCtor) {
  Matrix<double> matrix(linalg::ERows{2}, linalg::ECols{3});
  EXPECT_EQ(matrix.Rows(), 2);
  EXPECT_EQ(matrix.Cols(), 3);
}

TEST(MatrixCtors, InitializerListCtor) {
  Matrix<double> matrix{{1, 2, 3}, {4, 5, 6}};
  EXPECT_EQ(matrix.Rows(), 2);
  EXPECT_EQ(matrix.Cols(), 3);

  EXPECT_DOUBLE_EQ(matrix(0, 0), 1.0);
  EXPECT_DOUBLE_EQ(matrix(0, 1), 2.0);
  EXPECT_DOUBLE_EQ(matrix(0, 2), 3.0);
  EXPECT_DOUBLE_EQ(matrix(1, 0), 4.0);
  EXPECT_DOUBLE_EQ(matrix(1, 1), 5.0);
  EXPECT_DOUBLE_EQ(matrix(1, 2), 6.0);
}

TEST(MatrixCtors, CopyConstructor) {
  Matrix<double> matrix{{1, 2, 3}, {4, 5, 6}};
  Matrix copy(matrix);
  EXPECT_EQ(copy.Rows(), 2);
  EXPECT_EQ(copy.Cols(), 3);

  EXPECT_DOUBLE_EQ(copy(0, 0), 1.0);
  EXPECT_DOUBLE_EQ(copy(0, 1), 2.0);
  EXPECT_DOUBLE_EQ(copy(0, 2), 3.0);
  EXPECT_DOUBLE_EQ(copy(1, 0), 4.0);
  EXPECT_DOUBLE_EQ(copy(1, 1), 5.0);
  EXPECT_DOUBLE_EQ(copy(1, 2), 6.0);

  // No side effects on the original matrix.
  EXPECT_EQ(matrix.Rows(), 2);
  EXPECT_EQ(matrix.Cols(), 3);

  EXPECT_DOUBLE_EQ(matrix(0, 0), 1.0);
  EXPECT_DOUBLE_EQ(matrix(0, 1), 2.0);
  EXPECT_DOUBLE_EQ(matrix(0, 2), 3.0);
  EXPECT_DOUBLE_EQ(matrix(1, 0), 4.0);
  EXPECT_DOUBLE_EQ(matrix(1, 1), 5.0);
  EXPECT_DOUBLE_EQ(matrix(1, 2), 6.0);
}

TEST(MatrixCtors, MoveCtor) {
  Matrix<double> matrix{{1, 2, 3}, {4, 5, 6}};
  Matrix moved(std::move(matrix));
  EXPECT_EQ(moved.Rows(), 2);
  EXPECT_EQ(moved.Cols(), 3);

  EXPECT_DOUBLE_EQ(moved(0, 0), 1.0);
  EXPECT_DOUBLE_EQ(moved(0, 1), 2.0);
  EXPECT_DOUBLE_EQ(moved(0, 2), 3.0);
  EXPECT_DOUBLE_EQ(moved(1, 0), 4.0);
  EXPECT_DOUBLE_EQ(moved(1, 1), 5.0);
  EXPECT_DOUBLE_EQ(moved(1, 2), 6.0);

  // Data from matrix should be moved.
  EXPECT_EQ(matrix.Rows(), 0);
  EXPECT_EQ(matrix.Cols(), 0);
}

TEST(MatrixAssignment, CopyAssignment) {
  Matrix<double> matrix{{1, 2, 3}, {4, 5, 6}};
  Matrix<double> copy;
  copy = matrix;
  EXPECT_EQ(copy.Rows(), 2);
  EXPECT_EQ(copy.Cols(), 3);

  EXPECT_DOUBLE_EQ(copy(0, 0), 1.0);
  EXPECT_DOUBLE_EQ(copy(0, 1), 2.0);
  EXPECT_DOUBLE_EQ(copy(0, 2), 3.0);
  EXPECT_DOUBLE_EQ(copy(1, 0), 4.0);
  EXPECT_DOUBLE_EQ(copy(1, 1), 5.0);
  EXPECT_DOUBLE_EQ(copy(1, 2), 6.0);

  // No side effects on the original matrix.
  EXPECT_EQ(matrix.Rows(), 2);
  EXPECT_EQ(matrix.Cols(), 3);

  EXPECT_DOUBLE_EQ(matrix(0, 0), 1.0);
  EXPECT_DOUBLE_EQ(matrix(0, 1), 2.0);
  EXPECT_DOUBLE_EQ(matrix(0, 2), 3.0);
  EXPECT_DOUBLE_EQ(matrix(1, 0), 4.0);
  EXPECT_DOUBLE_EQ(matrix(1, 1), 5.0);
  EXPECT_DOUBLE_EQ(matrix(1, 2), 6.0);
}

TEST(MatrixAssignment, MoveAssignment) {
  Matrix<double> matrix{{1, 2, 3}, {4, 5, 6}};
  Matrix<double> moved;
  moved = std::move(matrix);
  EXPECT_EQ(moved.Rows(), 2);
  EXPECT_EQ(moved.Cols(), 3);

  EXPECT_DOUBLE_EQ(moved(0, 0), 1.0);
  EXPECT_DOUBLE_EQ(moved(0, 1), 2.0);
  EXPECT_DOUBLE_EQ(moved(0, 2), 3.0);
  EXPECT_DOUBLE_EQ(moved(1, 0), 4.0);
  EXPECT_DOUBLE_EQ(moved(1, 1), 5.0);
  EXPECT_DOUBLE_EQ(moved(1, 2), 6.0);

  // Data from matrix should be moved.
  EXPECT_EQ(matrix.Rows(), 0);
  EXPECT_EQ(matrix.Cols(), 0);
}

TEST(MatrixStaticCreate, Identity) {
  Matrix<double> identity = Matrix<double>::Identity(3);
  EXPECT_EQ(identity.Rows(), 3);
  EXPECT_EQ(identity.Cols(), 3);

  for (Size i = 0; i < 3; ++i) {
    for (Size j = 0; j < 3; ++j) {
      if (i == j) {
        EXPECT_DOUBLE_EQ(identity(i, j), 1.0);
      } else {
        EXPECT_DOUBLE_EQ(identity(i, j), 0.0);
      }
    }
  }
}

TEST(MatrixStaticCreate, Zero) {
  Matrix<double> zero = Matrix<double>::Zero(linalg::ERows{2}, linalg::ECols{3});
  EXPECT_EQ(zero.Rows(), 2);
  EXPECT_EQ(zero.Cols(), 3);

  for (Size i = 0; i < 2; ++i) {
    for (Size j = 0; j < 3; ++j) {
      EXPECT_DOUBLE_EQ(zero(i, j), 0.0);
    }
  }
}

TEST(MatrixStaticCreate, Unit) {
  Matrix<double> unit =
      Matrix<double>::SingleEntry(linalg::ERows{2}, linalg::ECols{3}, linalg::ERow{1}, linalg::ECol{2});
  EXPECT_EQ(unit.Rows(), 2);
  EXPECT_EQ(unit.Cols(), 3);

  for (Size i = 0; i < 2; ++i) {
    for (Size j = 0; j < 3; ++j) {
      if (i == 1 && j == 2) {
        EXPECT_DOUBLE_EQ(unit(i, j), 1.0);
      } else {
        EXPECT_DOUBLE_EQ(unit(i, j), 0.0);
      }
    }
  }
}

TEST(MatrixStaticCreate, DiagonalScalar) {
  Matrix<double> diagonal = Matrix<double>::ScalarMatrix(3, 2.0);
  EXPECT_EQ(diagonal.Rows(), 3);
  EXPECT_EQ(diagonal.Cols(), 3);

  for (Size i = 0; i < 3; ++i) {
    for (Size j = 0; j < 3; ++j) {
      if (i == j) {
        EXPECT_DOUBLE_EQ(diagonal(i, j), 2.0);
      } else {
        EXPECT_DOUBLE_EQ(diagonal(i, j), 0.0);
      }
    }
  }
}

TEST(MatrixStaticCreate, DiagonalList) {
  Matrix<double> diagonal = Matrix<double>::Diagonal({1.0, 2.0, 3.0});
  EXPECT_EQ(diagonal.Rows(), 3);
  EXPECT_EQ(diagonal.Cols(), 3);

  for (Size i = 0; i < 3; ++i) {
    for (Size j = 0; j < 3; ++j) {
      if (i == j) {
        EXPECT_DOUBLE_EQ(diagonal(i, j), static_cast<double>(i) + 1);
      } else {
        EXPECT_DOUBLE_EQ(diagonal(i, j), 0.0);
      }
    }
  }
}

TEST(MatrixStaticCreate, DiagonalIterator) {
  std::vector<double> values{1.0, 2.0, 3.0};
  Matrix<double> diagonal = Matrix<double>::Diagonal(values.begin(), values.end());
  EXPECT_EQ(diagonal.Rows(), 3);
  EXPECT_EQ(diagonal.Cols(), 3);

  for (Size i = 0; i < 3; ++i) {
    for (Size j = 0; j < 3; ++j) {
      if (i == j) {
        EXPECT_DOUBLE_EQ(diagonal(i, j), values[static_cast<decltype(values)::size_type>(i)]);
      } else {
        EXPECT_DOUBLE_EQ(diagonal(i, j), 0.0);
      }
    }
  }
}

TEST(Debug, Debug) {
  EXPECT_EQ(1, 1);
}

TEST(MatrixCompare, ExactEqual) {
  Matrix<double> lhs{{1, 2, 3}, {4, 5, 6}};
  Matrix<double> rhs{{1, 2, 3}, {4, 5, 6}};
  EXPECT_EQ(lhs, rhs);
}

TEST(MatrixCompare, ApproxEqual) {
  Matrix<double> lhs{{1. + std::numeric_limits<double>::epsilon() / 2, 2, 3}, {4, 5, 6}};
  Matrix<double> rhs{{1, 2. - std::numeric_limits<double>::epsilon() / 2, 3}, {4, 5, 6}};

  EXPECT_TRUE(lhs == rhs);
}

TEST(MatrixCompare, ExactNotEqual) {
  Matrix<double> lhs{{1, 2, 3}, {4, 5, 6}};
  Matrix<double> rhs{{1, 2, 3}, {4, 5, 7}};

  EXPECT_TRUE(lhs != rhs);
}

TEST(MatrixCompare, ApproxNotEqual) {
  Matrix<double> lhs{{1. + std::numeric_limits<double>::epsilon() / 2, 2, 3}, {4, 5, 6}};
  Matrix<double> rhs{{1, 2. - std::numeric_limits<double>::epsilon() / 2, 3}, {4, 5, 7}};

  EXPECT_TRUE(lhs != rhs);
}

TEST(MatrixArithmetic, Negate) {
  Matrix<double> matrix{{1, 2, 3}, {4, 5, 6}};
  Matrix<double> negated = -matrix;
  EXPECT_EQ(negated.Rows(), 2);
  EXPECT_EQ(negated.Cols(), 3);

  for (Size i = 0; i < 2; ++i) {
    for (Size j = 0; j < 3; ++j) {
      EXPECT_DOUBLE_EQ(negated(i, j), -matrix(i, j));
    }
  }
}

TEST(MatrixArithmetic, Addition) {
  Matrix<double> lhs{{1, 2, 3}, {4, 5, 6}};
  Matrix<double> rhs{{7, 8, 9}, {10, 11, 12}};
  Matrix<double> sum = lhs + rhs;
  EXPECT_EQ(sum.Rows(), 2);
  EXPECT_EQ(sum.Cols(), 3);

  for (Size i = 0; i < 2; ++i) {
    for (Size j = 0; j < 3; ++j) {
      EXPECT_DOUBLE_EQ(sum(i, j), lhs(i, j) + rhs(i, j));
    }
  }
}

TEST(MatrixArithmetic, InPlaceSubtraction) {
  Matrix<double> lhs{{1, 2, 3}, {4, 5, 6}};
  Matrix<double> rhs{{7, 8, 9}, {10, 11, 12}};
  Matrix<double> diff = lhs - rhs;
  EXPECT_EQ(diff.Rows(), 2);
  EXPECT_EQ(diff.Cols(), 3);

  for (Size i = 0; i < 2; ++i) {
    for (Size j = 0; j < 3; ++j) {
      EXPECT_DOUBLE_EQ(diff(i, j), lhs(i, j) - rhs(i, j));
    }
  }
}

TEST(MatrixArithmetic, InPlaceMultiplication) {
  Matrix<double> lhs{{1, 2, 3}, {4, 5, 6}};
  Matrix<double> rhs{{7, 8}, {9, 10}, {11, 12}};
  Matrix<double> expected{{58, 64}, {139, 154}};
  lhs *= rhs;

  EXPECT_EQ(lhs.Rows(), 2);
  EXPECT_EQ(lhs.Cols(), 2);

  EXPECT_EQ(lhs, expected);
}

TEST(MatrixArithmetic, OutOfPlaceAddition) {
  Matrix<double> lhs{{1, 2, 3}, {4, 5, 6}};
  Matrix<double> rhs{{7, 8, 9}, {10, 11, 12}};
  Matrix<double> sum = lhs + rhs;
  EXPECT_EQ(sum.Rows(), 2);
  EXPECT_EQ(sum.Cols(), 3);

  for (Size i = 0; i < 2; ++i) {
    for (Size j = 0; j < 3; ++j) {
      EXPECT_DOUBLE_EQ(sum(i, j), lhs(i, j) + rhs(i, j));
    }
  }
}

TEST(MatrixArithmetic, OutOfPlaceSubtraction) {
  Matrix<double> lhs{{1, 2, 3}, {4, 5, 6}};
  Matrix<double> rhs{{7, 8, 9}, {10, 11, 12}};
  Matrix<double> diff = lhs - rhs;
  EXPECT_EQ(diff.Rows(), 2);
  EXPECT_EQ(diff.Cols(), 3);

  for (Size i = 0; i < 2; ++i) {
    for (Size j = 0; j < 3; ++j) {
      EXPECT_DOUBLE_EQ(diff(i, j), lhs(i, j) - rhs(i, j));
    }
  }
}

TEST(MatrixArithmetic, OutOfPlaceMultiplication) {
  Matrix<double> lhs{{1, 2, 3}, {4, 5, 6}};
  Matrix<double> rhs{{7, 8}, {9, 10}, {11, 12}};
  Matrix<double> expected{{58, 64}, {139, 154}};
  Matrix<double> product = lhs * rhs;

  EXPECT_EQ(product.Rows(), 2);
  EXPECT_EQ(product.Cols(), 2);

  EXPECT_EQ(product, expected);
}

TEST(MatrixArithmetic, InPlaceScalarAddition) {
  Matrix<double> matrix{{1, 2, 3}, {4, 5, 6}, {7, 8, 9}};
  Matrix<double> expected{{3, 2, 3}, {4, 7, 6}, {7, 8, 11}};
  matrix += 2.0;

  EXPECT_EQ(matrix.Rows(), 3);
  EXPECT_EQ(matrix.Cols(), 3);

  EXPECT_EQ(matrix, expected);
}

TEST(MatrixArithmetic, InPlaceScalarSubtraction) {
  Matrix<double> matrix{{1, 2, 3}, {4, 5, 6}, {7, 8, 9}};
  Matrix<double> expected{{-1, 2, 3}, {4, 3, 6}, {7, 8, 7}};
  matrix -= 2.0;

  EXPECT_EQ(matrix.Rows(), 3);
  EXPECT_EQ(matrix.Cols(), 3);

  EXPECT_EQ(matrix, expected);
}

TEST(MatrixArithmetic, InPlaceScalarMultiplication) {
  Matrix<double> matrix{{1, 2, 3}, {4, 5, 6}, {7, 8, 9}};
  Matrix<double> expected{{2, 4, 6}, {8, 10, 12}, {14, 16, 18}};
  matrix *= 2.0;

  EXPECT_EQ(matrix.Rows(), 3);
  EXPECT_EQ(matrix.Cols(), 3);

  EXPECT_EQ(matrix, expected);
}

TEST(MatrixArithmetic, InPlaceScalarDivision) {
  Matrix<double> matrix{{2, 4, 6}, {8, 10, 12}, {14, 16, 18}};
  Matrix<double> expected{{1, 2, 3}, {4, 5, 6}, {7, 8, 9}};
  matrix /= 2.0;

  EXPECT_EQ(matrix.Rows(), 3);
  EXPECT_EQ(matrix.Cols(), 3);

  EXPECT_EQ(matrix, expected);
}

TEST(MatrixArithmetic, OutOfPlaceScalarAddition) {
  Matrix<double> matrix{{1, 2, 3}, {4, 5, 6}, {7, 8, 9}};
  Matrix<double> expected{{3, 2, 3}, {4, 7, 6}, {7, 8, 11}};
  Matrix<double> sum = matrix + 2.0;

  EXPECT_EQ(sum.Rows(), 3);
  EXPECT_EQ(sum.Cols(), 3);

  EXPECT_EQ(sum, expected);
}

TEST(MatrixArithmetic, OutOfPlaceScalarSubtraction) {
  Matrix<double> matrix{{1, 2, 3}, {4, 5, 6}, {7, 8, 9}};
  Matrix<double> expected{{-1, 2, 3}, {4, 3, 6}, {7, 8, 7}};
  Matrix<double> diff = matrix - 2.0;

  EXPECT_EQ(diff.Rows(), 3);
  EXPECT_EQ(diff.Cols(), 3);

  EXPECT_EQ(diff, expected);
}

TEST(MatrixArithmetic, OutOfPlaceScalarMultiplication) {
  Matrix<double> matrix{{1, 2, 3}, {4, 5, 6}, {7, 8, 9}};
  Matrix<double> expected{{2, 4, 6}, {8, 10, 12}, {14, 16, 18}};
  Matrix<double> product = matrix * 2.0;

  EXPECT_EQ(product.Rows(), 3);
  EXPECT_EQ(product.Cols(), 3);

  EXPECT_EQ(product, expected);
}

TEST(MatrixArithmetic, OutOfPlaceScalarDivision) {
  Matrix<double> matrix{{2, 4, 6}, {8, 10, 12}, {14, 16, 18}};
  Matrix<double> expected{{1, 2, 3}, {4, 5, 6}, {7, 8, 9}};
  Matrix<double> quotient = matrix / 2.0;

  EXPECT_EQ(quotient.Rows(), 3);
  EXPECT_EQ(quotient.Cols(), 3);

  EXPECT_EQ(quotient, expected);
}

TEST(MatrixArithmetic, OutOfPlaceScalarDifferentScalarAddition) {
  Matrix<std::complex<double>> matrix{{1, 2, 3}, {4, 5, 6}, {7, 8, 9}};
  Matrix<std::complex<double>> expected{{3, 2, 3}, {4, 7, 6}, {7, 8, 11}};
  auto sum = matrix + 2.0;

  EXPECT_EQ(sum.Rows(), 3);
  EXPECT_EQ(sum.Cols(), 3);

  EXPECT_EQ(sum, expected);
}

TEST(MatrixArithmetic, OutOfPlaceScalarDifferentScalarDivision) {
  Matrix<std::complex<double>> matrix{{2, 4, 6}, {8, 10, 12}, {14, 16, 18}};
  Matrix<std::complex<double>> expected{{1, 2, 3}, {4, 5, 6}, {7, 8, 9}};
  auto quotient = matrix / 2.0;

  EXPECT_EQ(quotient.Rows(), 3);
  EXPECT_EQ(quotient.Cols(), 3);

  EXPECT_EQ(quotient, expected);
}

TEST(MatrixOperations, FPTranspose) {
  Matrix<double> matrix{{1, 2, 3}, {4, 5, 6}, {7, 8, 9}};
  auto transposed = linalg::Transposed(matrix);
  std::vector<double> expected{1, 4, 7, 2, 5, 8, 3, 6, 9};
  std::vector<double> actual(transposed.begin(), transposed.end());
  EXPECT_EQ(actual, expected);
}

TEST(MatrixOperations, FPTransposeConst) {
  const Matrix<double> matrix{{1, 2, 3}, {4, 5, 6}, {7, 8, 9}};
  auto transposed = linalg::Transposed(matrix);
  std::vector<double> expected{1, 4, 7, 2, 5, 8, 3, 6, 9};
  std::vector<double> actual(transposed.begin(), transposed.end());
  EXPECT_EQ(actual, expected);
}

TEST(MatrixOperations, FPTransposeConstView) {
  const Matrix<double> matrix{{1, 2, 3}, {4, 5, 6}, {7, 8, 9}};
  auto view       = linalg::ConstMatrixView<double>(matrix);
  auto transposed = linalg::Transposed(view);
  std::vector<double> expected{1, 4, 7, 2, 5, 8, 3, 6, 9};
  std::vector<double> actual(transposed.begin(), transposed.end());
  EXPECT_EQ(actual, expected);
}

TEST(MatrixOperations, FPTransposeView) {
  Matrix<double> matrix{{1, 2, 3}, {4, 5, 6}, {7, 8, 9}};
  auto view       = linalg::MatrixView<double>(matrix);
  auto transposed = linalg::Transposed(view);
  std::vector<double> expected{1, 4, 7, 2, 5, 8, 3, 6, 9};
  std::vector<double> actual(transposed.begin(), transposed.end());
  EXPECT_EQ(actual, expected);
}

TEST(MatrixOperations, CTranspose) {
  Matrix<std::complex<double>> matrix{{{1, 1}, {2, 2}, {3, 3}}, {{4, 4}, {5, 5}, {6, 6}}, {{7, 7}, {8, 8}, {9, 9}}};
  auto transposed = linalg::Transposed(matrix);
  std::vector<std::complex<double>> expected{{1, 1}, {4, 4}, {7, 7}, {2, 2}, {5, 5}, {8, 8}, {3, 3}, {6, 6}, {9, 9}};
  std::vector<std::complex<double>> actual(transposed.begin(), transposed.end());
  EXPECT_EQ(actual, expected);
}

TEST(MatrixOperations, CTransposeConst) {
  const Matrix<std::complex<double>> matrix{
      {{1, 1}, {2, 2}, {3, 3}}, {{4, 4}, {5, 5}, {6, 6}}, {{7, 7}, {8, 8}, {9, 9}}};
  auto transposed = linalg::Transposed(matrix);
  std::vector<std::complex<double>> expected{{1, 1}, {4, 4}, {7, 7}, {2, 2}, {5, 5}, {8, 8}, {3, 3}, {6, 6}, {9, 9}};
  std::vector<std::complex<double>> actual(transposed.begin(), transposed.end());
  EXPECT_EQ(actual, expected);
}

TEST(MatrixOperations, CTransposeConstView) {
  const Matrix<std::complex<double>> matrix{
      {{1, 1}, {2, 2}, {3, 3}}, {{4, 4}, {5, 5}, {6, 6}}, {{7, 7}, {8, 8}, {9, 9}}};
  auto view       = linalg::ConstMatrixView<std::complex<double>>(matrix);
  auto transposed = linalg::Transposed(view);
  std::vector<std::complex<double>> expected{{1, 1}, {4, 4}, {7, 7}, {2, 2}, {5, 5}, {8, 8}, {3, 3}, {6, 6}, {9, 9}};
  std::vector<std::complex<double>> actual(transposed.begin(), transposed.end());
  EXPECT_EQ(actual, expected);
}

TEST(MatrixOperations, CTransposeView) {
  Matrix<std::complex<double>> matrix{{{1, 1}, {2, 2}, {3, 3}}, {{4, 4}, {5, 5}, {6, 6}}, {{7, 7}, {8, 8}, {9, 9}}};
  auto view       = linalg::MatrixView<std::complex<double>>(matrix);
  auto transposed = linalg::Transposed(view);
  std::vector<std::complex<double>> expected{{1, 1}, {4, 4}, {7, 7}, {2, 2}, {5, 5}, {8, 8}, {3, 3}, {6, 6}, {9, 9}};
  std::vector<std::complex<double>> actual(transposed.begin(), transposed.end());
  EXPECT_EQ(actual, expected);
}

TEST(MatrixOperations, FPConjugate) {
  Matrix<double> matrix{{1, 2, 3}, {4, 5, 6}, {7, 8, 9}};
  auto conjugated = linalg::Conjugated(matrix);

  // Issue described in src/matrix.h.
  static_assert(std::is_same_v<decltype(conjugated), linalg::MatrixView<double>>);

  std::vector<double> expected{1, 4, 7, 2, 5, 8, 3, 6, 9};
  std::vector<double> actual(conjugated.begin(), conjugated.end());
  EXPECT_EQ(actual, expected);
}

TEST(MatrixOperations, FPConjugateConst) {
  const Matrix<double> matrix{{1, 2, 3}, {4, 5, 6}, {7, 8, 9}};
  auto conjugated = linalg::Conjugated(matrix);

  // Issue described in src/matrix.h.
  static_assert(std::is_same_v<decltype(conjugated), linalg::ConstMatrixView<double>>);

  std::vector<double> expected{1, 4, 7, 2, 5, 8, 3, 6, 9};
  std::vector<double> actual(conjugated.begin(), conjugated.end());
  EXPECT_EQ(actual, expected);
}

TEST(MatrixOperations, FPConjugateConstView) {
  const Matrix<double> matrix{{1, 2, 3}, {4, 5, 6}, {7, 8, 9}};
  auto view       = linalg::ConstMatrixView<double>(matrix);
  auto conjugated = linalg::Conjugated(view);

  // Issue described in src/matrix.h.
  static_assert(std::is_same_v<decltype(conjugated), linalg::ConstMatrixView<double>>);

  std::vector<double> expected{1, 4, 7, 2, 5, 8, 3, 6, 9};
  std::vector<double> actual(conjugated.begin(), conjugated.end());
  EXPECT_EQ(actual, expected);
}

TEST(MatrixOperations, FPConjugateView) {
  Matrix<double> matrix{{1, 2, 3}, {4, 5, 6}, {7, 8, 9}};
  auto view       = linalg::MatrixView<double>(matrix);
  auto conjugated = linalg::Conjugated(view);

  // Issue described in src/matrix.h.
  static_assert(std::is_same_v<decltype(conjugated), linalg::MatrixView<double>>);

  std::vector<double> expected{1, 4, 7, 2, 5, 8, 3, 6, 9};
  std::vector<double> actual(conjugated.begin(), conjugated.end());
  EXPECT_EQ(actual, expected);
}

TEST(MatrixOperations, CConjugate) {
  Matrix<std::complex<double>> matrix{{{1, 1}, {2, 2}, {3, 3}}, {{4, 4}, {5, 5}, {6, 6}}, {{7, 7}, {8, 8}, {9, 9}}};
  auto conjugated = linalg::Conjugated(matrix);

  // Issue described in src/matrix.h.
  static_assert(std::is_same_v<decltype(conjugated), linalg::Matrix<std::complex<double>>>);

  std::vector<std::complex<double>> expected{{1, -1}, {4, -4}, {7, -7}, {2, -2}, {5, -5},
                                             {8, -8}, {3, -3}, {6, -6}, {9, -9}};
  std::vector<std::complex<double>> actual(conjugated.begin(), conjugated.end());
  EXPECT_EQ(actual, expected);
}

TEST(MatrixOperations, CConjugateConst) {
  const Matrix<std::complex<double>> matrix{
      {{1, 1}, {2, 2}, {3, 3}}, {{4, 4}, {5, 5}, {6, 6}}, {{7, 7}, {8, 8}, {9, 9}}};
  auto conjugated = linalg::Conjugated(matrix);

  // Issue described in src/matrix.h.
  static_assert(std::is_same_v<decltype(conjugated), linalg::Matrix<std::complex<double>>>);

  std::vector<std::complex<double>> expected{{1, -1}, {4, -4}, {7, -7}, {2, -2}, {5, -5},
                                             {8, -8}, {3, -3}, {6, -6}, {9, -9}};
  std::vector<std::complex<double>> actual(conjugated.begin(), conjugated.end());
  EXPECT_EQ(actual, expected);
}

TEST(MatrixOperations, CConjugateConstView) {
  const Matrix<std::complex<double>> matrix{
      {{1, 1}, {2, 2}, {3, 3}}, {{4, 4}, {5, 5}, {6, 6}}, {{7, 7}, {8, 8}, {9, 9}}};
  auto view       = linalg::ConstMatrixView<std::complex<double>>(matrix);
  auto conjugated = linalg::Conjugated(view);

  // Issue described in src/matrix.h.
  static_assert(std::is_same_v<decltype(conjugated), linalg::Matrix<std::complex<double>>>);

  std::vector<std::complex<double>> expected{{1, -1}, {4, -4}, {7, -7}, {2, -2}, {5, -5},
                                             {8, -8}, {3, -3}, {6, -6}, {9, -9}};
  std::vector<std::complex<double>> actual(conjugated.begin(), conjugated.end());
  EXPECT_EQ(actual, expected);
}

TEST(MatrixOperations, CConjugateView) {
  Matrix<std::complex<double>> matrix{{{1, 1}, {2, 2}, {3, 3}}, {{4, 4}, {5, 5}, {6, 6}}, {{7, 7}, {8, 8}, {9, 9}}};
  auto view       = linalg::MatrixView<std::complex<double>>(matrix);
  auto conjugated = linalg::Conjugated(view);

  // Issue described in src/matrix.h.
  static_assert(std::is_same_v<decltype(conjugated), linalg::Matrix<std::complex<double>>>);

  std::vector<std::complex<double>> expected{{1, -1}, {4, -4}, {7, -7}, {2, -2}, {5, -5},
                                             {8, -8}, {3, -3}, {6, -6}, {9, -9}};
  std::vector<std::complex<double>> actual(conjugated.begin(), conjugated.end());
  EXPECT_EQ(actual, expected);
}

TEST(MatrixOperations, FPEuclideanNorm) {
  Matrix<double> matrix{{1, 2, 3}};

  double norm = linalg::EuclideanVectorNorm(matrix);
  EXPECT_DOUBLE_EQ(norm, std::sqrt(14.0));
}

TEST(MatrixOperations, FPEuclideanNormConst) {
  const Matrix<double> matrix{{1, 2, 3}};

  double norm = linalg::EuclideanVectorNorm(matrix);
  EXPECT_DOUBLE_EQ(norm, std::sqrt(14.0));
}

TEST(MatrixOperations, FPEuclideanNormConstView) {
  const Matrix<double> matrix{{1, 2, 3}};
  auto view = linalg::ConstMatrixView<double>(matrix);

  double norm = linalg::EuclideanVectorNorm(view);
  EXPECT_DOUBLE_EQ(norm, std::sqrt(14.0));
}

TEST(MatrixOperations, FPEuclideanNormView) {
  Matrix<double> matrix{{1, 2, 3}};
  auto view = linalg::MatrixView<double>(matrix);

  double norm = linalg::EuclideanVectorNorm(view);
  EXPECT_DOUBLE_EQ(norm, std::sqrt(14.0));
}

TEST(MatrixSerialization, Floating) {
  Matrix<double> matrix{{1, 2, 3}, {4, 5, 6}};
  std::ostringstream oss;
  oss << std::fixed << std::setprecision(1) << matrix;
  std::string expected = "1.0 2.0 3.0\n4.0 5.0 6.0";
  EXPECT_EQ(oss.str(), expected);
}

TEST(MatrixSerialization, Complex) {
  Matrix<std::complex<double>> matrix{{{1, 1}, {2, 2}, {3, 3}}, {{4, 4}, {5, 5}, {6, 6}}};
  std::ostringstream oss;
  oss << std::fixed << std::setprecision(1) << matrix;
  std::string expected = "(1.0,1.0) (2.0,2.0) (3.0,3.0)\n(4.0,4.0) (5.0,5.0) (6.0,6.0)";
  EXPECT_EQ(oss.str(), expected);
}

TEST(MatrixSerialization, FloatingView) {
  Matrix<double> matrix{{1, 2, 3}, {4, 5, 6}};
  auto view = linalg::MatrixView<double>(matrix);
  std::ostringstream oss;
  oss << std::fixed << std::setprecision(1) << view;
  std::string expected = "1.0 2.0 3.0\n4.0 5.0 6.0";
  EXPECT_EQ(oss.str(), expected);
}

TEST(MatrixSerialization, ComplexView) {
  Matrix<std::complex<double>> matrix{{{1, 1}, {2, 2}, {3, 3}}, {{4, 4}, {5, 5}, {6, 6}}};
  auto view = linalg::MatrixView<std::complex<double>>(matrix);
  std::ostringstream oss;
  oss << std::fixed << std::setprecision(1) << view;
  std::string expected = "(1.0,1.0) (2.0,2.0) (3.0,3.0)\n(4.0,4.0) (5.0,5.0) (6.0,6.0)";
  EXPECT_EQ(oss.str(), expected);
}

TEST(MatrixDeserialization, Floating) {
  Matrix<double> matrix(linalg::ERows{2}, linalg::ECols{3});
  std::istringstream iss("1.0 2.0 3.0\n4.0 5.0 6.0");
  iss >> matrix;
  EXPECT_EQ(matrix.Rows(), 2);
  EXPECT_EQ(matrix.Cols(), 3);

  std::vector expected{
      1.0, 2.0, 3.0, 4.0, 5.0, 6.0,
  };
  std::vector<double> actual(matrix.begin(), matrix.end());

  EXPECT_EQ(actual, expected);
}

TEST(MatrixDeserialization, Complex) {
  Matrix<std::complex<double>> matrix(linalg::ERows{2}, linalg::ECols{3});
  std::istringstream iss("(1.0,1.0) (2.0,2.0) (3.0,3.0)\n(4.0,4.0) (5.0,5.0) (6.0,6.0)");
  iss >> matrix;
  EXPECT_EQ(matrix.Rows(), 2);
  EXPECT_EQ(matrix.Cols(), 3);

  std::vector<std::complex<double>> expected{
      {1.0, 1.0}, {2.0, 2.0}, {3.0, 3.0}, {4.0, 4.0}, {5.0, 5.0}, {6.0, 6.0},
  };
  std::vector<std::complex<double>> actual(matrix.begin(), matrix.end());

  EXPECT_EQ(actual, expected);
}

TEST(MatrixDeserialization, FloatingView) {
  Matrix<double> matrix(linalg::ERows{2}, linalg::ECols{3});
  auto view = linalg::MatrixView<double>(matrix);
  std::istringstream iss("1.0 2.0 3.0\n4.0 5.0 6.0");
  iss >> view;
  EXPECT_EQ(matrix.Rows(), 2);
  EXPECT_EQ(matrix.Cols(), 3);

  std::vector expected{
      1.0, 2.0, 3.0, 4.0, 5.0, 6.0,
  };
  std::vector<double> actual(matrix.begin(), matrix.end());

  EXPECT_EQ(actual, expected);
}

TEST(MatrixDeserialization, ComplexView) {
  Matrix<std::complex<double>> matrix(linalg::ERows{2}, linalg::ECols{3});
  auto view = linalg::MatrixView<std::complex<double>>(matrix);
  std::istringstream iss("(1.0,1.0) (2.0,2.0) (3.0,3.0)\n(4.0,4.0) (5.0,5.0) (6.0,6.0)");
  iss >> view;
  EXPECT_EQ(matrix.Rows(), 2);
  EXPECT_EQ(matrix.Cols(), 3);

  std::vector<std::complex<double>> expected{
      {1.0, 1.0}, {2.0, 2.0}, {3.0, 3.0}, {4.0, 4.0}, {5.0, 5.0}, {6.0, 6.0},
  };
  std::vector<std::complex<double>> actual(matrix.begin(), matrix.end());

  EXPECT_EQ(actual, expected);
}

}  // namespace linalg::test
