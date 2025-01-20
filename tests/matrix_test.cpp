#include "../src/matrix.h"

#include <gtest/gtest.h>

#include <complex>
#include <concepts>
#include <iterator>

namespace {

using linalg::Matrix;

TEST(MatrixCtors, DefaultCtor) {
  Matrix<double> matrix;
  EXPECT_EQ(matrix.Rows(), 0);
  EXPECT_EQ(matrix.Cols(), 0);
  EXPECT_TRUE(matrix.empty());
}

TEST(MatrixCtors, CopyCtor_EmptyMatrix) {
  Matrix<double> matrix;
  Matrix<double> copy(matrix);
  EXPECT_EQ(copy.Rows(), 0);
  EXPECT_EQ(copy.Cols(), 0);
  EXPECT_TRUE(copy.empty());
}

TEST(MatrixCtors, MoveCtor_EmptyMatrix) {
  Matrix<double> matrix;
  Matrix moved(std::move(matrix));
  EXPECT_EQ(moved.Rows(), 0);
  EXPECT_EQ(moved.Cols(), 0);
  EXPECT_TRUE(moved.empty());
}

TEST(MatrixCtors, SizeCtor) {
  Matrix<double> matrix(2, 3);
  EXPECT_EQ(matrix.Rows(), 2);
  EXPECT_EQ(matrix.Cols(), 3);
  EXPECT_FALSE(matrix.empty());
}

TEST(MatrixCtors, InitializerListCtor) {
  Matrix<double> matrix{{1, 2, 3}, {4, 5, 6}};
  EXPECT_EQ(matrix.Rows(), 2);
  EXPECT_EQ(matrix.Cols(), 3);
  EXPECT_FALSE(matrix.empty());
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
  EXPECT_FALSE(copy.empty());
  EXPECT_DOUBLE_EQ(copy(0, 0), 1.0);
  EXPECT_DOUBLE_EQ(copy(0, 1), 2.0);
  EXPECT_DOUBLE_EQ(copy(0, 2), 3.0);
  EXPECT_DOUBLE_EQ(copy(1, 0), 4.0);
  EXPECT_DOUBLE_EQ(copy(1, 1), 5.0);
  EXPECT_DOUBLE_EQ(copy(1, 2), 6.0);

  // No side effects on the original matrix.
  EXPECT_EQ(matrix.Rows(), 2);
  EXPECT_EQ(matrix.Cols(), 3);
  EXPECT_FALSE(matrix.empty());
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
  EXPECT_FALSE(moved.empty());
  EXPECT_DOUBLE_EQ(moved(0, 0), 1.0);
  EXPECT_DOUBLE_EQ(moved(0, 1), 2.0);
  EXPECT_DOUBLE_EQ(moved(0, 2), 3.0);
  EXPECT_DOUBLE_EQ(moved(1, 0), 4.0);
  EXPECT_DOUBLE_EQ(moved(1, 1), 5.0);
  EXPECT_DOUBLE_EQ(moved(1, 2), 6.0);

  // Data from matrix should be moved.
  EXPECT_EQ(matrix.Rows(), 0);
  EXPECT_EQ(matrix.Cols(), 0);
  EXPECT_TRUE(matrix.empty());
}

TEST(MatrixAssignment, CopyAssignment) {
  Matrix<double> matrix{{1, 2, 3}, {4, 5, 6}};
  Matrix<double> copy;
  copy = matrix;
  EXPECT_EQ(copy.Rows(), 2);
  EXPECT_EQ(copy.Cols(), 3);
  EXPECT_FALSE(copy.empty());
  EXPECT_DOUBLE_EQ(copy(0, 0), 1.0);
  EXPECT_DOUBLE_EQ(copy(0, 1), 2.0);
  EXPECT_DOUBLE_EQ(copy(0, 2), 3.0);
  EXPECT_DOUBLE_EQ(copy(1, 0), 4.0);
  EXPECT_DOUBLE_EQ(copy(1, 1), 5.0);
  EXPECT_DOUBLE_EQ(copy(1, 2), 6.0);

  // No side effects on the original matrix.
  EXPECT_EQ(matrix.Rows(), 2);
  EXPECT_EQ(matrix.Cols(), 3);
  EXPECT_FALSE(matrix.empty());
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
  EXPECT_FALSE(moved.empty());
  EXPECT_DOUBLE_EQ(moved(0, 0), 1.0);
  EXPECT_DOUBLE_EQ(moved(0, 1), 2.0);
  EXPECT_DOUBLE_EQ(moved(0, 2), 3.0);
  EXPECT_DOUBLE_EQ(moved(1, 0), 4.0);
  EXPECT_DOUBLE_EQ(moved(1, 1), 5.0);
  EXPECT_DOUBLE_EQ(moved(1, 2), 6.0);

  // Data from matrix should be moved.
  EXPECT_EQ(matrix.Rows(), 0);
  EXPECT_EQ(matrix.Cols(), 0);
  EXPECT_TRUE(matrix.empty());
}

TEST(MatrixStaticCreate, Identity) {
  Matrix<double> identity = Matrix<double>::Identity(3);
  EXPECT_EQ(identity.Rows(), 3);
  EXPECT_EQ(identity.Cols(), 3);
  EXPECT_FALSE(identity.empty());
  for (size_t i = 0; i < 3; ++i) {
    for (size_t j = 0; j < 3; ++j) {
      if (i == j) {
        EXPECT_DOUBLE_EQ(identity(i, j), 1.0);
      } else {
        EXPECT_DOUBLE_EQ(identity(i, j), 0.0);
      }
    }
  }
}

TEST(MatrixStaticCreate, Zero) {
  Matrix<double> zero = Matrix<double>::Zero(2, 3);
  EXPECT_EQ(zero.Rows(), 2);
  EXPECT_EQ(zero.Cols(), 3);
  EXPECT_FALSE(zero.empty());
  for (size_t i = 0; i < 2; ++i) {
    for (size_t j = 0; j < 3; ++j) {
      EXPECT_DOUBLE_EQ(zero(i, j), 0.0);
    }
  }
}

TEST(MatrixStaticCreate, Unit) {
  Matrix<double> unit = Matrix<double>::Unit(2, 3, 1, 2);
  EXPECT_EQ(unit.Rows(), 2);
  EXPECT_EQ(unit.Cols(), 3);
  EXPECT_FALSE(unit.empty());
  for (size_t i = 0; i < 2; ++i) {
    for (size_t j = 0; j < 3; ++j) {
      if (i == 1 && j == 2) {
        EXPECT_DOUBLE_EQ(unit(i, j), 1.0);
      } else {
        EXPECT_DOUBLE_EQ(unit(i, j), 0.0);
      }
    }
  }
}

TEST(MatrixStaticCreate, DiagonalScalar) {
  Matrix<double> diagonal = Matrix<double>::Diagonal(3, 2.0);
  EXPECT_EQ(diagonal.Rows(), 3);
  EXPECT_EQ(diagonal.Cols(), 3);
  EXPECT_FALSE(diagonal.empty());
  for (size_t i = 0; i < 3; ++i) {
    for (size_t j = 0; j < 3; ++j) {
      if (i == j) {
        EXPECT_DOUBLE_EQ(diagonal(i, j), 2.0);
      } else {
        EXPECT_DOUBLE_EQ(diagonal(i, j), 0.0);
      }
    }
  }
}

TEST(MatrixStaticCreate, DiagonalList) {
  Matrix<double> diagonal = Matrix<double>::Diagonal(3, {1.0, 2.0, 3.0});
  EXPECT_EQ(diagonal.Rows(), 3);
  EXPECT_EQ(diagonal.Cols(), 3);
  EXPECT_FALSE(diagonal.empty());
  for (size_t i = 0; i < 3; ++i) {
    for (size_t j = 0; j < 3; ++j) {
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
  EXPECT_FALSE(diagonal.empty());
  for (size_t i = 0; i < 3; ++i) {
    for (size_t j = 0; j < 3; ++j) {
      if (i == j) {
        EXPECT_DOUBLE_EQ(diagonal(i, j), values[i]);
      } else {
        EXPECT_DOUBLE_EQ(diagonal(i, j), 0.0);
      }
    }
  }
}

// Copied from https://stackoverflow.com/a/60491447/22523445
// based on https://en.cppreference.com/w/cpp/named_req/Container
// Using only for testing purposes.
template <class ContainerType>
concept Container = requires(ContainerType a, const ContainerType b) {
  requires std::regular<ContainerType>;
  requires std::swappable<ContainerType>;
  requires std::destructible<typename ContainerType::value_type>;
  requires std::same_as<typename ContainerType::reference, typename ContainerType::value_type &>;
  requires std::same_as<typename ContainerType::const_reference, const typename ContainerType::value_type &>;
  requires std::forward_iterator<typename ContainerType::iterator>;
  requires std::forward_iterator<typename ContainerType::const_iterator>;
  requires std::signed_integral<typename ContainerType::difference_type>;
  requires std::same_as<typename ContainerType::difference_type,
                        typename std::iterator_traits<typename ContainerType::iterator>::difference_type>;
  requires std::same_as<typename ContainerType::difference_type,
                        typename std::iterator_traits<typename ContainerType::const_iterator>::difference_type>;
  { a.begin() } -> std::same_as<typename ContainerType::iterator>;
  { a.end() } -> std::same_as<typename ContainerType::iterator>;
  { b.begin() } -> std::same_as<typename ContainerType::const_iterator>;
  { b.end() } -> std::same_as<typename ContainerType::const_iterator>;
  { a.cbegin() } -> std::same_as<typename ContainerType::const_iterator>;
  { a.cend() } -> std::same_as<typename ContainerType::const_iterator>;
  { a.size() } -> std::same_as<typename ContainerType::size_type>;
  { a.max_size() } -> std::same_as<typename ContainerType::size_type>;
  { a.empty() } -> std::same_as<bool>;
};

template <typename ContainerType>
concept ReversibleContainer = Container<ContainerType> && requires(ContainerType a) {
  requires std::same_as<typename ContainerType::reverse_iterator,
                        std::reverse_iterator<typename ContainerType::iterator>>;
  requires std::same_as<typename ContainerType::const_reverse_iterator,
                        std::reverse_iterator<typename ContainerType::const_iterator>>;
  { a.rbegin() } -> std::same_as<typename ContainerType::reverse_iterator>;
  { a.rend() } -> std::same_as<typename ContainerType::reverse_iterator>;
  { a.crbegin() } -> std::same_as<typename ContainerType::const_reverse_iterator>;
  { a.crend() } -> std::same_as<typename ContainerType::const_reverse_iterator>;
};

template <typename ContainerType>
concept ContiguousContainer = Container<ContainerType> && requires(ContainerType a) {
  requires std::contiguous_iterator<typename ContainerType::iterator>;
  requires std::contiguous_iterator<typename ContainerType::const_iterator>;
};

TEST(MatrixStaticAsserts, ContainerConcepts) {
  static_assert(Container<Matrix<double>>);
  static_assert(Container<Matrix<std::complex<double>>>);

  static_assert(ReversibleContainer<Matrix<double>>);
  static_assert(ReversibleContainer<Matrix<std::complex<double>>>);

  static_assert(ContiguousContainer<Matrix<double>>);
  static_assert(ContiguousContainer<Matrix<std::complex<double>>>);
}

TEST(MatrixArithmetic, Negate) {
  Matrix<double> matrix{{1, 2, 3}, {4, 5, 6}};
  Matrix<double> negated = -matrix;
  EXPECT_EQ(negated.Rows(), 2);
  EXPECT_EQ(negated.Cols(), 3);
  EXPECT_FALSE(negated.empty());
  for (size_t i = 0; i < 2; ++i) {
    for (size_t j = 0; j < 3; ++j) {
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
  EXPECT_FALSE(sum.empty());
  for (size_t i = 0; i < 2; ++i) {
    for (size_t j = 0; j < 3; ++j) {
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
  EXPECT_FALSE(diff.empty());
  for (size_t i = 0; i < 2; ++i) {
    for (size_t j = 0; j < 3; ++j) {
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
  EXPECT_FALSE(lhs.empty());
  EXPECT_EQ(lhs, expected);
}

TEST(MatrixArithmetic, OutOfPlaceAddition) {
  Matrix<double> lhs{{1, 2, 3}, {4, 5, 6}};
  Matrix<double> rhs{{7, 8, 9}, {10, 11, 12}};
  Matrix<double> sum = lhs + rhs;
  EXPECT_EQ(sum.Rows(), 2);
  EXPECT_EQ(sum.Cols(), 3);
  EXPECT_FALSE(sum.empty());
  for (size_t i = 0; i < 2; ++i) {
    for (size_t j = 0; j < 3; ++j) {
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
  EXPECT_FALSE(diff.empty());
  for (size_t i = 0; i < 2; ++i) {
    for (size_t j = 0; j < 3; ++j) {
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
  EXPECT_FALSE(product.empty());
  EXPECT_EQ(product, expected);
}

TEST(MatrixArithmetic, InPlaceScalarAddition) {
  Matrix<double> matrix{{1, 2, 3}, {4, 5, 6}, {7, 8, 9}};
  Matrix<double> expected{{3, 2, 3}, {4, 7, 6}, {7, 8, 11}};
  matrix += 2.0;

  EXPECT_EQ(matrix.Rows(), 3);
  EXPECT_EQ(matrix.Cols(), 3);
  EXPECT_FALSE(matrix.empty());
  EXPECT_EQ(matrix, expected);
}

TEST(MatrixArithmetic, InPlaceScalarSubtraction) {
  Matrix<double> matrix{{1, 2, 3}, {4, 5, 6}, {7, 8, 9}};
  Matrix<double> expected{{-1, 2, 3}, {4, 3, 6}, {7, 8, 7}};
  matrix -= 2.0;

  EXPECT_EQ(matrix.Rows(), 3);
  EXPECT_EQ(matrix.Cols(), 3);
  EXPECT_FALSE(matrix.empty());
  EXPECT_EQ(matrix, expected);
}

TEST(MatrixArithmetic, InPlaceScalarMultiplication) {
  Matrix<double> matrix{{1, 2, 3}, {4, 5, 6}, {7, 8, 9}};
  Matrix<double> expected{{2, 4, 6}, {8, 10, 12}, {14, 16, 18}};
  matrix *= 2.0;

  EXPECT_EQ(matrix.Rows(), 3);
  EXPECT_EQ(matrix.Cols(), 3);
  EXPECT_FALSE(matrix.empty());
  EXPECT_EQ(matrix, expected);
}

TEST(MatrixArithmetic, InPlaceScalarDivision) {
  Matrix<double> matrix{{2, 4, 6}, {8, 10, 12}, {14, 16, 18}};
  Matrix<double> expected{{1, 2, 3}, {4, 5, 6}, {7, 8, 9}};
  matrix /= 2.0;

  EXPECT_EQ(matrix.Rows(), 3);
  EXPECT_EQ(matrix.Cols(), 3);
  EXPECT_FALSE(matrix.empty());
  EXPECT_EQ(matrix, expected);
}

TEST(MatrixArithmetic, OutOfPlaceScalarAddition) {
  Matrix<double> matrix{{1, 2, 3}, {4, 5, 6}, {7, 8, 9}};
  Matrix<double> expected{{3, 2, 3}, {4, 7, 6}, {7, 8, 11}};
  Matrix<double> sum = matrix + 2.0;

  EXPECT_EQ(sum.Rows(), 3);
  EXPECT_EQ(sum.Cols(), 3);
  EXPECT_FALSE(sum.empty());
  EXPECT_EQ(sum, expected);
}

TEST(MatrixArithmetic, OutOfPlaceScalarSubtraction) {
  Matrix<double> matrix{{1, 2, 3}, {4, 5, 6}, {7, 8, 9}};
  Matrix<double> expected{{-1, 2, 3}, {4, 3, 6}, {7, 8, 7}};
  Matrix<double> diff = matrix - 2.0;

  EXPECT_EQ(diff.Rows(), 3);
  EXPECT_EQ(diff.Cols(), 3);
  EXPECT_FALSE(diff.empty());
  EXPECT_EQ(diff, expected);
}

TEST(MatrixArithmetic, OutOfPlaceScalarMultiplication) {
  Matrix<double> matrix{{1, 2, 3}, {4, 5, 6}, {7, 8, 9}};
  Matrix<double> expected{{2, 4, 6}, {8, 10, 12}, {14, 16, 18}};
  Matrix<double> product = matrix * 2.0;

  EXPECT_EQ(product.Rows(), 3);
  EXPECT_EQ(product.Cols(), 3);
  EXPECT_FALSE(product.empty());
  EXPECT_EQ(product, expected);
}

TEST(MatrixArithmetic, OutOfPlaceScalarDivision) {
  Matrix<double> matrix{{2, 4, 6}, {8, 10, 12}, {14, 16, 18}};
  Matrix<double> expected{{1, 2, 3}, {4, 5, 6}, {7, 8, 9}};
  Matrix<double> quotient = matrix / 2.0;

  EXPECT_EQ(quotient.Rows(), 3);
  EXPECT_EQ(quotient.Cols(), 3);
  EXPECT_FALSE(quotient.empty());
  EXPECT_EQ(quotient, expected);
}

}  // namespace
