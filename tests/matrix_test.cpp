#include "../src/matrix.h"

#include <gtest/gtest.h>

#include <cmath>
#include <complex>
#include <limits>

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
  EXPECT_TRUE(matrix(0, 0) == 1.0);
  EXPECT_TRUE(matrix(0, 1) == 2.0);
  EXPECT_TRUE(matrix(0, 2) == 3.0);
  EXPECT_TRUE(matrix(1, 0) == 4.0);
  EXPECT_TRUE(matrix(1, 1) == 5.0);
  EXPECT_TRUE(matrix(1, 2) == 6.0);
}

TEST(MatrixCtors, CopyConstructor) {
  Matrix<double> matrix{{1, 2, 3}, {4, 5, 6}};
  Matrix copy(matrix);
  EXPECT_EQ(copy.Rows(), 2);
  EXPECT_EQ(copy.Cols(), 3);
  EXPECT_FALSE(copy.empty());
  EXPECT_TRUE(copy(0, 0) == 1.0);
  EXPECT_TRUE(copy(0, 1) == 2.0);
  EXPECT_TRUE(copy(0, 2) == 3.0);
  EXPECT_TRUE(copy(1, 0) == 4.0);
  EXPECT_TRUE(copy(1, 1) == 5.0);
  EXPECT_TRUE(copy(1, 2) == 6.0);

  // No side effects on the original matrix.
  EXPECT_EQ(matrix.Rows(), 2);
  EXPECT_EQ(matrix.Cols(), 3);
  EXPECT_FALSE(matrix.empty());
  EXPECT_TRUE(matrix(0, 0) == 1.0);
  EXPECT_TRUE(matrix(0, 1) == 2.0);
  EXPECT_TRUE(matrix(0, 2) == 3.0);
  EXPECT_TRUE(matrix(1, 0) == 4.0);
  EXPECT_TRUE(matrix(1, 1) == 5.0);
  EXPECT_TRUE(matrix(1, 2) == 6.0);
}

TEST(MatrixCtors, MoveCtor) {
  Matrix<double> matrix{{1, 2, 3}, {4, 5, 6}};
  Matrix moved(std::move(matrix));
  EXPECT_EQ(moved.Rows(), 2);
  EXPECT_EQ(moved.Cols(), 3);
  EXPECT_FALSE(moved.empty());
  EXPECT_TRUE(moved(0, 0) == 1.0);
  EXPECT_TRUE(moved(0, 1) == 2.0);
  EXPECT_TRUE(moved(0, 2) == 3.0);
  EXPECT_TRUE(moved(1, 0) == 4.0);
  EXPECT_TRUE(moved(1, 1) == 5.0);
  EXPECT_TRUE(moved(1, 2) == 6.0);

  // Data from matrix should be moved.
  EXPECT_TRUE(matrix.Rows() == 0);
  EXPECT_TRUE(matrix.Cols() == 0);
  EXPECT_TRUE(matrix.empty());
}

TEST(MatrixAssignment, CopyAssignment) {
  Matrix<double> matrix{{1, 2, 3}, {4, 5, 6}};
  Matrix<double> copy;
  copy = matrix;
  EXPECT_EQ(copy.Rows(), 2);
  EXPECT_EQ(copy.Cols(), 3);
  EXPECT_FALSE(copy.empty());
  EXPECT_TRUE(copy(0, 0) == 1.0);
  EXPECT_TRUE(copy(0, 1) == 2.0);
  EXPECT_TRUE(copy(0, 2) == 3.0);
  EXPECT_TRUE(copy(1, 0) == 4.0);
  EXPECT_TRUE(copy(1, 1) == 5.0);
  EXPECT_TRUE(copy(1, 2) == 6.0);

  // No side effects on the original matrix.
  EXPECT_EQ(matrix.Rows(), 2);
  EXPECT_EQ(matrix.Cols(), 3);
  EXPECT_FALSE(matrix.empty());
  EXPECT_TRUE(matrix(0, 0) == 1.0);
  EXPECT_TRUE(matrix(0, 1) == 2.0);
  EXPECT_TRUE(matrix(0, 2) == 3.0);
  EXPECT_TRUE(matrix(1, 0) == 4.0);
  EXPECT_TRUE(matrix(1, 1) == 5.0);
  EXPECT_TRUE(matrix(1, 2) == 6.0);
}

TEST(MatrixAssignment, MoveAssignment) {
  Matrix<double> matrix{{1, 2, 3}, {4, 5, 6}};
  Matrix<double> moved;
  moved = std::move(matrix);
  EXPECT_EQ(moved.Rows(), 2);
  EXPECT_EQ(moved.Cols(), 3);
  EXPECT_FALSE(moved.empty());
  EXPECT_TRUE(moved(0, 0) == 1.0);
  EXPECT_TRUE(moved(0, 1) == 2.0);
  EXPECT_TRUE(moved(0, 2) == 3.0);
  EXPECT_TRUE(moved(1, 0) == 4.0);
  EXPECT_TRUE(moved(1, 1) == 5.0);
  EXPECT_TRUE(moved(1, 2) == 6.0);

  // Data from matrix should be moved.
  EXPECT_TRUE(matrix.Rows() == 0);
  EXPECT_TRUE(matrix.Cols() == 0);
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
        EXPECT_TRUE(identity(i, j) == 1.0);
      } else {
        EXPECT_TRUE(identity(i, j) == 0.0);
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
      EXPECT_TRUE(zero(i, j) == 0.0);
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
        EXPECT_TRUE(unit(i, j) == 1.0);
      } else {
        EXPECT_TRUE(unit(i, j) == 0.0);
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
        EXPECT_TRUE(diagonal(i, j) == 2.0);
      } else {
        EXPECT_TRUE(diagonal(i, j) == 0.0);
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
        EXPECT_TRUE(std::fabs(diagonal(i, j) - (static_cast<double>(i) + 1)) < std::numeric_limits<double>::epsilon());
      } else {
        EXPECT_TRUE(diagonal(i, j) == 0.0);
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
        EXPECT_TRUE(std::fabs(diagonal(i, j) - values[i]) < std::numeric_limits<double>::epsilon());
      } else {
        EXPECT_TRUE(diagonal(i, j) == 0.0);
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

TEST(MatrixStaticAsserts, ContainerConcepts) {
  static_assert(Container<Matrix<double>>);
  static_assert(Container<Matrix<std::complex<double>>>);
}

}  // namespace
