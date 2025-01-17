#include "../src/matrix.h"

#include <gtest/gtest.h>

namespace {

template <typename T>
using Matrix = linalg::Matrix<T>;

TEST(Matrix, DefaultConstructor) {
  Matrix<double> matrix;
  EXPECT_EQ(matrix.Rows(), 0);
  EXPECT_EQ(matrix.Cols(), 0);
  EXPECT_TRUE(matrix.Empty());
}

TEST(Matrix, CopyConstructor_EmptyMatrix) {
  Matrix<double> matrix;
  Matrix<double> copy(matrix);
  EXPECT_EQ(copy.Rows(), 0);
  EXPECT_EQ(copy.Cols(), 0);
  EXPECT_TRUE(copy.Empty());
}

TEST(Matrix, MoveConstructor_EmptyMatrix) {
  Matrix<double> matrix;
  Matrix<double> moved(std::move(matrix));
  EXPECT_EQ(moved.Rows(), 0);
  EXPECT_EQ(moved.Cols(), 0);
  EXPECT_TRUE(moved.Empty());
}
}  // namespace
