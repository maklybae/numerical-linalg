#include "../src/matrix.h"

#include <gtest/gtest.h>

namespace {

using linalg::Matrix;

TEST(MatrixCtors, DefaultCtor) {
  Matrix<double> matrix;
  EXPECT_EQ(matrix.Rows(), 0);
  EXPECT_EQ(matrix.Cols(), 0);
  EXPECT_TRUE(matrix.Empty());
}

TEST(MatrixCtors, CopyCtor_EmptyMatrix) {
  Matrix<double> matrix;
  Matrix<double> copy(matrix);
  EXPECT_EQ(copy.Rows(), 0);
  EXPECT_EQ(copy.Cols(), 0);
  EXPECT_TRUE(copy.Empty());
}

TEST(MatrixCtors, MoveCtor_EmptyMatrix) {
  Matrix<double> matrix;
  Matrix moved(std::move(matrix));
  EXPECT_EQ(moved.Rows(), 0);
  EXPECT_EQ(moved.Cols(), 0);
  EXPECT_TRUE(moved.Empty());
}

TEST(MatrixCtors, SizeCtor) {
  Matrix<double> matrix(2, 3);
  EXPECT_EQ(matrix.Rows(), 2);
  EXPECT_EQ(matrix.Cols(), 3);
  EXPECT_FALSE(matrix.Empty());
}

TEST(MatrixCtors, InitializerListCtor) {
  Matrix<double> matrix{{1, 2, 3}, {4, 5, 6}};
  EXPECT_EQ(matrix.Rows(), 2);
  EXPECT_EQ(matrix.Cols(), 3);
  EXPECT_FALSE(matrix.Empty());
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
  EXPECT_FALSE(copy.Empty());
  EXPECT_TRUE(copy(0, 0) == 1.0);
  EXPECT_TRUE(copy(0, 1) == 2.0);
  EXPECT_TRUE(copy(0, 2) == 3.0);
  EXPECT_TRUE(copy(1, 0) == 4.0);
  EXPECT_TRUE(copy(1, 1) == 5.0);
  EXPECT_TRUE(copy(1, 2) == 6.0);

  // No side effects on the original matrix.
  EXPECT_EQ(matrix.Rows(), 2);
  EXPECT_EQ(matrix.Cols(), 3);
  EXPECT_FALSE(matrix.Empty());
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
  EXPECT_FALSE(moved.Empty());
  EXPECT_TRUE(moved(0, 0) == 1.0);
  EXPECT_TRUE(moved(0, 1) == 2.0);
  EXPECT_TRUE(moved(0, 2) == 3.0);
  EXPECT_TRUE(moved(1, 0) == 4.0);
  EXPECT_TRUE(moved(1, 1) == 5.0);
  EXPECT_TRUE(moved(1, 2) == 6.0);

  // Data from matrix should be moved.
  EXPECT_TRUE(matrix.Rows() == 0);
  EXPECT_TRUE(matrix.Cols() == 0);
  EXPECT_TRUE(matrix.Empty());
}

TEST(MatrixAssignment, CopyAssignment) {
  Matrix<double> matrix{{1, 2, 3}, {4, 5, 6}};
  Matrix<double> copy;
  copy = matrix;
  EXPECT_EQ(copy.Rows(), 2);
  EXPECT_EQ(copy.Cols(), 3);
  EXPECT_FALSE(copy.Empty());
  EXPECT_TRUE(copy(0, 0) == 1.0);
  EXPECT_TRUE(copy(0, 1) == 2.0);
  EXPECT_TRUE(copy(0, 2) == 3.0);
  EXPECT_TRUE(copy(1, 0) == 4.0);
  EXPECT_TRUE(copy(1, 1) == 5.0);
  EXPECT_TRUE(copy(1, 2) == 6.0);

  // No side effects on the original matrix.
  EXPECT_EQ(matrix.Rows(), 2);
  EXPECT_EQ(matrix.Cols(), 3);
  EXPECT_FALSE(matrix.Empty());
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
  EXPECT_FALSE(moved.Empty());
  EXPECT_TRUE(moved(0, 0) == 1.0);
  EXPECT_TRUE(moved(0, 1) == 2.0);
  EXPECT_TRUE(moved(0, 2) == 3.0);
  EXPECT_TRUE(moved(1, 0) == 4.0);
  EXPECT_TRUE(moved(1, 1) == 5.0);
  EXPECT_TRUE(moved(1, 2) == 6.0);

  // Data from matrix should be moved.
  EXPECT_TRUE(matrix.Rows() == 0);
  EXPECT_TRUE(matrix.Cols() == 0);
  EXPECT_TRUE(matrix.Empty());
}

}  // namespace
