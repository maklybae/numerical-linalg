#include <gtest/gtest.h>

#include <iterator>
#include <vector>

#include "../src/matrix.h"
#include "../src/matrix_iterator.h"
#include "../src/matrix_view.h"

namespace {

using linalg::ConstMatrixView;
using linalg::Matrix;
using linalg::MatrixView;
using linalg::iterators::ConstMatrixBlockIterator;
using linalg::iterators::MatrixBlockIterator;
using linalg::types::SubmatrixRange;

TEST(MatrixBlockIterator, IteratorConcept) {
  static_assert(std::bidirectional_iterator<MatrixBlockIterator<double>>);
  static_assert(std::bidirectional_iterator<MatrixBlockIterator<const double>>);

  static_assert(std::bidirectional_iterator<ConstMatrixBlockIterator<double>>);
  static_assert(std::bidirectional_iterator<ConstMatrixBlockIterator<const double>>);
}

TEST(MatrixBlockIterator, ConstMatrixView) {
  const Matrix<double> matrix{{1, 2, 3, 4}, {5, 6, 7, 8}, {9, 10, 11, 12}};
  ConstMatrixView<double> view{matrix};

  static_assert(std::is_same_v<decltype(view)::iterator, ConstMatrixBlockIterator<double>>);
  static_assert(std::is_same_v<decltype(view)::const_iterator, ConstMatrixBlockIterator<double>>);
}

TEST(MatrixBlockIterator, MatrixView) {
  Matrix<double> matrix{{1, 2, 3, 4}, {5, 6, 7, 8}, {9, 10, 11, 12}};
  MatrixView<double> view{matrix};

  static_assert(std::is_same_v<decltype(view)::iterator, MatrixBlockIterator<double>>);
  static_assert(std::is_same_v<decltype(view)::const_iterator, ConstMatrixBlockIterator<double>>);
}

TEST(MatrixViewIterator, FullMatrixIterate) {
  Matrix<double> matrix{{1, 2, 3, 4}, {5, 6, 7, 8}, {9, 10, 11, 12}};
  MatrixView<double> view(matrix, SubmatrixRange::FullMatrix(3, 4));
  std::vector<double> expected{1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12};
  std::vector<double> actual(view.begin(), view.end());

  EXPECT_EQ(actual, expected);
}

TEST(MatrixViewIterator, LeadingSubmatrix) {
  Matrix<double> matrix{{1, 2, 3, 4, 5},      {6, 7, 8, 9, 10},     {11, 12, 13, 14, 15},
                        {16, 17, 18, 19, 20}, {21, 22, 23, 24, 25}, {26, 27, 28, 29, 30}};
  MatrixView<double> view(matrix, SubmatrixRange::LeadingSubmatrix(4, 3));
  std::vector<double> expected{1, 2, 3, 6, 7, 8, 11, 12, 13, 16, 17, 18};
  std::vector<double> actual(view.begin(), view.end());

  EXPECT_EQ(actual, expected);
}

TEST(MatrixViewIterator, FromBeginEnd) {
  Matrix<double> matrix{{1, 2, 3, 4, 5},      {6, 7, 8, 9, 10},     {11, 12, 13, 14, 15},
                        {16, 17, 18, 19, 20}, {21, 22, 23, 24, 25}, {26, 27, 28, 29, 30}};
  MatrixView<double> view(matrix, SubmatrixRange::FromBeginEnd(1, 3, 1, 4));
  std::vector<double> expected{7, 8, 9, 12, 13, 14};
  std::vector<double> actual(view.begin(), view.end());

  EXPECT_EQ(actual, expected);
}

TEST(MatrixViewIterator, FromBeginSize) {
  Matrix<double> matrix{{1, 2, 3, 4, 5},      {6, 7, 8, 9, 10},     {11, 12, 13, 14, 15},
                        {16, 17, 18, 19, 20}, {21, 22, 23, 24, 25}, {26, 27, 28, 29, 30}};
  MatrixView<double> view(matrix, SubmatrixRange::FromBeginSize(1, 3, 1, 4));
  std::vector<double> expected{7, 8, 9, 10, 12, 13, 14, 15, 17, 18, 19, 20};
  std::vector<double> actual(view.begin(), view.end());

  EXPECT_EQ(actual, expected);
}

}  // namespace
