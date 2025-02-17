#include <gtest/gtest.h>

#include <iterator>
#include <vector>

#include "../src/iterator_helper.h"
#include "../src/matrix.h"
#include "../src/matrix_view.h"

namespace {

using linalg::ConstMatrixView;
using linalg::Matrix;
using linalg::MatrixView;
using linalg::types::SubmatrixRange;

TEST(MatrixRowIterator, IteratorConcept) {
  static_assert(std::contiguous_iterator<Matrix<double>::iterator>);
  static_assert(std::contiguous_iterator<Matrix<const double>::iterator>);

  static_assert(std::contiguous_iterator<Matrix<double>::const_iterator>);
  static_assert(std::contiguous_iterator<Matrix<const double>::const_iterator>);
}

TEST(MatrixRowIterator, IterateOverMatrix) {
  Matrix<double> matrix{{1, 2, 3}, {4, 5, 6}, {7, 8, 9}};
  Matrix<double>::iterator iter{matrix.begin()};
  std::vector<double> expected{1, 2, 3, 4, 5, 6, 7, 8, 9};
  std::vector<double> actual(iter, iter + 9);

  EXPECT_EQ(actual, expected);
}

TEST(MatrixRowIterator, IterateOverConstMatrix) {
  const Matrix<double> matrix{{1, 2, 3}, {4, 5, 6}, {7, 8, 9}};
  Matrix<double>::const_iterator iter{matrix.begin()};
  std::vector<double> expected{1, 2, 3, 4, 5, 6, 7, 8, 9};
  std::vector<double> actual(iter, iter + 9);

  EXPECT_EQ(actual, expected);
}

TEST(MatrixBlockIterator, IteratorConcept) {
  static_assert(std::bidirectional_iterator<MatrixView<double>::iterator>);
  static_assert(std::bidirectional_iterator<MatrixView<const double>::iterator>);

  static_assert(std::bidirectional_iterator<ConstMatrixView<double>::iterator>);
  static_assert(std::bidirectional_iterator<ConstMatrixView<const double>::iterator>);
}

TEST(MatrixBlockIterator, ConstMatrixView) {
  const Matrix<double> matrix{{1, 2, 3, 4}, {5, 6, 7, 8}, {9, 10, 11, 12}};
  ConstMatrixView<double> view{matrix};

  static_assert(std::is_same_v<decltype(view)::iterator, ConstMatrixView<double>::iterator>);
  static_assert(std::is_same_v<decltype(view)::const_iterator, ConstMatrixView<double>::iterator>);
}

TEST(MatrixBlockIterator, MatrixView) {
  Matrix<double> matrix{{1, 2, 3, 4}, {5, 6, 7, 8}, {9, 10, 11, 12}};
  MatrixView<double> view{matrix};

  static_assert(std::is_same_v<decltype(view)::iterator, MatrixView<double>::iterator>);
  static_assert(std::is_same_v<decltype(view)::const_iterator, ConstMatrixView<double>::iterator>);
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
