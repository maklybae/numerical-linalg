#include "../src/iterators.h"

#include <gtest/gtest.h>

#include <complex>
#include <functional>
#include <iterator>
#include <vector>

#include "linalg.h"

namespace {

using linalg::ConstMatrixView;
using linalg::Matrix;
using linalg::MatrixView;
using linalg::SubmatrixRange;

TEST(MatrixRowIterator, IteratorConcept) {
  static_assert(std::contiguous_iterator<Matrix<double>::iterator>);
  static_assert(std::contiguous_iterator<Matrix<double>::const_iterator>);
  static_assert(std::bidirectional_iterator<Matrix<double>::ColIterator>);
  static_assert(std::bidirectional_iterator<Matrix<double>::ConstColIterator>);

  static_assert(std::bidirectional_iterator<MatrixView<double>::iterator>);
  static_assert(std::bidirectional_iterator<ConstMatrixView<double>::iterator>);
  // TODO: Add separate assertions for MatrixView col/row iterators.
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

  static_assert(std::bidirectional_iterator<ConstMatrixView<double>::iterator>);
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
  MatrixView<double> view(matrix, SubmatrixRange::FullMatrix(linalg::ERows{3}, linalg::ECols{4}));
  std::vector<double> expected{1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12};
  std::vector<double> actual(view.begin(), view.end());

  EXPECT_EQ(actual, expected);
}

TEST(MatrixIterator, FullMatrixIterate) {
  Matrix<double> matrix{{1, 2, 3, 4}, {5, 6, 7, 8}, {9, 10, 11, 12}};
  Matrix<double>::iterator iter{matrix.begin()};
  std::vector<double> expected{1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12};
  std::vector<double> actual(iter, iter + 12);

  EXPECT_EQ(actual, expected);
}

TEST(MatrixIterator, FullMatrixColIterate) {
  Matrix<double> matrix{{1, 2, 3, 4}, {5, 6, 7, 8}, {9, 10, 11, 12}};
  std::vector<double> expected{1, 5, 9, 2, 6, 10, 3, 7, 11, 4, 8, 12};
  std::vector<double> actual(matrix.ColWiseBegin(), matrix.ColWiseEnd());

  EXPECT_EQ(actual, expected);
}

TEST(MatrixIterator, FullMatrixConstColIterate) {
  const Matrix<double> matrix{{1, 2, 3, 4}, {5, 6, 7, 8}, {9, 10, 11, 12}};
  std::vector<double> expected{1, 5, 9, 2, 6, 10, 3, 7, 11, 4, 8, 12};
  std::vector<double> actual(matrix.ColWiseCBegin(), matrix.ColWiseCEnd());

  EXPECT_EQ(actual, expected);
}

TEST(MatrixIterator, FullMatrixReverseColIterate) {
  Matrix<double> matrix{{1, 2, 3, 4}, {5, 6, 7, 8}, {9, 10, 11, 12}};
  std::vector<double> expected{12, 8, 4, 11, 7, 3, 10, 6, 2, 9, 5, 1};
  std::vector<double> actual(matrix.ColWiseRBegin(), matrix.ColWiseREnd());

  EXPECT_EQ(actual, expected);
}

TEST(MatrixViewIterator, LeadingSubmatrix) {
  Matrix<double> matrix{{1, 2, 3, 4, 5},      {6, 7, 8, 9, 10},     {11, 12, 13, 14, 15},
                        {16, 17, 18, 19, 20}, {21, 22, 23, 24, 25}, {26, 27, 28, 29, 30}};
  MatrixView<double> view(matrix, SubmatrixRange::LeadingSubmatrix(linalg::ERows{4}, linalg::ECols{3}));
  std::vector<double> expected{1, 2, 3, 6, 7, 8, 11, 12, 13, 16, 17, 18};
  std::vector<double> actual(view.begin(), view.end());

  EXPECT_EQ(actual, expected);
}

TEST(MatrixViewIterator, FromBeginEnd) {
  Matrix<double> matrix{{1, 2, 3, 4, 5},      {6, 7, 8, 9, 10},     {11, 12, 13, 14, 15},
                        {16, 17, 18, 19, 20}, {21, 22, 23, 24, 25}, {26, 27, 28, 29, 30}};
  MatrixView<double> view(matrix, SubmatrixRange::FromBeginEnd(linalg::ERowBegin{1}, linalg::ERowEnd{3},
                                                               linalg::EColBegin{1}, linalg::EColEnd{4}));
  std::vector<double> expected{7, 8, 9, 12, 13, 14};
  std::vector<double> actual(view.begin(), view.end());

  EXPECT_EQ(actual, expected);
}

TEST(MatrixViewIterator, FromBeginSize) {
  Matrix<double> matrix{{1, 2, 3, 4, 5},      {6, 7, 8, 9, 10},     {11, 12, 13, 14, 15},
                        {16, 17, 18, 19, 20}, {21, 22, 23, 24, 25}, {26, 27, 28, 29, 30}};
  MatrixView<double> view(matrix, SubmatrixRange::FromBeginSize(linalg::ERowBegin{1}, linalg::ERows{3},
                                                                linalg::EColBegin{1}, linalg::ECols{4}));
  std::vector<double> expected{7, 8, 9, 10, 12, 13, 14, 15, 17, 18, 19, 20};
  std::vector<double> actual(view.begin(), view.end());

  EXPECT_EQ(actual, expected);
}

TEST(MatrixViewIterator, FromBeginSizeColWise) {
  Matrix<double> matrix{{1, 2, 3, 4, 5},      {6, 7, 8, 9, 10},     {11, 12, 13, 14, 15},
                        {16, 17, 18, 19, 20}, {21, 22, 23, 24, 25}, {26, 27, 28, 29, 30}};
  MatrixView<double> view(matrix, SubmatrixRange::FromBeginSize(linalg::ERowBegin{1}, linalg::ERows{3},
                                                                linalg::EColBegin{1}, linalg::ECols{4}));
  std::vector<double> expected{7, 12, 17, 8, 13, 18, 9, 14, 19, 10, 15, 20};
  std::vector<double> actual(view.ColWiseBegin(), view.ColWiseEnd());

  EXPECT_EQ(actual, expected);
}

TEST(MatrixViewIterator, FromBeginSizeConstColWise) {
  Matrix<double> matrix{{1, 2, 3, 4, 5},      {6, 7, 8, 9, 10},     {11, 12, 13, 14, 15},
                        {16, 17, 18, 19, 20}, {21, 22, 23, 24, 25}, {26, 27, 28, 29, 30}};
  MatrixView<double> view(matrix, SubmatrixRange::FromBeginSize(linalg::ERowBegin{1}, linalg::ERows{3},
                                                                linalg::EColBegin{1}, linalg::ECols{4}));
  std::vector<double> expected{7, 12, 17, 8, 13, 18, 9, 14, 19, 10, 15, 20};
  std::vector<double> actual(view.ColWiseCBegin(), view.ColWiseCEnd());

  EXPECT_EQ(actual, expected);
}

TEST(MatrixViewIterator, FromBeginSizeConstReverseColWise) {
  Matrix<double> matrix{{1, 2, 3, 4, 5},      {6, 7, 8, 9, 10},     {11, 12, 13, 14, 15},
                        {16, 17, 18, 19, 20}, {21, 22, 23, 24, 25}, {26, 27, 28, 29, 30}};
  MatrixView<double> view(matrix, SubmatrixRange::FromBeginSize(linalg::ERowBegin{1}, linalg::ERows{3},
                                                                linalg::EColBegin{1}, linalg::ECols{4}));
  std::vector<double> expected{20, 15, 10, 19, 14, 9, 18, 13, 8, 17, 12, 7};
  std::vector<double> actual(view.ColWiseCRBegin(), view.ColWiseCREnd());

  EXPECT_EQ(actual, expected);
}

TEST(MatrixView, ImplicitConstCtor) {
  Matrix<double> matrix{{1, 2, 3, 4, 5},      {6, 7, 8, 9, 10},     {11, 12, 13, 14, 15},
                        {16, 17, 18, 19, 20}, {21, 22, 23, 24, 25}, {26, 27, 28, 29, 30}};
  MatrixView<double> view(matrix, SubmatrixRange::FromBeginEnd(linalg::ERowBegin{1}, linalg::ERowEnd{3},
                                                               linalg::EColBegin{1}, linalg::EColEnd{4}));
  ConstMatrixView<double> const_view{view};

  static_assert(std::is_same_v<decltype(const_view)::iterator, ConstMatrixView<double>::iterator>);
  static_assert(std::is_same_v<decltype(const_view)::const_iterator, ConstMatrixView<double>::iterator>);
}

TEST(MatrixView, TemplateApply) {
  Matrix<std::complex<double>> lhs_mat{{1, 2, 3, 4, 5},      {6, 7, 8, 9, 10},     {11, 12, 13, 14, 15},
                                       {16, 17, 18, 19, 20}, {21, 22, 23, 24, 25}, {26, 27, 28, 29, 30}};
  MatrixView<std::complex<double>> lhs_view(
      lhs_mat,
      SubmatrixRange::FromBeginSize(linalg::ERowBegin{0}, linalg::ERows{2}, linalg::EColBegin{0}, linalg::ECols{2}));
  Matrix<double> rhs_mat{{1, 2}, {3, 4}};
  linalg::detail::Apply(lhs_view, rhs_mat, std::plus<>());
}

TEST(IteratorAliases, NonConstToConstCast) {
  linalg::detail::iterators::RowIterator<double> row_iter;
  linalg::detail::iterators::ConstRowIterator<double> const_row_iter{row_iter};

  linalg::detail::iterators::BlockIterator<double> col_block_iter;
  linalg::detail::iterators::ConstBlockIterator<double> const_col_block_iter{col_block_iter};
}

}  // namespace
