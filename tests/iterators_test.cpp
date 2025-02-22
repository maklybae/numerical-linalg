#include "../src/iterators.h"

#include <gtest/gtest.h>

#include <complex>
#include <functional>
#include <iterator>
#include <vector>

#include "linalg"

namespace {

using linalg::ConstMatrixView;
using linalg::Matrix;
using linalg::MatrixView;
using linalg::SubmatrixRange;

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
  MatrixView<double> view(matrix, SubmatrixRange::FullMatrix(linalg::ERows{3}, linalg::ECols{4}));
  std::vector<double> expected{1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12};
  std::vector<double> actual(view.begin(), view.end());

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

  std::cout << lhs_mat(0, 0) << lhs_mat(0, 1) << lhs_mat(0, 2) << lhs_mat(0, 3) << lhs_mat(0, 4) << lhs_mat(1, 0)
            << lhs_mat(1, 1) << lhs_mat(1, 2) << lhs_mat(1, 3) << lhs_mat(1, 4) << lhs_mat(2, 0) << lhs_mat(2, 1)
            << lhs_mat(2, 2) << lhs_mat(2, 3) << lhs_mat(2, 4) << lhs_mat(3, 0) << lhs_mat(3, 1) << lhs_mat(3, 2)
            << std::endl;
}
}  // namespace
