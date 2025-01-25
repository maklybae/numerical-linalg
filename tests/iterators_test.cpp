#include <gtest/gtest.h>

#include <iterator>

#include "../src/matrix_iterator.h"

namespace {

using linalg::iterators::ConstMatrixBlockIterator;
using linalg::iterators::MatrixBlockIterator;

TEST(MatrixBlockIterator, IteratorConcept) {
  static_assert(std::bidirectional_iterator<MatrixBlockIterator<double>>);
  static_assert(std::bidirectional_iterator<MatrixBlockIterator<const double>>);

  static_assert(std::bidirectional_iterator<ConstMatrixBlockIterator<double>>);
  static_assert(std::bidirectional_iterator<ConstMatrixBlockIterator<const double>>);
}

}  // namespace
