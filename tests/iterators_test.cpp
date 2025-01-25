#include <gtest/gtest.h>

#include <iterator>
#include <type_traits>

#include "../src/matrix_iterator.h"

TEST(MatrixBlockIterator, IteratorConcept) {
  static_assert(std::bidirectional_iterator<linalg::iterators::MatrixBlockIterator<double, std::false_type>>);
}
