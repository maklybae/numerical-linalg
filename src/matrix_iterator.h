#ifndef MATRIX_ITERATOR_H
#define MATRIX_ITERATOR_H

#include <iterator>
#include <type_traits>

#include "types.h"

namespace linalg::iterators {

// Cannot implement random access iterator
// Unable to define (it1 - it2) operation
template <typename Scalar, typename IsConst>
class BaseMatrixBlockIterator {
 public:
  using Size = types::Size;

  // NOLINTBEGIN(readability-identifier-naming)
  using difference_type   = types::Difference;
  using value_type        = std::remove_cv_t<Scalar>;
  using pointer           = std::conditional_t<IsConst::value, const value_type*, value_type*>;
  using reference         = std::conditional_t<IsConst::value, const value_type&, value_type&>;
  using iterator_category = std::bidirectional_iterator_tag;
  // NOLINTEND(readability-identifier-naming)

  BaseMatrixBlockIterator() = default;

  explicit BaseMatrixBlockIterator(pointer ptr) : ptr_{ptr} {}

  BaseMatrixBlockIterator(pointer ptr, Size cols, Size shift) : ptr_{ptr}, cols_{cols}, shift_{shift} {}

  reference operator*() const {
    return *ptr_;
  }

  pointer operator->() const {
    return ptr_;
  }

  BaseMatrixBlockIterator& operator++() {
    ++ptr_;
    ++col_count_;
    if (col_count_ == cols_) {
      col_count_ = 0;
      ptr_ += shift_;
    }
    return *this;
  }

  BaseMatrixBlockIterator operator++(int) {
    BaseMatrixBlockIterator tmp = *this;
    ++(*this);
    return tmp;
  }

  BaseMatrixBlockIterator& operator--() {
    --ptr_;
    --col_count_;
    if (col_count_ == kNoColsLimit) {
      col_count_ = cols_ - 1;
      ptr_ -= shift_;
    }
    return *this;
  }

  BaseMatrixBlockIterator operator--(int) {
    BaseMatrixBlockIterator tmp = *this;
    --(*this);
    return tmp;
  }

  friend bool operator==(const BaseMatrixBlockIterator& lhs, const BaseMatrixBlockIterator& rhs) {
    return lhs.ptr_ == rhs.ptr_;
  }

 private:
  static constexpr Size kNoColsLimit = std::numeric_limits<Size>::max();

  Scalar* ptr_{nullptr};
  Size cols_{kNoColsLimit};
  Size col_count_{0};
  Size shift_{0};
};

template <typename Scalar>
using MatrixBlockIterator = BaseMatrixBlockIterator<Scalar, std::false_type>;

template <typename Scalar>
using ConstMatrixBlockIterator = BaseMatrixBlockIterator<Scalar, std::true_type>;

}  // namespace linalg::iterators

#endif
