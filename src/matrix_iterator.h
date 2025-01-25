#ifndef MATRIX_ITERATOR_H
#define MATRIX_ITERATOR_H

#include <iterator>
#include <type_traits>

#include "types.h"

namespace linalg::iterators {

// Cannot implement random access iterator
// Unable to define (it1 - it2) operation
template <typename T, typename IsConst>
class MatrixBlockIterator {
 public:
  using Size = types::Size;

  // NOLINTBEGIN(readability-identifier-naming)
  using difference_type   = types::Difference;
  using value_type        = std::remove_cv_t<T>;
  using pointer           = std::conditional_t<IsConst::value, const value_type*, value_type*>;
  using reference         = std::conditional_t<IsConst::value, const value_type&, value_type&>;
  using iterator_category = std::bidirectional_iterator_tag;
  // NOLINTEND(readability-identifier-naming)

  MatrixBlockIterator() = default;

  explicit MatrixBlockIterator(pointer ptr) : ptr_{ptr} {}

  MatrixBlockIterator(pointer ptr, Size cols, Size shift) : ptr_{ptr}, cols_{cols}, shift_{shift} {}

  reference operator*() const {
    return *ptr_;
  }

  pointer operator->() const {
    return ptr_;
  }

  MatrixBlockIterator& operator++() {
    ++ptr_;
    ++col_count_;
    if (col_count_ == cols_) {
      col_count_ = 0;
      ptr_ += shift_;
    }
    return *this;
  }

  MatrixBlockIterator operator++(int) {
    MatrixBlockIterator tmp = *this;
    ++(*this);
    return tmp;
  }

  MatrixBlockIterator& operator--() {
    --ptr_;
    --col_count_;
    if (col_count_ == kNoColsLimit) {
      col_count_ = cols_ - 1;
      ptr_ -= shift_;
    }
    return *this;
  }

  MatrixBlockIterator operator--(int) {
    MatrixBlockIterator tmp = *this;
    --(*this);
    return tmp;
  }

  friend bool operator==(const MatrixBlockIterator& lhs, const MatrixBlockIterator& rhs) {
    return lhs.ptr_ == rhs.ptr_;
  }

 private:
  static constexpr Size kNoColsLimit = std::numeric_limits<Size>::max();

  T* ptr_{nullptr};
  Size cols_{kNoColsLimit};
  Size col_count_{0};
  Size shift_{0};
};
}  // namespace linalg::iterators

#endif
