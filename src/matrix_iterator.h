#ifndef MATRIX_ITERATOR_H
#define MATRIX_ITERATOR_H

#include <iterator>
#include <type_traits>

namespace linalg::iterators {

// Cannot implement random access iterator
// Unable to define (it1 - it2) operation
template <typename T, typename IsConst>
class MatrixBlockIterator {
 public:
  // NOLINTBEGIN(readability-identifier-naming)
  using difference_type   = std::ptrdiff_t;
  using value_type        = std::remove_cv_t<T>;
  using pointer           = std::conditional_t<IsConst::value, const value_type*, value_type*>;
  using reference         = std::conditional_t<IsConst::value, const value_type&, value_type&>;
  using iterator_category = std::bidirectional_iterator_tag;
  // NOLINTEND(readability-identifier-naming)

  MatrixBlockIterator() = default;

  explicit MatrixBlockIterator(pointer ptr) : ptr_{ptr} {}

  MatrixBlockIterator(pointer ptr, size_t cols, size_t shift) : ptr_{ptr}, cols_{cols}, shift_{shift} {}

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
  static constexpr size_t kNoColsLimit = std::numeric_limits<size_t>::max();

  T* ptr_{nullptr};
  size_t cols_{kNoColsLimit};
  size_t col_count_{0};
  size_t shift_{0};
};
}  // namespace linalg::iterators

#endif
