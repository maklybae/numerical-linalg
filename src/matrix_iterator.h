#ifndef MATRIX_ITERATOR_H
#define MATRIX_ITERATOR_H

#include <iterator>
#include <type_traits>

#include "matrix.h"
#include "types.h"

namespace linalg::iterators {

// Cannot implement random access iterator
// Unable to define (it1 - it2) operation
template <typename Scalar, typename IsConst>
class BaseMatrixBlockIterator {
 public:
  using Size       = types::Size;
  using Difference = types::Difference;
  using StorageIterator =
      std::conditional_t<IsConst::value, typename Matrix<Scalar>::const_iterator, typename Matrix<Scalar>::iterator>;

  // NOLINTBEGIN(readability-identifier-naming)
  using difference_type   = types::Difference;
  using value_type        = std::remove_cv_t<Scalar>;
  using pointer           = std::conditional_t<IsConst::value, const value_type*, value_type*>;
  using reference         = std::conditional_t<IsConst::value, const value_type&, value_type&>;
  using iterator_category = std::bidirectional_iterator_tag;
  // NOLINTEND(readability-identifier-naming)

  BaseMatrixBlockIterator() = default;

  explicit BaseMatrixBlockIterator(StorageIterator ptr) : storage_iter_{ptr} {}

  BaseMatrixBlockIterator(StorageIterator ptr, Size cols, Size shift)
      : storage_iter_{ptr}, cols_{cols}, shift_{shift} {}

  reference operator*() const {
    return *storage_iter_;
  }

  pointer operator->() const {
    return storage_iter_;
  }

  BaseMatrixBlockIterator& operator++() {
    ++storage_iter_;
    ++col_count_;
    if (col_count_ == cols_) {
      col_count_ = 0;
      storage_iter_ += static_cast<Difference>(shift_);
    }
    return *this;
  }

  BaseMatrixBlockIterator operator++(int) {
    BaseMatrixBlockIterator tmp = *this;
    ++(*this);
    return tmp;
  }

  BaseMatrixBlockIterator& operator--() {
    --storage_iter_;
    --col_count_;
    if (col_count_ == kNoColsLimit) {
      col_count_ = cols_ - 1;
      storage_iter_ -= static_cast<Difference>(shift_);
    }
    return *this;
  }

  BaseMatrixBlockIterator operator--(int) {
    BaseMatrixBlockIterator tmp = *this;
    --(*this);
    return tmp;
  }

  friend bool operator==(const BaseMatrixBlockIterator& lhs, const BaseMatrixBlockIterator& rhs) {
    return lhs.storage_iter_ == rhs.storage_iter_;
  }

 private:
  static constexpr Size kNoColsLimit = std::numeric_limits<Size>::max();

  StorageIterator storage_iter_{};
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
