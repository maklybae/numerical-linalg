#ifndef ITERATOR_HELPER_H
#define ITERATOR_HELPER_H

#include "matrix.h"
#include "types.h"

namespace linalg::iterators {

template <typename T>
concept DefinesPolicy = requires {
  typename T::StorageIterator;
  typename T::difference_type;
  typename T::value_type;
  typename T::pointer;
  typename T::reference;
};

template <typename Scalar>
struct Defines {
  using StorageIterator = typename Matrix<Scalar>::iterator;

  // NOLINTBEGIN(readability-identifier-naming)
  using difference_type = types::Difference;
  using value_type      = std::remove_cv_t<Scalar>;
  using pointer         = value_type*;
  using reference       = value_type&;
  // NOLINTEND(readability-identifier-naming)
};

template <typename Scalar>
struct ConstDefines {
  using StorageIterator = typename Matrix<Scalar>::const_iterator;

  // NOLINTBEGIN(readability-identifier-naming)
  using difference_type = types::Difference;
  using value_type      = std::remove_cv_t<Scalar>;
  using pointer         = const value_type*;
  using reference       = const value_type&;
  // NOLINTEND(readability-identifier-naming)
};

template <DefinesPolicy Defines>
class Accessor : public Defines {
 public:
  using typename Defines::pointer;
  using typename Defines::reference;
  using typename Defines::StorageIterator;

  Accessor() = default;

  reference operator*() const {
    return *storage_iter_;
  }

  pointer operator->() const {
    if constexpr (std::is_pointer_v<pointer>) {
      return storage_iter_;
    }
    return storage_iter_.operator->();
  }

 protected:
  StorageIterator storage_iter_;

  explicit Accessor(StorageIterator iter) : storage_iter_{iter} {}
};

template <typename Accessor>
class BlockMovingLogic : public Accessor {
 public:
  // NOLINTNEXTLINE(readability-identifier-naming)
  using iterator_category = std::bidirectional_iterator_tag;
  using typename Accessor::StorageIterator;
  using Size = types::Size;

  BlockMovingLogic() = default;

  explicit BlockMovingLogic(StorageIterator iter) : Accessor{iter} {}

  BlockMovingLogic(StorageIterator iter, Size cols, Size shift) : Accessor{iter}, cols_{cols}, shift_{shift} {}

  BlockMovingLogic& operator++() {
    ++storage_iter_;
    ++col_count_;
    if (col_count_ == cols_) {
      col_count_ = 0;
      storage_iter_ += shift_;
    }
    return *this;
  }

  BlockMovingLogic operator++(int) {
    BlockMovingLogic tmp = *this;
    ++(*this);
    return tmp;
  }

  BlockMovingLogic& operator--() {
    --storage_iter_;
    --col_count_;
    if (col_count_ == -1) {
      col_count_ = cols_ - 1;
      storage_iter_ -= shift_;
    }
    return *this;
  }

  BlockMovingLogic operator--(int) {
    BlockMovingLogic tmp = *this;
    --(*this);
    return tmp;
  }

  friend bool operator==(const BlockMovingLogic& lhs, const BlockMovingLogic& rhs) {
    return lhs.storage_iter_ == rhs.storage_iter_;
  }

  friend bool operator!=(const BlockMovingLogic& lhs, const BlockMovingLogic& rhs) {
    return !(lhs == rhs);
  }

 private:
  using Accessor::storage_iter_;
  static constexpr Size kNoColsLimit = std::numeric_limits<Size>::max();
  Size cols_{0};
  Size shift_{0};
  Size col_count_{0};
};

template <typename Scalar>
using DebugBlockIterator = BlockMovingLogic<Accessor<Defines<Scalar>>>;

template <typename Scalar>
using DebugConstBlockIterator = BlockMovingLogic<Accessor<ConstDefines<Scalar>>>;

}  // namespace linalg::iterators

#endif
