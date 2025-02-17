#ifndef ITERATOR_HELPER_H
#define ITERATOR_HELPER_H

#include <compare>
#include <iterator>
#include <type_traits>

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
struct DefaultDefines {
  using StorageIterator = typename types::StorageIterator<Scalar>;

  // NOLINTBEGIN(readability-identifier-naming)
  using difference_type = types::Difference;
  using value_type      = std::remove_cv_t<Scalar>;
  using pointer         = value_type*;
  using reference       = value_type&;
  // NOLINTEND(readability-identifier-naming)
};

template <typename Scalar>
struct ConstDefines {
  using StorageIterator = typename types::ConstStorageIterator<Scalar>;

  // NOLINTBEGIN(readability-identifier-naming)
  using difference_type = types::Difference;
  using value_type      = std::remove_cv_t<Scalar>;
  using pointer         = const value_type*;
  using reference       = const value_type&;
  // NOLINTEND(readability-identifier-naming)
};

template <DefinesPolicy Defines>
class DefaultAccessor : public Defines {
 public:
  using typename Defines::pointer;
  using typename Defines::reference;
  using typename Defines::StorageIterator;

  DefaultAccessor() = default;  // to satisfy default constructible iterator concept

  reference operator*() const {
    return *storage_iter_;
  }

  pointer operator->() const {
    if constexpr (std::is_pointer_v<StorageIterator>) {
      return storage_iter_;
    }
    return storage_iter_.operator->();
  }

 protected:
  StorageIterator storage_iter_;

  explicit DefaultAccessor(StorageIterator iter) : storage_iter_{iter} {}
};

template <DefinesPolicy Defines>
class RandomAccessor : public DefaultAccessor<Defines> {
 public:
  using Accessor = DefaultAccessor<Defines>;
  using Accessor::Accessor;
  using typename Accessor::difference_type;
  using typename Accessor::pointer;
  using typename Accessor::reference;
  using typename Accessor::StorageIterator;

  reference operator[](difference_type n) const {
    return storage_iter_[n];
  }

 protected:
  using Accessor::storage_iter_;
};

// TODO: Add AccessorPolicy concept

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

template <typename Accessor>
class RowMovingLogic : public Accessor {
 public:
  // NOLINTNEXTLINE(readability-identifier-naming)
  using iterator_category = std::contiguous_iterator_tag;
  using typename Accessor::StorageIterator;
  using Size       = types::Size;
  using Difference = types::Difference;

  RowMovingLogic() = default;

  explicit RowMovingLogic(StorageIterator iter) : Accessor{iter} {}

  RowMovingLogic& operator++() {
    ++storage_iter_;
    return *this;
  }

  RowMovingLogic operator++(int) {
    RowMovingLogic tmp = *this;
    ++(*this);
    return tmp;
  }

  RowMovingLogic& operator--() {
    --storage_iter_;
    return *this;
  }

  RowMovingLogic operator--(int) {
    RowMovingLogic tmp = *this;
    --(*this);
    return tmp;
  }

  RowMovingLogic& operator+=(Size n) {
    storage_iter_ += n;
    return *this;
  }

  friend RowMovingLogic operator+(RowMovingLogic lhs, Size n) {
    lhs += n;
    return lhs;
  }

  friend RowMovingLogic operator+(Size n, RowMovingLogic rhs) {
    rhs += n;
    return rhs;
  }

  RowMovingLogic& operator-=(Size n) {
    storage_iter_ -= n;
    return *this;
  }

  friend RowMovingLogic operator-(RowMovingLogic lhs, Size n) {
    lhs -= n;
    return lhs;
  }

  friend Difference operator-(const RowMovingLogic& lhs, const RowMovingLogic& rhs) {
    return lhs.storage_iter_ - rhs.storage_iter_;
  }

  friend std::strong_ordering operator<=>(const RowMovingLogic& lhs, const RowMovingLogic& rhs) {
    return lhs.storage_iter_ <=> rhs.storage_iter_;
  }

  friend bool operator==(const RowMovingLogic& lhs, const RowMovingLogic& rhs) {
    return lhs.storage_iter_ == rhs.storage_iter_;
  }

  friend bool operator!=(const RowMovingLogic& lhs, const RowMovingLogic& rhs) {
    return !(lhs == rhs);
  }

 private:
  using Accessor::storage_iter_;
};
}  // namespace linalg::iterators

#endif
