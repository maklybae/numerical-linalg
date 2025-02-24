#ifndef ITERATORS_H
#define ITERATORS_H

#include <iterator>

#include "core_types.h"

namespace linalg {
namespace detail {
namespace iterators {

template <typename T>
concept DefinesPolicy = requires {
  typename T::MyStorageIterator;

  typename T::difference_type;
  typename T::value_type;
  typename T::pointer;
  typename T::reference;
};

template <typename Scalar>
struct DefaultDefines {
  using MyStorageIterator = StorageIterator<Scalar>;

  // NOLINTBEGIN(readability-identifier-naming)
  using difference_type = Difference;
  using value_type      = std::remove_cv_t<Scalar>;
  using pointer         = value_type*;
  using reference       = value_type&;
  // NOLINTEND(readability-identifier-naming)
};

template <typename Scalar>
struct ConstDefines {
  using MyStorageIterator = ConstStorageIterator<Scalar>;

  // NOLINTBEGIN(readability-identifier-naming)
  using difference_type = Difference;
  using value_type      = std::remove_cv_t<Scalar>;
  using pointer         = const value_type*;
  using reference       = const value_type&;
  // NOLINTEND(readability-identifier-naming)
};

template <DefinesPolicy Defines>
class DefaultAccessor : public Defines {
 public:
  using typename Defines::MyStorageIterator;
  using typename Defines::pointer;
  using typename Defines::reference;

  DefaultAccessor() = default;  // to satisfy default constructible iterator concept

  reference operator*() const {
    return *storage_iter_;
  }

  pointer operator->() const {
    if constexpr (std::is_pointer_v<MyStorageIterator>) {
      return storage_iter_;
    }
    return storage_iter_.operator->();
  }

 protected:
  MyStorageIterator storage_iter_;

  explicit DefaultAccessor(MyStorageIterator iter) : storage_iter_{iter} {}
};

// TODO: удалить RandomAccessor, перенести логику [] в moving logic, тк [n] <=> *(it + n)

template <DefinesPolicy Defines>
class RandomAccessor : public DefaultAccessor<Defines> {
 public:
  using Accessor = DefaultAccessor<Defines>;
  using Accessor::Accessor;
  using typename Accessor::difference_type;
  // using typename Accessor::MyStorageIterator;
  // using typename Accessor::pointer;
  using typename Accessor::reference;

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
  using typename Accessor::MyStorageIterator;

  BlockMovingLogic() = default;

  explicit BlockMovingLogic(MyStorageIterator iter) : Accessor{iter} {}

  BlockMovingLogic(MyStorageIterator iter, Size step_size, Size max_step, Difference shift)
      : Accessor{iter}, step_size_{step_size}, max_step_{max_step}, shift_{shift} {}

  BlockMovingLogic& operator++() {
    storage_iter_ += step_size_;
    ++cur_step_;
    if (cur_step_ == max_step_) {
      cur_step_ = 0;
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
    storage_iter_ -= step_size_;
    --cur_step_;
    if (cur_step_ == -1) {
      cur_step_ = max_step_ - 1;
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
  Size step_size_{0};
  Size cur_step_{0};
  Size max_step_{0};
  Difference shift_{0};
};

template <typename Accessor>
class RowMovingLogic : public Accessor {
 public:
  // NOLINTNEXTLINE(readability-identifier-naming)
  using iterator_category = std::contiguous_iterator_tag;
  using typename Accessor::MyStorageIterator;

  RowMovingLogic() = default;

  explicit RowMovingLogic(MyStorageIterator iter) : Accessor{iter} {}

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

}  // namespace iterators
}  // namespace detail
}  // namespace linalg

#endif  // ITERATORS_H
