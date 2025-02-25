#ifndef ITERATORS_H
#define ITERATORS_H

#include <iterator>
#include <type_traits>

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
  // Constraint to manage constness not in iterator but in Matrix types.
  static_assert(!std::is_const_v<Scalar> && !std::is_volatile_v<Scalar>,
                "Scalar type must not be const and not be volatile");

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
  // Constraint to manage constness not in iterator but in Matrix types.
  static_assert(!std::is_const_v<Scalar> && !std::is_volatile_v<Scalar>,
                "Scalar type must not be const and not be volatile");

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

template <typename Accessor>
class RowBlockMovingLogic : public Accessor {
 public:
  // NOLINTNEXTLINE(readability-identifier-naming)
  using iterator_category = std::bidirectional_iterator_tag;
  using typename Accessor::MyStorageIterator;
  using typename Accessor::value_type;

  RowBlockMovingLogic() = default;

  RowBlockMovingLogic(MyStorageIterator iter, Size step_size, Size max_step, Difference shift)
      : Accessor{iter}, step_size_{step_size}, max_step_{max_step}, shift_{shift} {}

  template <typename OtherAccessor>
  friend class RowBlockMovingLogic;

  template <typename OtherRowBlockMovingLogic>
    requires std::is_same_v<value_type, typename OtherRowBlockMovingLogic::value_type> &&
                 std::is_base_of_v<ConstDefines<value_type>, RowBlockMovingLogic> &&
                 std::is_base_of_v<DefaultDefines<value_type>, OtherRowBlockMovingLogic>
  // NOLINTNEXTLINE(google-explicit-constructor)
  explicit RowBlockMovingLogic(const OtherRowBlockMovingLogic& other)
      : Accessor{other.storage_iter_}, step_size_{other.step_size_}, max_step_{other.max_step_}, shift_{other.shift_} {}

  RowBlockMovingLogic& operator++() {
    storage_iter_ += step_size_;
    ++cur_step_;
    if (cur_step_ == max_step_) {
      cur_step_ = 0;
      storage_iter_ += shift_;
    }
    return *this;
  }

  RowBlockMovingLogic operator++(int) {
    RowBlockMovingLogic tmp = *this;
    ++(*this);
    return tmp;
  }

  RowBlockMovingLogic& operator--() {
    storage_iter_ -= step_size_;
    --cur_step_;
    if (cur_step_ == -1) {
      cur_step_ = max_step_ - 1;
      storage_iter_ -= shift_;
    }
    return *this;
  }

  RowBlockMovingLogic operator--(int) {
    RowBlockMovingLogic tmp = *this;
    --(*this);
    return tmp;
  }

  friend bool operator==(const RowBlockMovingLogic& lhs, const RowBlockMovingLogic& rhs) {
    return lhs.storage_iter_ == rhs.storage_iter_;
  }

  friend bool operator!=(const RowBlockMovingLogic& lhs, const RowBlockMovingLogic& rhs) {
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
class ColBlockMovingLogic : public Accessor {
 public:
  // NOLINTNEXTLINE(readability-identifier-naming)
  using iterator_category = std::bidirectional_iterator_tag;
  using typename Accessor::MyStorageIterator;
  using typename Accessor::value_type;

  ColBlockMovingLogic() = default;

  ColBlockMovingLogic(MyStorageIterator iter, MyStorageIterator threshold, Size step_size, Size max_step,
                      Difference shift, Size cur_step = 0)
      : Accessor{iter}
      , threshold_{threshold}
      , step_size_{step_size}
      , max_step_{max_step}
      , shift_{shift}
      , cur_step_{cur_step} {}

  template <typename OtherAccessor>
  friend class ColBlockMovingLogic;

  template <typename OtherColBlockMovingLogic>
    requires std::is_same_v<value_type, typename OtherColBlockMovingLogic::value_type> &&
                 std::is_base_of_v<ConstDefines<value_type>, ColBlockMovingLogic> &&
                 std::is_base_of_v<DefaultDefines<value_type>, OtherColBlockMovingLogic>
  // NOLINTNEXTLINE(google-explicit-constructor)
  explicit ColBlockMovingLogic(const OtherColBlockMovingLogic& other)
      : Accessor{other.storage_iter_}
      , threshold_{other.threshold_}
      , step_size_{other.step_size_}
      , max_step_{other.max_step_}
      , shift_{other.shift_}
      , cur_step_{other.cur_step_} {}

  ColBlockMovingLogic& operator++() {
    storage_iter_ += step_size_;
    ++cur_step_;
    if (storage_iter_ < threshold_ && cur_step_ >= max_step_) {
      cur_step_ = 0;
      storage_iter_ += shift_;
    }
    return *this;
  }

  ColBlockMovingLogic operator++(int) {
    RowBlockMovingLogic tmp = *this;
    ++(*this);
    return tmp;
  }

  ColBlockMovingLogic& operator--() {
    storage_iter_ -= step_size_;
    --cur_step_;
    if (cur_step_ < 0) {
      cur_step_ = max_step_ - 1;
      storage_iter_ -= shift_;
    }
    return *this;
  }

  ColBlockMovingLogic operator--(int) {
    RowBlockMovingLogic tmp = *this;
    --(*this);
    return tmp;
  }

  friend bool operator==(const ColBlockMovingLogic& lhs, const ColBlockMovingLogic& rhs) {
    return lhs.storage_iter_ == rhs.storage_iter_;
  }

  friend bool operator!=(const ColBlockMovingLogic& lhs, const ColBlockMovingLogic& rhs) {
    return !(lhs == rhs);
  }

 private:
  using Accessor::storage_iter_;
  MyStorageIterator threshold_{};
  Size step_size_{0};
  Size max_step_{0};
  Difference shift_{0};
  Size cur_step_{0};
};

template <typename Accessor>
class RowMovingLogic : public Accessor {
  using Accessor::storage_iter_;

 public:
  // NOLINTNEXTLINE(readability-identifier-naming)
  using iterator_category = std::contiguous_iterator_tag;
  using typename Accessor::difference_type;
  using typename Accessor::MyStorageIterator;
  using typename Accessor::reference;
  using typename Accessor::value_type;

  RowMovingLogic() = default;

  explicit RowMovingLogic(MyStorageIterator iter) : Accessor{iter} {}

  template <typename OtherAccessor>
  friend class RowMovingLogic;

  template <typename OtherRowMovingLogic>
    requires std::is_same_v<value_type, typename OtherRowMovingLogic::value_type> &&
             std::is_base_of_v<ConstDefines<value_type>, RowMovingLogic> &&
             std::is_base_of_v<DefaultDefines<value_type>, OtherRowMovingLogic>
  // NOLINTNEXTLINE(google-explicit-constructor)
  explicit RowMovingLogic(const OtherRowMovingLogic& other) : Accessor{other.storage_iter_} {}

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

  reference operator[](difference_type n) const {
    return *(*this + n);
  }
};

template <typename Scalar>
using RowIterator = RowMovingLogic<DefaultAccessor<DefaultDefines<Scalar>>>;

template <typename Scalar>
using ConstRowIterator = RowMovingLogic<DefaultAccessor<ConstDefines<Scalar>>>;

template <typename Scalar>
using ColBlockIterator = ColBlockMovingLogic<DefaultAccessor<DefaultDefines<Scalar>>>;

template <typename Scalar>
using ConstColBlockIterator = ColBlockMovingLogic<DefaultAccessor<ConstDefines<Scalar>>>;

template <typename Scalar>
using RowBlockIterator = RowBlockMovingLogic<DefaultAccessor<DefaultDefines<Scalar>>>;

template <typename Scalar>
using ConstRowBlockIterator = RowBlockMovingLogic<DefaultAccessor<ConstDefines<Scalar>>>;

// EXPERIMENTAL CODE (не смог сделать обертку над всеми итераторами, описал в ПР):

// template <typename Scalar, template <typename> class Defines, template <typename> class Accessor,
//           template <typename> class MovingLogic>
// class Iterator : public MovingLogic<Accessor<Defines<Scalar>>> {
//   using Base = MovingLogic<Accessor<Defines<Scalar>>>;
//   using Base::Base;

//  public:
//   template <typename OtherScalar, template <typename> class OtherDefines, template <typename> class OtherAccessor,
//             template <typename> class OtherMovingLogic>
//     requires std::is_same_v<Scalar, OtherScalar> && std::is_same_v<Defines<Scalar>, ConstDefines<Scalar>> &&
//              std::is_same_v<OtherDefines<Scalar>, DefaultDefines<Scalar>>
//   // NOLINTNEXTLINE(google-explicit-constructor)
//   Iterator(const Iterator<OtherScalar, OtherDefines, OtherAccessor, OtherMovingLogic>& other) : Base{other} {}
// };

// template <typename Scalar>
// using ExperimentalRowBlockIterator = Iterator<Scalar, DefaultDefines, DefaultAccessor, RowBlockMovingLogic>;

// template <typename Scalar>
// using ExperimentalConstRowBlockIterator = Iterator<Scalar, ConstDefines, DefaultAccessor, RowBlockMovingLogic>;

// template <typename Scalar>
// using ExperimentalColBlockIterator = Iterator<Scalar, DefaultDefines, DefaultAccessor, ColBlockMovingLogic>;

// template <typename Scalar>
// using ExperimentalConstColBlockIterator = Iterator<Scalar, ConstDefines, DefaultAccessor, ColBlockMovingLogic>;

// template <typename Scalar>
// using ExperimentalRowIterator = Iterator<Scalar, DefaultDefines, DefaultAccessor, RowMovingLogic>;

// template <typename Scalar>
// using ExperimentalConstRowIterator = Iterator<Scalar, ConstDefines, DefaultAccessor, RowMovingLogic>;

}  // namespace iterators
}  // namespace detail
}  // namespace linalg

#endif  // ITERATORS_H
