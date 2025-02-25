#ifndef ITERATORS_H
#define ITERATORS_H

#include <iterator>
#include <type_traits>

#include "core_types.h"
#include "matrix_types.h"
#include "scalar_types.h"

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
class BlockIteratorLogic : public Accessor {
 public:
  // NOLINTNEXTLINE(readability-identifier-naming)
  using iterator_category = std::bidirectional_iterator_tag;
  using typename Accessor::MyStorageIterator;
  using typename Accessor::value_type;

  template <typename OtherAccessor>
  friend class BlockIteratorLogic;

  template <typename, ConstnessEnum>
  friend class linalg::detail::BaseMatrixView;

  template <detail::FloatingOrComplexType>
  friend class linalg::Matrix;

  BlockIteratorLogic() = default;

  template <typename OtherBlockIteratorLogic>
    requires std::is_same_v<value_type, typename OtherBlockIteratorLogic::value_type> &&
                 std::is_base_of_v<ConstDefines<value_type>, BlockIteratorLogic> &&
                 std::is_base_of_v<DefaultDefines<value_type>, OtherBlockIteratorLogic>
  // NOLINTNEXTLINE(google-explicit-constructor)
  BlockIteratorLogic(const OtherBlockIteratorLogic& other)
      : Accessor{other.storage_iter_}
      , step_size_{other.step_size_}
      , max_step_{other.max_step_}
      , shift_{other.shift_}
      , threshold_{other.threshold_}
      , cur_step_{other.cur_step_} {}

  BlockIteratorLogic& operator++() {
    storage_iter_ += step_size_;
    ++cur_step_;
    if ((threshold_ == kDefaultThreshold || storage_iter_ < threshold_) && cur_step_ >= max_step_) {
      cur_step_ = 0;
      storage_iter_ += shift_;
    }
    return *this;
  }

  BlockIteratorLogic operator++(int) {
    BlockIteratorLogic tmp = *this;
    ++(*this);
    return tmp;
  }

  BlockIteratorLogic& operator--() {
    storage_iter_ -= step_size_;
    --cur_step_;
    if (cur_step_ < 0) {
      cur_step_ = max_step_ - 1;
      storage_iter_ -= shift_;
    }
    return *this;
  }

  BlockIteratorLogic operator--(int) {
    BlockIteratorLogic tmp = *this;
    --(*this);
    return tmp;
  }

  friend bool operator==(const BlockIteratorLogic& lhs, const BlockIteratorLogic& rhs) {
    return lhs.storage_iter_ == rhs.storage_iter_;
  }

  friend bool operator!=(const BlockIteratorLogic& lhs, const BlockIteratorLogic& rhs) {
    return !(lhs == rhs);
  }

 private:
  static constexpr MyStorageIterator kDefaultThreshold = MyStorageIterator{};

  BlockIteratorLogic(MyStorageIterator iter, Size step_size, Size max_step, Difference shift,
                     MyStorageIterator threshold = kDefaultThreshold, Size cur_step = 0)
      : Accessor{iter}
      , step_size_{step_size}
      , max_step_{max_step}
      , shift_{shift}
      , threshold_{threshold}
      , cur_step_{cur_step} {}

  using Accessor::storage_iter_;
  Size step_size_{0};
  Size max_step_{0};
  Difference shift_{0};
  MyStorageIterator threshold_{kDefaultThreshold};
  Size cur_step_{0};
};

template <typename Accessor>
class RowIteratorLogic : public Accessor {
  using Accessor::storage_iter_;

 public:
  // NOLINTNEXTLINE(readability-identifier-naming)
  using iterator_category = std::contiguous_iterator_tag;
  using typename Accessor::difference_type;
  using typename Accessor::MyStorageIterator;
  using typename Accessor::reference;
  using typename Accessor::value_type;

  template <typename OtherAccessor>
  friend class RowIteratorLogic;

  template <detail::FloatingOrComplexType>
  friend class linalg::Matrix;

  RowIteratorLogic() = default;

  template <typename OtherRowIteratorLogic>
    requires std::is_same_v<value_type, typename OtherRowIteratorLogic::value_type> &&
             std::is_base_of_v<ConstDefines<value_type>, RowIteratorLogic> &&
             std::is_base_of_v<DefaultDefines<value_type>, OtherRowIteratorLogic>
  // NOLINTNEXTLINE(google-explicit-constructor)
  RowIteratorLogic(const OtherRowIteratorLogic& other) : Accessor{other.storage_iter_} {}

  RowIteratorLogic& operator++() {
    ++storage_iter_;
    return *this;
  }

  RowIteratorLogic operator++(int) {
    RowIteratorLogic tmp = *this;
    ++(*this);
    return tmp;
  }

  RowIteratorLogic& operator--() {
    --storage_iter_;
    return *this;
  }

  RowIteratorLogic operator--(int) {
    RowIteratorLogic tmp = *this;
    --(*this);
    return tmp;
  }

  RowIteratorLogic& operator+=(Size n) {
    storage_iter_ += n;
    return *this;
  }

  friend RowIteratorLogic operator+(RowIteratorLogic lhs, Size n) {
    lhs += n;
    return lhs;
  }

  friend RowIteratorLogic operator+(Size n, RowIteratorLogic rhs) {
    rhs += n;
    return rhs;
  }

  RowIteratorLogic& operator-=(Size n) {
    storage_iter_ -= n;
    return *this;
  }

  friend RowIteratorLogic operator-(RowIteratorLogic lhs, Size n) {
    lhs -= n;
    return lhs;
  }

  friend Difference operator-(const RowIteratorLogic& lhs, const RowIteratorLogic& rhs) {
    return lhs.storage_iter_ - rhs.storage_iter_;
  }

  friend std::strong_ordering operator<=>(const RowIteratorLogic& lhs, const RowIteratorLogic& rhs) {
    return lhs.storage_iter_ <=> rhs.storage_iter_;
  }

  friend bool operator==(const RowIteratorLogic& lhs, const RowIteratorLogic& rhs) {
    return lhs.storage_iter_ == rhs.storage_iter_;
  }

  friend bool operator!=(const RowIteratorLogic& lhs, const RowIteratorLogic& rhs) {
    return !(lhs == rhs);
  }

  reference operator[](difference_type n) const {
    return *(*this + n);
  }

 private:
  explicit RowIteratorLogic(MyStorageIterator iter) : Accessor{iter} {}
};

template <typename Scalar>
using RowIterator = RowIteratorLogic<DefaultAccessor<DefaultDefines<Scalar>>>;

template <typename Scalar>
using ConstRowIterator = RowIteratorLogic<DefaultAccessor<ConstDefines<Scalar>>>;

template <typename Scalar>
using BlockIterator = BlockIteratorLogic<DefaultAccessor<DefaultDefines<Scalar>>>;

template <typename Scalar>
using ConstBlockIterator = BlockIteratorLogic<DefaultAccessor<ConstDefines<Scalar>>>;

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
