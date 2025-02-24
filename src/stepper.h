#ifndef STEPPER_H
#define STEPPER_H

#include <cassert>
#include <compare>

#include "core_types.h"

namespace linalg {
namespace detail {

class Stepper {
 public:
  explicit Stepper(Size modulo) : modulo_{modulo} {
    assert(modulo_ > 0 && "Modulo should be positive");
  }

  Stepper& operator+=(Difference steps) {
    step_ += steps;
    iteration_ += step_ / modulo_;
    step_ %= modulo_;
    return *this;
  }

  Stepper& operator-=(Difference steps) {
    step_ -= steps;
    if (step_ < 0) {
      iteration_ -= (-step_ + modulo_ - 1) / modulo_;
      step_ = (step_ % modulo_ + modulo_) % modulo_;
    }
    return *this;
  }

  Stepper& operator++() {
    return *this += 1;
  }

  Stepper operator++(int) {
    Stepper tmp = *this;
    ++(*this);
    return tmp;
  }

  Stepper& operator--() {
    return *this -= 1;
  }

  Stepper operator--(int) {
    Stepper tmp = *this;
    --(*this);
    return tmp;
  }

  Difference GetStep() const {
    return step_;
  }

  Difference GetIteration() const {
    return iteration_;
  }

  friend std::partial_ordering operator<=>(const Stepper& lhs, const Stepper& rhs) {
    if (lhs.modulo_ != rhs.modulo_) {
      return std::partial_ordering::unordered;
    }
    if (auto c = lhs.iteration_ <=> rhs.iteration_; c != 0) {
      return c;
    }
    return lhs.step_ <=> rhs.step_;
  }

  friend bool operator==(const Stepper& lhs, const Stepper& rhs) {
    return lhs <=> rhs == 0;
  }

 private:
  Difference step_{};
  Difference iteration_{};
  Size modulo_{1};
};

}  // namespace detail
}  // namespace linalg

#endif  // STEPPER_H
