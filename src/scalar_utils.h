#ifndef SCALARS_UTILS_H
#define SCALARS_UTILS_H

#include <cmath>
#include <complex>
#include <concepts>

#include "scalar_types.h"

namespace linalg::detail {

template <std::floating_point T>
bool ApproxEqual(T lhs, T rhs, T epsilon = kEpsilon<T>) {
  return std::abs(lhs - rhs) < epsilon;
}

template <std::floating_point T>
bool ApproxEqual(std::complex<T> lhs, std::complex<T> rhs, T epsilon = kEpsilon<T>) {
  return ApproxEqual(lhs.real(), rhs.real(), epsilon) && ApproxEqual(lhs.imag(), rhs.imag(), epsilon);
}

template <std::floating_point T>
bool ApproxEqual(T lhs, std::complex<T> rhs, T epsilon = kEpsilon<T>) {
  return ApproxEqual(lhs, rhs.real(), epsilon) && ApproxZero(rhs.imag(), epsilon);
}

template <std::floating_point T>
bool ApproxEqual(std::complex<T> lhs, T rhs, T epsilon = kEpsilon<T>) {
  return ApproxEqual(lhs.real(), rhs) && ApproxZero(lhs.imag(), epsilon);
}

template <FloatingOrComplexType T>
bool ApproxZero(T val, UnderlyingScalarT<T> epsilon = kEpsilon<UnderlyingScalarT<T>>) {
  return ApproxEqual(val, T{0}, epsilon);
}

template <FloatingOrComplexType T>
T Sign(T val) {
  using ReturnType = UnderlyingScalarT<T>;

  if constexpr (std::is_floating_point_v<T>) {
    return val < T{0} ? ReturnType{-1} : ReturnType{1};
  } else {
    if (ApproxZero(val)) {
      return ReturnType{1};
    }

    return val / std::sqrt(std::norm(val));
  }
}

}  // namespace linalg::detail

#endif  // SCALARS_UTILS_H
