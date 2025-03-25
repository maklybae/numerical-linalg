#ifndef SCALARS_UTILS_H
#define SCALARS_UTILS_H

#include <complex>
#include <concepts>

#include "scalar_types.h"

namespace linalg::detail {

template <std::floating_point T>
bool ApproxEqual(T lhs, T rhs) {
  return std::abs(lhs - rhs) < std::numeric_limits<T>::epsilon();
}

template <std::floating_point T>
bool ApproxEqual(std::complex<T> lhs, std::complex<T> rhs) {
  return ApproxEqual(lhs.real(), rhs.real()) && ApproxEqual(lhs.imag(), rhs.imag());
}

template <std::floating_point T>
bool ApproxEqual(T lhs, std::complex<T> rhs) {
  return ApproxEqual(lhs, rhs.real()) && ApproxEqual(0.0, rhs.imag());
}

template <std::floating_point T>
bool ApproxEqual(std::complex<T> lhs, T rhs) {
  return ApproxEqual(lhs.real(), rhs) && ApproxEqual(lhs.imag(), 0.0);
}

template <FloatingOrComplexType T>
bool ApproxZero(T val) {
  return ApproxEqual(val, T{0});
}

}  // namespace linalg::detail

#endif  // SCALARS_UTILS_H
