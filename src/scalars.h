#ifndef SCALARS_H
#define SCALARS_H

#include <complex>
#include <concepts>

namespace linalg::types {

template <typename T>
struct IsComplex : std::false_type {};

template <std::floating_point T>
struct IsComplex<std::complex<T>> : std::true_type {};

template <typename T>
constexpr bool kIsComplexV = IsComplex<T>::value;

template <typename T>
concept FloatingOrComplexType = std::is_floating_point_v<T> || kIsComplexV<T>;

}  // namespace linalg::types

namespace linalg::scalar_utils {

template <std::floating_point T>
bool ApproxEqual(T lhs, T rhs) {
  return std::abs(lhs - rhs) < std::numeric_limits<T>::epsilon();
}

template <std::floating_point T>
bool ApproxEqual(std::complex<T> lhs, std::complex<T> rhs) {
  return ApproxEqual(lhs.real(), rhs.real()) && ApproxEqual(lhs.imag(), rhs.imag());
}

}  // namespace linalg::scalar_utils

#endif
