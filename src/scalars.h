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

#endif
