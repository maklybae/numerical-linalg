#ifndef MATRIX_H
#define MATRIX_H

#include <complex>
#include <concepts>
#include <type_traits>

namespace linalg {

template <typename T>
struct IsComplex : std::false_type {};

template <std::floating_point T>
struct IsComplex<std::complex<T>> : std::true_type {};

template <typename T>
constexpr bool kIsComplexV = IsComplex<T>::value;

template <typename T>
concept FloatingOrComplexType = std::is_floating_point_v<T> || kIsComplexV<T>;

template <FloatingOrComplexType T>
class Matrix {};
}  // namespace linalg

#endif
