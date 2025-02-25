#ifndef SCALAR_TYPES_H
#define SCALAR_TYPES_H

#include <complex>
#include <concepts>
#include <type_traits>

namespace linalg {
namespace detail {

template <typename T>
struct IsComplex : std::false_type {};

template <std::floating_point T>
struct IsComplex<std::complex<T>> : std::true_type {};

template <std::floating_point T>
struct IsComplex<const std::complex<T>> : std::true_type {};

template <typename T>
constexpr bool kIsComplexV = IsComplex<T>::value;

template <typename T>
concept FloatingOrComplexType = std::is_floating_point_v<T> || kIsComplexV<T>;

}  // namespace detail
}  // namespace linalg

#endif  // SCALAR_TYPES_H
