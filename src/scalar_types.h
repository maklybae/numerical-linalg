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

// Issue with single template using and std::conditional_t:
// T::value_type is evaluated for built-in types despite conditional branch.
// This is a workaround.
template <typename T>
struct UnderlyingScalar {
  // NOLINTNEXTLINE(readability-identifier-naming)
  using type = T;
};

template <typename T>
struct UnderlyingScalar<std::complex<T>> {
  // NOLINTNEXTLINE(readability-identifier-naming)
  using type = T;
};

template <typename T>
using UnderlyingScalarT = typename UnderlyingScalar<T>::type;

}  // namespace detail
}  // namespace linalg

#endif  // SCALAR_TYPES_H
