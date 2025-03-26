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

// Limits: need to control epsilon for different types for debugging and testing.

template <FloatingOrComplexType T>
struct NumericLimits {
  static constexpr T kEpsilon = std::numeric_limits<T>::epsilon();
};

template <>
struct NumericLimits<float> {
  static constexpr float kEpsilon = 1e-6f;
};

template <>
struct NumericLimits<double> {
  static constexpr double kEpsilon = 1e-11;
};

template <>
struct NumericLimits<long double> {
  static constexpr long double kEpsilon = 1e-14L;
};

template <FloatingOrComplexType T>
struct NumericLimits<std::complex<T>> {
  static constexpr T kEpsilon = NumericLimits<T>::kEpsilon;
};

template <FloatingOrComplexType T>
constexpr T kEpsilon = NumericLimits<T>::kEpsilon;

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
