#ifndef TYPES_H
#define TYPES_H

#include <complex>
#include <concepts>
#include <cstddef>

namespace linalg::types::details {
template <typename T>
struct IsComplex : std::false_type {};

template <std::floating_point T>
struct IsComplex<std::complex<T>> : std::true_type {};

template <typename T>
constexpr bool kIsComplexV = IsComplex<T>::value;
}  // namespace linalg::types::details

namespace linalg::types {

template <typename T>
concept FloatingOrComplexType = std::is_floating_point_v<T> || details::kIsComplexV<T>;

using Size       = std::ptrdiff_t;
using Index      = std::ptrdiff_t;
using Difference = std::ptrdiff_t;

template <typename T>
using Storage = std::vector<T>;

template <typename T>
using StorageIterator = typename Storage<T>::iterator;

template <typename T>
using ConstStorageIterator = typename Storage<T>::const_iterator;
}  // namespace linalg::types

#endif
