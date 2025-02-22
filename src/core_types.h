#ifndef CORE_TYPES_H
#define CORE_TYPES_H

#include <cstdint>
#include <vector>

namespace linalg {
namespace detail {

template <typename T>
using Storage = std::vector<T>;

template <typename T>
using StorageIterator = typename Storage<T>::iterator;

template <typename T>
using ConstStorageIterator = typename Storage<T>::const_iterator;

enum class ConstnessEnum { kConst, kNonConst };

}  // namespace detail

using Size       = std::int64_t;
using Index      = std::int64_t;
using Difference = std::int64_t;

enum ERows : Size {};
enum ECols : Size {};
enum ERowBegin : Index {};
enum ERowEnd : Index {};
enum EColBegin : Index {};
enum EColEnd : Index {};

}  // namespace linalg

#endif  // CORE_TYPES_H
