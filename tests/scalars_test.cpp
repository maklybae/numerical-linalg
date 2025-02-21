#include "../src/scalars.h"

#include <gtest/gtest.h>

#include <complex>
#include <concepts>
#include <type_traits>
#include <vector>

#include "../src/types_details.h"

namespace {

TEST(ScalarTypesConcepts, StaticAssertions) {
  static_assert(linalg::types::FloatingOrComplexType<const double>);
  static_assert(linalg::types::FloatingOrComplexType<const std::complex<const double>>);

  static_assert(linalg::types::FloatingOrComplexType<float>);
  static_assert(linalg::types::FloatingOrComplexType<double>);
  static_assert(linalg::types::FloatingOrComplexType<long double>);
  static_assert(linalg::types::FloatingOrComplexType<std::complex<float>>);
  static_assert(linalg::types::FloatingOrComplexType<std::complex<double>>);
  static_assert(linalg::types::FloatingOrComplexType<std::complex<long double>>);
}

TEST(ScalarUtils, ApproxEqualOneType) {
  // Exact equality.
  EXPECT_TRUE(linalg::scalar_utils::ApproxEqual(1.0, 1.0));
  EXPECT_TRUE(linalg::scalar_utils::ApproxEqual(1.0f, 1.0f));
  EXPECT_TRUE(linalg::scalar_utils::ApproxEqual(1.0L, 1.0L));
  EXPECT_TRUE(linalg::scalar_utils::ApproxEqual(std::complex{1.0, 1.0}, std::complex{1.0, 1.0}));
  EXPECT_TRUE(linalg::scalar_utils::ApproxEqual(std::complex{1.0f, 1.0f}, std::complex{1.0f, 1.0f}));
  EXPECT_TRUE(linalg::scalar_utils::ApproxEqual(std::complex{1.0L, 1.0L}, std::complex{1.0L, 1.0L}));

  // Approximate equality.
  EXPECT_TRUE(linalg::scalar_utils::ApproxEqual(1.0, 1.0 + std::numeric_limits<double>::epsilon() / 2));
  EXPECT_TRUE(linalg::scalar_utils::ApproxEqual(1.0f, 1.0f + std::numeric_limits<float>::epsilon() / 2));
  EXPECT_TRUE(linalg::scalar_utils::ApproxEqual(1.0L, 1.0L + std::numeric_limits<long double>::epsilon() / 2));
  EXPECT_TRUE(linalg::scalar_utils::ApproxEqual(std::complex{1.0, 1.0},
                                                std::complex{1.0 + std::numeric_limits<double>::epsilon() / 2, 1.0}));
  EXPECT_TRUE(linalg::scalar_utils::ApproxEqual(std::complex{1.0f, 1.0f},
                                                std::complex{1.0f + std::numeric_limits<float>::epsilon() / 2, 1.0f}));
  EXPECT_TRUE(linalg::scalar_utils::ApproxEqual(
      std::complex{1.0L, 1.0L}, std::complex{1.0L + std::numeric_limits<long double>::epsilon() / 2, 1.0L}));
}

TEST(ComplexOperations, ImplicitCast) {
  // std::complex<double> a{1, 0};
  // std::complex<double> b{2, 0};

  // long double c{3};

  // auto res = c + b;

  std::vector<const int> a{};
  std::vector<const int> b(a);
  static_assert(std::same_as<std::vector<const int>::value_type, const int>);

  static_assert(linalg::types::details::kIsComplexV<const std::complex<double>>);
  static_assert(
      std::is_same_v<std::common_type_t<std::complex<const double>, std::complex<double>>, std::complex<const double>>);
  static_assert(
      std::is_same_v<std::common_type_t<const double, std::complex<const double>>, std::complex<const double>>);
}

TEST(ScalarUtils, ApproxEqualDifferentTypes) {
  EXPECT_TRUE(1.0 == 1.0f);
  EXPECT_TRUE(1.0 == 1.0L);
  EXPECT_TRUE(1.0f == 1.0L);
  // EXPECT_TRUE(linalg::scalar_utils::ApproxEqual(1.0, 1.0f));
}

}  // namespace
