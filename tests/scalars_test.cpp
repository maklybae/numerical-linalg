#include <gtest/gtest.h>

#include <complex>
#include <type_traits>

#include "linalg.h"

namespace {

TEST(ScalarTypesConcepts, StaticAssertions) {
  static_assert(linalg::detail::FloatingOrComplexType<const double>);
  static_assert(linalg::detail::FloatingOrComplexType<const std::complex<const double>>);

  static_assert(linalg::detail::FloatingOrComplexType<float>);
  static_assert(linalg::detail::FloatingOrComplexType<double>);
  static_assert(linalg::detail::FloatingOrComplexType<long double>);
  static_assert(linalg::detail::FloatingOrComplexType<std::complex<float>>);
  static_assert(linalg::detail::FloatingOrComplexType<std::complex<double>>);
  static_assert(linalg::detail::FloatingOrComplexType<std::complex<long double>>);
}

TEST(ScalarUtils, ApproxEqualOneType) {
  // Exact equality.
  EXPECT_TRUE(linalg::detail::ApproxEqual(1.0, 1.0));
  EXPECT_TRUE(linalg::detail::ApproxEqual(1.0f, 1.0f));
  EXPECT_TRUE(linalg::detail::ApproxEqual(1.0L, 1.0L));
  EXPECT_TRUE(linalg::detail::ApproxEqual(std::complex{1.0, 1.0}, std::complex{1.0, 1.0}));
  EXPECT_TRUE(linalg::detail::ApproxEqual(std::complex{1.0f, 1.0f}, std::complex{1.0f, 1.0f}));
  EXPECT_TRUE(linalg::detail::ApproxEqual(std::complex{1.0L, 1.0L}, std::complex{1.0L, 1.0L}));

  // Approximate equality.
  EXPECT_TRUE(linalg::detail::ApproxEqual(1.0, 1.0 + std::numeric_limits<double>::epsilon() / 2));
  EXPECT_TRUE(linalg::detail::ApproxEqual(1.0f, 1.0f + std::numeric_limits<float>::epsilon() / 2));
  EXPECT_TRUE(linalg::detail::ApproxEqual(1.0L, 1.0L + std::numeric_limits<long double>::epsilon() / 2));
  EXPECT_TRUE(linalg::detail::ApproxEqual(std::complex{1.0, 1.0},
                                          std::complex{1.0 + std::numeric_limits<double>::epsilon() / 2, 1.0}));
  EXPECT_TRUE(linalg::detail::ApproxEqual(std::complex{1.0f, 1.0f},
                                          std::complex{1.0f + std::numeric_limits<float>::epsilon() / 2, 1.0f}));
  EXPECT_TRUE(linalg::detail::ApproxEqual(std::complex{1.0L, 1.0L},
                                          std::complex{1.0L + std::numeric_limits<long double>::epsilon() / 2, 1.0L}));
}

TEST(ComplexOperations, ImplicitCast) {
  // std::complex<double> a{1, 0};
  // std::complex<double> b{2, 0};

  // long double c{3};

  // auto res = c + b;

  // static_assert(linalg::detail::kIsComplexV<const std::complex<double>>);
  // static_assert(
  //     std::is_same_v<std::common_type_t<std::complex<const double>, std::complex<double>>, std::complex<const
  //     double>>);
  static_assert(
      std::is_same_v<std::common_type_t<const double, std::complex<const double>>, std::complex<const double>>);
}

}  // namespace
