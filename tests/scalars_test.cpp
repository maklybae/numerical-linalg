#include "../src/scalars.h"

#include <gtest/gtest.h>

#include "../src/types_details.h"

namespace {

TEST(ScalarTypesConcepts, StaticAssertions) {
  static_assert(linalg::types::FloatingOrComplexType<float>);
  static_assert(linalg::types::FloatingOrComplexType<double>);
  static_assert(linalg::types::FloatingOrComplexType<long double>);
  static_assert(linalg::types::FloatingOrComplexType<std::complex<float>>);
  static_assert(linalg::types::FloatingOrComplexType<std::complex<double>>);
  static_assert(linalg::types::FloatingOrComplexType<std::complex<long double>>);
}

TEST(ScalarUtils, ApproxEqual) {
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

}  // namespace
