#include <gtest/gtest.h>

#include <concepts>

#include "linalg.h"

namespace {
TEST(Stepper, BasicOperations) {

  linalg::detail::Stepper stepper{5};

  stepper += 3;
  EXPECT_EQ(stepper.GetStep(), 3);
  EXPECT_EQ(stepper.GetIteration(), 0);

  stepper += 2;
  EXPECT_EQ(stepper.GetStep(), 0);
  EXPECT_EQ(stepper.GetIteration(), 1);

  stepper -= 1;
  EXPECT_EQ(stepper.GetStep(), 4);
  EXPECT_EQ(stepper.GetIteration(), 0);

  ++stepper;
  EXPECT_EQ(stepper.GetStep(), 0);
  EXPECT_EQ(stepper.GetIteration(), 1);

  stepper--;
  EXPECT_EQ(stepper.GetStep(), 4);
  EXPECT_EQ(stepper.GetIteration(), 0);

  ++stepper;
  EXPECT_EQ(stepper.GetStep(), 0);
  EXPECT_EQ(stepper.GetIteration(), 1);

  stepper += 5;
  EXPECT_EQ(stepper.GetStep(), 0);
  EXPECT_EQ(stepper.GetIteration(), 2);

  stepper -= 5;
  EXPECT_EQ(stepper.GetStep(), 0);
  EXPECT_EQ(stepper.GetIteration(), 1);
}

TEST(Stepper, BoundaryConditions) {
  linalg::detail::Stepper stepper{5};

  stepper += 5;
  EXPECT_EQ(stepper.GetStep(), 0);
  EXPECT_EQ(stepper.GetIteration(), 1);

  stepper -= 5;
  EXPECT_EQ(stepper.GetStep(), 0);
  EXPECT_EQ(stepper.GetIteration(), 0);

  stepper += 10;
  EXPECT_EQ(stepper.GetStep(), 0);
  EXPECT_EQ(stepper.GetIteration(), 2);

  stepper -= 10;
  EXPECT_EQ(stepper.GetStep(), 0);
  EXPECT_EQ(stepper.GetIteration(), 0);
}

TEST(Stepper, Compare) {
  static_assert(std::totally_ordered<linalg::detail::Stepper>);

  linalg::detail::Stepper stepper1{5};
  linalg::detail::Stepper stepper2{5};
  linalg::detail::Stepper stepper3{4};

  EXPECT_EQ(stepper1, stepper2);
  EXPECT_EQ(stepper1 <=> stepper3, std::partial_ordering::unordered);

  stepper1 += 3;
  EXPECT_GT(stepper1, stepper2);

  stepper1 += 7;  // now (2, 0)
  stepper2 += 5;  // now (1, 0)

  EXPECT_EQ(stepper1.GetIteration(), 2);
  EXPECT_EQ(stepper1.GetStep(), 0);
  EXPECT_EQ(stepper2.GetIteration(), 1);
  EXPECT_EQ(stepper2.GetStep(), 0);
  EXPECT_GT(stepper1, stepper2);
  EXPECT_LT(stepper2, stepper1);
}

}  // namespace
