#include <gtest/gtest.h>

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

}  // namespace
