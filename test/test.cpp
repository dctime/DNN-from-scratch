#include <iostream>
#include <gtest/gtest.h>

TEST(ADDTEST, ADDTEST_TRUE) {
  int num = 1;
  EXPECT_EQ(num, 1);
}

int main(int argc, char** argv) {
  testing::InitGoogleTest(&argc, argv);
  return RUN_ALL_TESTS();
}
