#include "beads_gym/beads/bead.hpp"

#include <gtest/gtest.h>
#include <iostream>
#include <numeric>
#include <string>
#include <Eigen/Dense>


namespace beads_gym::beads {
  
  TEST(BeadTest, BeadConstructor) {
    std::vector<double> position{0, 0, 0};
    Eigen::Vector3d position_3d{position.data()};
    Bead<Eigen::Vector3d> bead_3d{0, position_3d, 1.0, true};
    EXPECT_EQ(bead_3d.get_position(), position_3d);
  }

// TEST(FooTest, StringVectorOutput) {
//   std::vector<std::string> result;
//   ASSERT_NO_THROW(result = stringVectorOutput(8));
//   EXPECT_EQ(result.size(), 8);
//   for (const auto& it : result) {
//     EXPECT_EQ(it, std::to_string(8));
//   }
// }

// TEST(FooTest, StringVectorValueInput) {
//   const std::vector<std::string> data{"1", "2", "3", "4", "5"};
//   int                            size = 0;
//   ASSERT_NO_THROW(size = stringVectorInput(data));
//   EXPECT_EQ(size, 5);
// }

// TEST(FooTest, StringVectorRefInput) {
//   const std::vector<std::string> data{"1", "2", "3", "4", "5"};
//   int                            size = 0;
//   ASSERT_NO_THROW(size = stringVectorRefInput(data));
//   EXPECT_EQ(size, 5);
// }

// TEST(FooTest, StringJaggedArrayOutput) {
//   std::vector<std::vector<std::string>> result;
//   ASSERT_NO_THROW(result = stringJaggedArrayOutput(8));
//   EXPECT_EQ(result.size(), 8);
//   for (std::size_t i = 0; i < result.size(); ++i) {
//     EXPECT_EQ(i + 1, result[i].size());
//   }
//   for (std::size_t i = 1; i <= result.size(); ++i) {
//     const auto& inner = result[i - 1];
//     for (const auto& it : inner) {
//       EXPECT_EQ(it, std::to_string(i));
//     }
//   }
// }

// TEST(FooTest, StringJaggedArrayValueInput) {
//   const std::vector<std::vector<std::string>> data{{"1", "2", "3"}, {"4", "5"}};
//   int                                         size = 0;
//   ASSERT_NO_THROW(size = stringJaggedArrayInput(data));
//   EXPECT_EQ(size, 2);
// }

// TEST(FooTest, StringJaggedArrayRefInput) {
//   const std::vector<std::vector<std::string>> data{{"1", "2", "3"}, {"4", "5"}};
//   int                                         size = 0;
//   ASSERT_NO_THROW(size = stringJaggedArrayRefInput(data));
//   EXPECT_EQ(size, 2);
// }

// TEST(FooTest, PairVectorOutput) {
//   std::vector<std::pair<int, int>> result;
//   ASSERT_NO_THROW(result = pairVectorOutput(8));
//   EXPECT_EQ(result.size(), 8);
//   for (const auto& it : result) {
//     EXPECT_EQ(it.first, 8);
//     EXPECT_EQ(it.second, 8);
//   }
// }

// TEST(FooTest, PairVectorValueInput) {
//   const std::vector<std::pair<int, int>> data{{1, 2}, {3, 4}, {5, 6}};
//   int                                    size = 0;
//   ASSERT_NO_THROW(size = pairVectorInput(data));
//   EXPECT_EQ(size, 3);
// }

// TEST(FooTest, PairVectorRefInput) {
//   const std::vector<std::pair<int, int>> data{{1, 2}, {3, 4}, {5, 6}};
//   int                                    size = 0;
//   ASSERT_NO_THROW(size = pairVectorRefInput(data));
//   EXPECT_EQ(size, 3);
// }

// TEST(FooTest, PairJaggedArrayOutput) {
//   std::vector<std::vector<std::pair<int, int>>> result;
//   ASSERT_NO_THROW(result = pairJaggedArrayOutput(8));
//   EXPECT_EQ(result.size(), 8);
//   for (std::size_t i = 0; i < result.size(); ++i) {
//     EXPECT_EQ(i + 1, result[i].size());
//   }
//   for (int i = 1; i <= static_cast<int>(result.size()); ++i) {
//     const auto& inner = result[i - 1];
//     for (const auto& it : inner) {
//       EXPECT_EQ(it, std::make_pair(i, i));
//     }
//   }
// }

// TEST(FooTest, PairJaggedArrayValueInput) {
//   std::vector<std::vector<std::pair<int, int>>> data{{{1, 1}, {2, 2}, {3, 3}}, {{4, 4}, {5, 5}}};
//   int                                           size = 0;
//   ASSERT_NO_THROW(size = pairJaggedArrayInput(data));
//   EXPECT_EQ(size, 2);
// }

// TEST(FooTest, PairJaggedArrayRefInput) {
//   std::vector<std::vector<std::pair<int, int>>> data{{{1, 1}, {2, 2}, {3, 3}}, {{4, 4}, {5, 5}}};
//   int                                           size = 0;
//   ASSERT_NO_THROW(size = pairJaggedArrayRefInput(data));
//   EXPECT_EQ(size, 2);
// }

// TEST(FooTest, StaticMethods) {
//   EXPECT_NO_THROW(Bead::staticFunction(42));
//   EXPECT_NO_THROW(Bead::staticFunction(int64_t{42}));
// }

// TEST(FooTest, Constructor) {
//   Bead* b = new Bead();
//   EXPECT_NE(b, nullptr);
// }

// TEST(FooTest, IntMethods) {
//   Bead bead;
//   ASSERT_NO_THROW(bead.setInt(42));
//   EXPECT_EQ(42, bead.getInt());
// }

// TEST(FooTest, Int64Methods) {
//   Bead bead;
//   ASSERT_NO_THROW(bead.setInt64(31));
//   EXPECT_EQ(31, bead.getInt64());
// }

// TEST(FooTest, PrintMethod) {
//   Bead         bead;
//   std::string str("");
//   ASSERT_NO_THROW(str = bead());
//   EXPECT_EQ("\"Bead\":{\"int\":0,\"int64\":0}", str);
// }

} // namespace beads_gym::beads
