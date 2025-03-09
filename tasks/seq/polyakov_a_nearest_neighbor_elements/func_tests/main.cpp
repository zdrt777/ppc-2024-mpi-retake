#include <gtest/gtest.h>

#include <cstddef>
#include <cstdint>
#include <fstream>
#include <memory>
#include <string>
#include <vector>

#include "core/task/include/task.hpp"
#include "seq/polyakov_a_nearest_neighbor_elements/include/ops_seq.hpp"

TEST(polyakov_a_nearest_neighbor_elements_seq, test_validation_one) {
  std::vector<int> in;
  std::vector<int> out(2, 0);

  auto task_data_seq = std::make_shared<ppc::core::TaskData>();
  task_data_seq->inputs.emplace_back(reinterpret_cast<uint8_t *>(in.data()));
  task_data_seq->inputs_count.emplace_back(in.size());
  task_data_seq->outputs.emplace_back(reinterpret_cast<uint8_t *>(out.data()));
  task_data_seq->outputs_count.emplace_back(out.size());

  polyakov_a_nearest_neighbor_elements_seq::NearestNeighborElementsSeq test_task_sequential(task_data_seq);
  ASSERT_FALSE(test_task_sequential.ValidationImpl());
}

TEST(polyakov_a_nearest_neighbor_elements_seq, test_validation_two) {
  std::vector<int> in;
  std::vector<int> out(2, 0);

  auto task_data_seq = std::make_shared<ppc::core::TaskData>();
  task_data_seq->inputs.emplace_back(reinterpret_cast<uint8_t *>(in.data()));
  task_data_seq->inputs_count.emplace_back(in.size());
  task_data_seq->outputs.emplace_back(reinterpret_cast<uint8_t *>(out.data()));
  task_data_seq->outputs_count.emplace_back(out.size());

  polyakov_a_nearest_neighbor_elements_seq::NearestNeighborElementsSeq test_task_sequential(task_data_seq);
  ASSERT_FALSE(test_task_sequential.ValidationImpl());
}

TEST(polyakov_a_nearest_neighbor_elements_seq, test_validation_three) {
  std::vector<int> in(1, 0);
  std::vector<int> out(2, 0);

  auto task_data_seq = std::make_shared<ppc::core::TaskData>();
  task_data_seq->inputs.emplace_back(reinterpret_cast<uint8_t *>(in.data()));
  task_data_seq->inputs_count.emplace_back(in.size());
  task_data_seq->outputs.emplace_back(reinterpret_cast<uint8_t *>(out.data()));
  task_data_seq->outputs_count.emplace_back(out.size());

  polyakov_a_nearest_neighbor_elements_seq::NearestNeighborElementsSeq test_task_sequential(task_data_seq);
  ASSERT_FALSE(test_task_sequential.ValidationImpl());
}

TEST(polyakov_a_nearest_neighbor_elements_seq, test_validation_four) {
  std::vector<int> in(2, 0);
  std::vector<int> out(1, 0);

  auto task_data_seq = std::make_shared<ppc::core::TaskData>();
  task_data_seq->inputs.emplace_back(reinterpret_cast<uint8_t *>(in.data()));
  task_data_seq->inputs_count.emplace_back(in.size());
  task_data_seq->outputs.emplace_back(reinterpret_cast<uint8_t *>(out.data()));
  task_data_seq->outputs_count.emplace_back(out.size());

  polyakov_a_nearest_neighbor_elements_seq::NearestNeighborElementsSeq test_task_sequential(task_data_seq);
  ASSERT_FALSE(test_task_sequential.ValidationImpl());
}

TEST(polyakov_a_nearest_neighbor_elements_seq, test_two_elements) {
  std::vector<int> in = {1, 2};

  std::vector<int> out(2, 0);

  std::vector<int> res = {1, 2};

  auto task_data_seq = std::make_shared<ppc::core::TaskData>();
  task_data_seq->inputs.emplace_back(reinterpret_cast<uint8_t *>(in.data()));
  task_data_seq->inputs_count.emplace_back(in.size());
  task_data_seq->outputs.emplace_back(reinterpret_cast<uint8_t *>(out.data()));
  task_data_seq->outputs_count.emplace_back(out.size());

  polyakov_a_nearest_neighbor_elements_seq::NearestNeighborElementsSeq test_task_sequential(task_data_seq);

  ASSERT_EQ(test_task_sequential.ValidationImpl(), true);
  test_task_sequential.PreProcessingImpl();
  test_task_sequential.RunImpl();
  test_task_sequential.PostProcessingImpl();

  EXPECT_EQ(res[0], out[0]);
  EXPECT_EQ(res[1], out[1]);
}

TEST(polyakov_a_nearest_neighbor_elements_seq, test_ascending_order) {
  std::vector<int> in = {-21, -16, -10, -5, -1, 2, 4, 1000};

  std::vector<int> out(2, 0);

  std::vector<int> res = {2, 4};

  auto task_data_seq = std::make_shared<ppc::core::TaskData>();
  task_data_seq->inputs.emplace_back(reinterpret_cast<uint8_t *>(in.data()));
  task_data_seq->inputs_count.emplace_back(in.size());
  task_data_seq->outputs.emplace_back(reinterpret_cast<uint8_t *>(out.data()));
  task_data_seq->outputs_count.emplace_back(out.size());

  polyakov_a_nearest_neighbor_elements_seq::NearestNeighborElementsSeq test_task_sequential(task_data_seq);

  ASSERT_EQ(test_task_sequential.ValidationImpl(), true);
  test_task_sequential.PreProcessingImpl();
  test_task_sequential.RunImpl();
  test_task_sequential.PostProcessingImpl();

  EXPECT_EQ(res[0], out[0]);
  EXPECT_EQ(res[1], out[1]);
}

TEST(polyakov_a_nearest_neighbor_elements_seq, test_descending_order) {
  std::vector<int> in = {100, 50, 30, 20, 0, -5, -10, -14, -20, -30};

  std::vector<int> out(2, 0);

  std::vector<int> res = {-10, -14};

  auto task_data_seq = std::make_shared<ppc::core::TaskData>();
  task_data_seq->inputs.emplace_back(reinterpret_cast<uint8_t *>(in.data()));
  task_data_seq->inputs_count.emplace_back(in.size());
  task_data_seq->outputs.emplace_back(reinterpret_cast<uint8_t *>(out.data()));
  task_data_seq->outputs_count.emplace_back(out.size());

  polyakov_a_nearest_neighbor_elements_seq::NearestNeighborElementsSeq test_task_sequential(task_data_seq);

  ASSERT_EQ(test_task_sequential.ValidationImpl(), true);
  test_task_sequential.PreProcessingImpl();
  test_task_sequential.RunImpl();
  test_task_sequential.PostProcessingImpl();

  EXPECT_EQ(res[0], out[0]);
  EXPECT_EQ(res[1], out[1]);
}

TEST(polyakov_a_nearest_neighbor_elements_seq, test_end_array) {
  std::vector<int> in = {100, 50, 60, 2, 60, 70, 75, 74};

  std::vector<int> out(2, 0);

  std::vector<int> res = {75, 74};

  auto task_data_seq = std::make_shared<ppc::core::TaskData>();
  task_data_seq->inputs.emplace_back(reinterpret_cast<uint8_t *>(in.data()));
  task_data_seq->inputs_count.emplace_back(in.size());
  task_data_seq->outputs.emplace_back(reinterpret_cast<uint8_t *>(out.data()));
  task_data_seq->outputs_count.emplace_back(out.size());

  polyakov_a_nearest_neighbor_elements_seq::NearestNeighborElementsSeq test_task_sequential(task_data_seq);

  ASSERT_EQ(test_task_sequential.ValidationImpl(), true);
  test_task_sequential.PreProcessingImpl();
  test_task_sequential.RunImpl();
  test_task_sequential.PostProcessingImpl();

  EXPECT_EQ(res[0], out[0]);
  EXPECT_EQ(res[1], out[1]);
}

TEST(polyakov_a_nearest_neighbor_elements_seq, test_beginning_of_array) {
  std::vector<int> in = {1, 0, 3, -3, 6, -6, 100, 1000};

  std::vector<int> out(2, 0);

  std::vector<int> res = {1, 0};

  auto task_data_seq = std::make_shared<ppc::core::TaskData>();
  task_data_seq->inputs.emplace_back(reinterpret_cast<uint8_t *>(in.data()));
  task_data_seq->inputs_count.emplace_back(in.size());
  task_data_seq->outputs.emplace_back(reinterpret_cast<uint8_t *>(out.data()));
  task_data_seq->outputs_count.emplace_back(out.size());

  polyakov_a_nearest_neighbor_elements_seq::NearestNeighborElementsSeq test_task_sequential(task_data_seq);

  ASSERT_EQ(test_task_sequential.ValidationImpl(), true);
  test_task_sequential.PreProcessingImpl();
  test_task_sequential.RunImpl();
  test_task_sequential.PostProcessingImpl();

  EXPECT_EQ(res[0], out[0]);
  EXPECT_EQ(res[1], out[1]);
}

TEST(polyakov_a_nearest_neighbor_elements_seq, test_two_equals_elements) {
  std::vector<int> in = {1, 2, 3, 4, 5, 5, 6, 7, 8, -10};

  std::vector<int> out(2, 0);

  std::vector<int> res = {5, 5};

  auto task_data_seq = std::make_shared<ppc::core::TaskData>();
  task_data_seq->inputs.emplace_back(reinterpret_cast<uint8_t *>(in.data()));
  task_data_seq->inputs_count.emplace_back(in.size());
  task_data_seq->outputs.emplace_back(reinterpret_cast<uint8_t *>(out.data()));
  task_data_seq->outputs_count.emplace_back(out.size());

  polyakov_a_nearest_neighbor_elements_seq::NearestNeighborElementsSeq test_task_sequential(task_data_seq);

  ASSERT_EQ(test_task_sequential.ValidationImpl(), true);
  test_task_sequential.PreProcessingImpl();
  test_task_sequential.RunImpl();
  test_task_sequential.PostProcessingImpl();

  EXPECT_EQ(res[0], out[0]);
  EXPECT_EQ(res[1], out[1]);
}

TEST(polyakov_a_nearest_neighbor_elements_seq, test_equals_elements) {
  std::vector<int> in = {100, 10, 1, 1, 2, 2, 2, 2, 2};

  std::vector<int> out(2, 0);

  std::vector<int> res = {1, 1};

  auto task_data_seq = std::make_shared<ppc::core::TaskData>();
  task_data_seq->inputs.emplace_back(reinterpret_cast<uint8_t *>(in.data()));
  task_data_seq->inputs_count.emplace_back(in.size());
  task_data_seq->outputs.emplace_back(reinterpret_cast<uint8_t *>(out.data()));
  task_data_seq->outputs_count.emplace_back(out.size());

  polyakov_a_nearest_neighbor_elements_seq::NearestNeighborElementsSeq test_task_sequential(task_data_seq);

  ASSERT_EQ(test_task_sequential.ValidationImpl(), true);
  test_task_sequential.PreProcessingImpl();
  test_task_sequential.RunImpl();
  test_task_sequential.PostProcessingImpl();

  EXPECT_EQ(res[0], out[0]);
  EXPECT_EQ(res[1], out[1]);
}