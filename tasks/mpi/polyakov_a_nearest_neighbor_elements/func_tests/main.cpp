#include <gtest/gtest.h>

#include <boost/mpi/communicator.hpp>
#include <cstdint>
#include <memory>
#include <random>
#include <vector>

#include "core/task/include/task.hpp"
#include "mpi/polyakov_a_nearest_neighbor_elements/include/ops_mpi.hpp"

namespace polyakov_a_nearest_neighbor_elements_mpi {
std::vector<int> Generator(int sz) {
  std::random_device rd;
  std::mt19937 gen(rd());
  std::uniform_int_distribution<int> dist(-10000000, 10000000);
  std::vector<int> ans(sz);

  for (auto &i : ans) {
    i = dist(gen);
  }

  return ans;
}
}  // namespace polyakov_a_nearest_neighbor_elements_mpi

TEST(polyakov_a_nearest_neighbor_elements_mpi, random_test_vec1000) {
  boost::mpi::communicator world;
  std::vector<int> in(1000);
  std::vector<int> out(2);

  auto task_data_mpi = std::make_shared<ppc::core::TaskData>();
  if (world.rank() == 0) {
    in = Generator(1000);
    task_data_mpi->inputs.emplace_back(reinterpret_cast<uint8_t *>(in.data()));
    task_data_mpi->inputs_count.emplace_back(in.size());
    task_data_mpi->outputs.emplace_back(reinterpret_cast<uint8_t *>(ouy.data()));
    task_data_mpi->outputs_count.emplace_back(out.size());
  }

  polyakov_a_nearest_neighbor_elements_mpi::NearestNeighborElementsMpi test_mpi_task_parallel(task_data_mpi);
  ASSERT_EQ(test_mpi_task_parallel.ValidationImpl(), true);
  test_mpi_task_parallel.PreProcessingImpl();
  test_mpi_task_parallel.RunImpl();
  test_mpi_task_parallel.PostProcessingImpl();

  if (world.rank() == 0) {
    std::vector<int> checking(2);

    auto task_data_seq = std::make_shared<ppc::core::TaskData>();
    task_data_seq->inputs.emplace_back(reinterpret_cast<uint8_t *>(in.data()));
    task_data_seq->inputs_count.emplace_back(in.size());
    task_data_seq->outputs.emplace_back(reinterpret_cast<uint8_t *>(checking.data()));
    task_data_seq->outputs_count.emplace_back(checking.size());

    polyakov_a_nearest_neighbor_elements_mpi::NearestNeighborElementsSeq test_task_sequential(task_data_seq);
    ASSERT_EQ(test_task_sequential.ValidationImpl(), true);
    test_task_sequential.PreProcessingImpl();
    test_task_sequential.RunImpl();
    test_task_sequential.PostProcessingImpl();

    ASSERT_EQ(checking[0], out[0]);
    ASSERT_EQ(checking[1], out[1]);
  }
}

TEST(polyakov_a_nearest_neighbor_elements_mpi, random_test_vec10000) {
  boost::mpi::communicator world;
  std::vector<int> in(10000);
  std::vector<int> out(2);

  auto task_data_mpi = std::make_shared<ppc::core::TaskData>();
  if (world.rank() == 0) {
    in = Generator(10000);
    task_data_mpi->inputs.emplace_back(reinterpret_cast<uint8_t *>(in.data()));
    task_data_mpi->inputs_count.emplace_back(in.size());
    task_data_mpi->outputs.emplace_back(reinterpret_cast<uint8_t *>(out.data()));
    task_data_mpi->outputs_count.emplace_back(out.size());
  }

  polyakov_a_nearest_neighbor_elements_mpi::NearestNeighborElementsMpi test_mpi_task_parallel(task_data_mpi);
  ASSERT_EQ(test_mpi_task_parallel.ValidationImpl(), true);
  test_mpi_task_parallel.PreProcessingImpl();
  test_mpi_task_parallel.RunImpl();
  test_mpi_task_parallel.PostProcessingImpl();

  if (world.rank() == 0) {
    std::vector<int> checking(2);

    auto task_data_seq = std::make_shared<ppc::core::TaskData>();
    task_data_seq->inputs.emplace_back(reinterpret_cast<uint8_t *>(in.data()));
    task_data_seq->inputs_count.emplace_back(in.size());
    task_data_seq->outputs.emplace_back(reinterpret_cast<uint8_t *>(checking.data()));
    task_data_seq->outputs_count.emplace_back(checking.size());

    polyakov_a_nearest_neighbor_elements_mpi::NearestNeighborElementsSeq test_task_sequential(task_data_seq);
    ASSERT_EQ(test_task_sequential.ValidationImpl(), true);
    test_task_sequential.PreProcessingImpl();
    test_task_sequential.RunImpl();
    test_task_sequential.PostProcessingImpl();

    ASSERT_EQ(checking[0], out[0]);
    ASSERT_EQ(checking[1], out[1]);
  }
}

TEST(polyakov_a_nearest_neighbor_elements_mpi, empty_input_test) {
  boost::mpi::communicator world;
  std::vector<int> in;

  auto task_data_mpi = std::make_shared<ppc::core::TaskData>();
  if (world.rank() == 0) {
    std::vector<int> out(2, 0);

    auto task_data_seq = std::make_shared<ppc::core::TaskData>();
    task_data_seq->inputs.emplace_back(reinterpret_cast<uint8_t *>(in.data()));
    task_data_seq->inputs_count.emplace_back(in.size());
    task_data_seq->outputs.emplace_back(reinterpret_cast<uint8_t *>(out.data()));
    task_data_seq->outputs_count.emplace_back(out.size());

    polyakov_a_nearest_neighbor_elements_mpi::NearestNeighborElementsSeq test_task_sequential(task_data_seq);
    ASSERT_EQ(test_task_sequential.ValidationImpl(), false);
  }
}

TEST(polyakov_a_nearest_neighbor_elements_mpi, wrong_input_test) {
  boost::mpi::communicator world;
  std::vector<int> in(1);

  auto task_data_mpi = std::make_shared<ppc::core::TaskData>();
  if (world.rank() == 0) {
    std::vector<int> out(2, 0);

    auto task_data_seq = std::make_shared<ppc::core::TaskData>();
    task_data_seq->inputs.emplace_back(reinterpret_cast<uint8_t *>(in.data()));
    task_data_seq->inputs_count.emplace_back(in.size());
    task_data_seq->outputs.emplace_back(reinterpret_cast<uint8_t *>(out.data()));
    task_data_seq->outputs_count.emplace_back(out.size());

    polyakov_a_nearest_neighbor_elements_mpi::NearestNeighborElementsSeq test_task_sequential(task_data_seq);
    ASSERT_EQ(test_task_sequential.ValidationImpl(), false);
  }
}

TEST(polyakov_a_nearest_neighbor_elements_mpi, wront_out_test) {
  boost::mpi::communicator world;
  std::vector<int> in(2);

  auto task_data_mpi = std::make_shared<ppc::core::TaskData>();
  if (world.rank() == 0) {
    std::vector<int> out(1, 0);

    auto task_data_seq = std::make_shared<ppc::core::TaskData>();
    task_data_seq->inputs.emplace_back(reinterpret_cast<uint8_t *>(in.data()));
    task_data_seq->inputs_count.emplace_back(in.size());
    task_data_seq->outputs.emplace_back(reinterpret_cast<uint8_t *>(out.data()));
    task_data_seq->outputs_count.emplace_back(out.size());

    polyakov_a_nearest_neighbor_elements_mpi::NearestNeighborElementsSeq test_task_sequential(task_data_seq);
    ASSERT_EQ(test_task_sequential.ValidationImpl(), false);
  }
}
