#include <gtest/gtest.h>

#include <boost/mpi/timer.hpp>
#include <cstdint>
#include <memory>
#include <vector>

#include "boost/mpi/communicator.hpp"
#include "core/perf/include/perf.hpp"
#include "core/task/include/task.hpp"
#include "mpi/polyakov_a_nearest_neighbor_elements/include/ops_mpi.hpp"

TEST(polyakov_a_nearest_neighbor_elements_mpi, test_pipeline_run) {
  boost::mpi::communicator world;
  constexpr int kCount = 20000000;

  std::vector<int> in(kCount);
  std::vector<int> out(2, 0);

  std::random_device rd;
  std::mt19937 gen(rd());
  std::uniform_int_distribution<int> dist(-10000000, 10000000);
  for (auto &i : in) {
    i = dist(gen);
  }

  auto task_data_mpi = std::make_shared<ppc::core::TaskData>();
  if (world.rank() == 0) {
    task_data_mpi->inputs.emplace_back(reinterpret_cast<uint8_t *>(in.data()));
    task_data_mpi->inputs_count.emplace_back(in.size());
    task_data_mpi->outputs.emplace_back(reinterpret_cast<uint8_t *>(out.data()));
    task_data_mpi->outputs_count.emplace_back(out.size());
  }

  auto test_task_mpi =
      std::make_shared<polyakov_a_nearest_neighbor_elements_mpi::NearestNeighborElementsMpi>(task_data_mpi);
  ASSERT_EQ(test_task_mpi->ValidationImpl(), true);
  test_task_mpi->PreProcessingImpl();
  test_task_mpi->RunImpl();
  test_task_mpi->PostProcessingImpl();

  auto perf_attr = std::make_shared<ppc::core::PerfAttr>();
  perf_attr->num_running = 25;
  const boost::mpi::timer current_timer;
  perf_attr->current_timer = [&] { return current_timer.elapsed(); };

  auto perf_results = std::make_shared<ppc::core::PerfResults>();

  auto perf_analyzer = std::make_shared<ppc::core::Perf>(test_task_mpi);
  perf_analyzer->PipelineRun(perf_attr, perf_results);
  if (world.rank() == 0) {
    ppc::core::Perf::PrintPerfStatistic(perf_results);
    ASSERT_EQ(out.size(), static_cast<size_t>(2));
  }
}

TEST(polyakov_a_nearest_neighbor_elements_mpi, test_task_run) {
  boost::mpi::communicator world;
  constexpr int kCount = 20000000;

  std::vector<int> in(kCount);
  std::vector<int> out(2, 0);

  std::random_device rd;
  std::mt19937 gen(rd());
  std::uniform_int_distribution<int> dist(-10000000, 10000000);
  for (auto &i : in) {
    i = dist(gen);
  }

  auto task_data_mpi = std::make_shared<ppc::core::TaskData>();
  if (world.rank() == 0) {
    task_data_mpi->inputs.emplace_back(reinterpret_cast<uint8_t *>(in.data()));
    task_data_mpi->inputs_count.emplace_back(in.size());
    task_data_mpi->outputs.emplace_back(reinterpret_cast<uint8_t *>(out.data()));
    task_data_mpi->outputs_count.emplace_back(out.size());
  }

  auto test_task_mpi =
      std::make_shared<polyakov_a_nearest_neighbor_elements_mpi::NearestNeighborElementsMpi>(task_data_mpi);
  ASSERT_EQ(test_task_mpi->ValidationImpl(), true);
  test_task_mpi->PreProcessingImpl();
  test_task_mpi->RunImpl();
  test_task_mpi->PostProcessingImpl();

  auto perf_attr = std::make_shared<ppc::core::PerfAttr>();
  perf_attr->num_running = 25;
  const boost::mpi::timer current_timer;
  perf_attr->current_timer = [&] { return current_timer.elapsed(); };

  auto perf_results = std::make_shared<ppc::core::PerfResults>();

  auto perf_analyzer = std::make_shared<ppc::core::Perf>(test_task_mpi);
  perf_analyzer->TaskRun(perf_attr, perf_results);
  if (world.rank() == 0) {
    ppc::core::Perf::PrintPerfStatistic(perf_results);
    ASSERT_EQ(out.size(), static_cast<size_t>(2));
  }
}
