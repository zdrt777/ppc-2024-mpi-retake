#pragma once

#include <boost/mpi/collectives.hpp>
#include <boost/mpi/communicator.hpp>
#include <climits>
#include <cstddef>
#include <memory>
#include <utility>
#include <vector>

#include "core/task/include/task.hpp"

namespace polyakov_a_nearest_neighbor_elements_mpi {

class NearestNeighborElementsSeq : public ppc::core::Task {
 public:
  explicit NearestNeighborElementsSeq(ppc::core::TaskDataPtr task_data) : Task(std::move(task_data)) {}
  bool ValidationImpl() override;
  bool PreProcessingImpl() override;
  bool RunImpl() override;
  bool PostProcessingImpl() override;

 private:
  std::vector<int> input_, output_;
  size_t size_{};
};

class NearestNeighborElementsMpi : public ppc::core::Task {
 public:
  explicit NearestNeighborElementsMpi(ppc::core::TaskDataPtr task_data) : Task(std::move(task_data)) {}
  bool ValidationImpl() override;
  bool PreProcessingImpl() override;
  bool RunImpl() override;
  bool PostProcessingImpl() override;

 private:
  std::vector<int> input_, local_input_;
  std::pair<int, int> res_;
  size_t size_{};
  size_t start_;
  boost::mpi::communicator world_;
};

}  // namespace polyakov_a_nearest_neighbor_elements_mpi