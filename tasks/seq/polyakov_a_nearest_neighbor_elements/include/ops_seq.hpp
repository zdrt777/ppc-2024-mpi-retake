#pragma once

#include <utility>
#include <cstddef>
#include <vector>

#include "core/task/include/task.hpp"

namespace polyakov_a_nearest_neighbor_elements_seq {

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

}  // namespace polyakov_a_nearest_neighbor_elements_seq