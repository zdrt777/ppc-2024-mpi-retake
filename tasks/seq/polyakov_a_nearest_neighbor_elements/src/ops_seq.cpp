#include "seq/polyakov_a_nearest_neighbor_elements/include/ops_seq.hpp"

#include <cmath>
#include <cstddef>
#include <limits>
#include <vector>

bool polyakov_a_nearest_neighbor_elements_seq::NearestNeighborElementsSeq::ValidationImpl() {
  return !task_data->inputs_count.empty() && task_data->outputs_count[0] == 2 && task_data->inputs_count[0] > 1;
}

bool polyakov_a_nearest_neighbor_elements_seq::NearestNeighborElementsSeq::PreProcessingImpl() {
  unsigned int input_size = task_data->inputs_count[0];
  auto *in_ptr = reinterpret_cast<int *>(task_data->inputs[0]);
  input_ = std::vector<int>(in_ptr, in_ptr + input_size);

  unsigned int output_size = task_data->outputs_count[0];
  output_ = std::vector<int>(output_size, 0);

  size_ = input_size;

  return true;
}

bool polyakov_a_nearest_neighbor_elements_seq::NearestNeighborElementsSeq::RunImpl() {
  int min = std::numeric_limits<int>::max();
  int tmp{};
  int ind{};

  for (int i = 0; i < size_ - 1; i++) {
    tmp = std::abs(input_[i + 1] - input_[i]);
    if (tmp < min) {
      min = tmp;
      ind = i;
    }
  }
  output_[0] = input_[ind];
  output_[1] = input_[ind + 1];
  return true;
}

bool polyakov_a_nearest_neighbor_elements_seq::NearestNeighborElementsSeq::PostProcessingImpl() {
  for (size_t i = 0; i < output_.size(); i++) {
    reinterpret_cast<int *>(task_data->outputs[0])[i] = output_[i];
  }

  return true;
}
