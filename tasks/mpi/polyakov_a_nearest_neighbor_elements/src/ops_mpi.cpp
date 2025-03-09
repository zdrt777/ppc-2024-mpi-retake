#include "mpi/polyakov_a_nearest_neighbor_elements/include/ops_mpi.hpp"

#include <algorithm>
#include <boost/mpi/collectives/broadcast.hpp>
#include <boost/mpi/collectives/reduce.hpp>
#include <boost/mpi/operations.hpp>
#include <climits>
#include <cmath>
#include <cstddef>
#include <utility>
#include <vector>

bool polyakov_a_nearest_neighbor_elements_mpi::NearestNeighborElementsSeq::ValidationImpl() {
  return !task_data->inputs_count.empty() && task_data->outputs_count[0] == 2 && task_data->inputs_count[0] > 1;
}

bool polyakov_a_nearest_neighbor_elements_mpi::NearestNeighborElementsSeq::PreProcessingImpl() {
  unsigned int input_size = task_data->inputs_count[0];
  auto* in_ptr = reinterpret_cast<int*>(task_data->inputs[0]);
  input_ = std::vector<int>(in_ptr, in_ptr + input_size);

  unsigned int output_size = task_data->outputs_count[0];
  output_ = std::vector<int>(output_size, 0);

  size_ = input_size;

  return true;
}

bool polyakov_a_nearest_neighbor_elements_mpi::NearestNeighborElementsSeq::RunImpl() {
  int min = std::numeric_limits<int>::max();
  int tmp{};
  size_t ind{};

  for (size_t i = 0; i < size_ - 1; i++) {
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

bool polyakov_a_nearest_neighbor_elements_mpi::NearestNeighborElementsSeq::PostProcessingImpl() {
  for (size_t i = 0; i < output_.size(); i++) {
    reinterpret_cast<int*>(task_data->outputs[0])[i] = output_[i];
  }

  return true;
}

bool polyakov_a_nearest_neighbor_elements_mpi::NearestNeighborElementsMpi::ValidationImpl() {
  if (world_.rank() == 0) {
    return !task_data->inputs_count.empty() && task_data->outputs_count[0] == 2 && task_data->inputs_count[0] > 1;
  }
  return true;
}

bool polyakov_a_nearest_neighbor_elements_mpi::NearestNeighborElementsMpi::PreProcessingImpl() {
  if (world_.rank() == 0) {
    unsigned int input_size = task_data->inputs_count[0];

    auto* in_ptr = reinterpret_cast<int*>(task_data->inputs[0]);
    input_ = std::vector<int>(in_ptr, in_ptr + input_size);

    size_ = input_size;
  }

  res_ = {0, 0};

  return true;
}

bool polyakov_a_nearest_neighbor_elements_mpi::NearestNeighborElementsMpi::RunImpl() {
  unsigned int delta = 0;
  unsigned int remains = 0;

  if (world_.rank() == 0) {
    delta = size_ / world_.size();
    remains = size_ % world_.size();
  }
  boost::mpi::broadcast(world_, delta, 0);

  if (world_.rank() == 0) {
    if (world_.size() > 1) {
      world_.send(1, 0, input_.data(), delta + 1);
    }
    for (int i = 2; i < world_.size(); i++) {
      world_.send(i, 0, input_.data() + ((i - 1) * delta), delta + 1);
    }
  }

  local_input_ = std::vector<int>(delta + 1);

  if (world_.rank() == 0) {
    unsigned int zero_delta = delta + remains;
    local_input_ = std::vector<int>(input_.begin() + (world_.size() - 1) * delta,
                                    input_.begin() + delta * world_.size() + zero_delta);
  } else {
    world_.recv(0, 0, local_input_.data(), delta + 1);
  }

  std::pair<int, int> local_ans = {std::numeric_limits<int>::max(), -1};
  std::pair<int, int> tmp = {0, 0};
  int offset = delta * world.rank();

  for (size_t i = 0; i < local_input_.size() - 1; i++) {
    tmp = {std::abs(local_input_[i + 1] - local_input_[i]), i + offset};
    if (tmp < local_ans) {
      local_ans = tmp;
    }
  }
  reduce(world_, local_ans, res_, boost::mpi::minimum<std::pair<int, int>>(), 0);

  return true;
}

bool polyakov_a_nearest_neighbor_elements_mpi::NearestNeighborElementsMpi::PostProcessingImpl() {
  if (world_.rank() == 0) {
    reinterpret_cast<int*>(task_data->outputs[0])[0] = input_[res_.second];
    reinterpret_cast<int*>(task_data->outputs[0])[1] = input_[res_.second + 1];
  }
  return true;
}