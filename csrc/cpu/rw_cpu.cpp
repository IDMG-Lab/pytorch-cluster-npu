#include "rw_cpu.h"

#include <ATen/Parallel.h>

#include "utils.h"

template <typename IndexT>
void uniform_sampling(const IndexT *rowptr, const IndexT *col,
                      const IndexT *start, IndexT *n_out, IndexT *e_out,
                      const int64_t numel, const int64_t walk_length) {

  auto rand = torch::rand({numel, walk_length});
  auto rand_data = rand.data_ptr<float>();

  int64_t grain_size = at::internal::GRAIN_SIZE / walk_length;
  at::parallel_for(0, numel, grain_size, [&](int64_t begin, int64_t end) {
    for (auto n = begin; n < end; n++) {
      int64_t n_cur = start[n], e_cur, row_start, row_end, idx;

      n_out[n * (walk_length + 1)] = n_cur;

      for (auto l = 0; l < walk_length; l++) {
        row_start = rowptr[n_cur], row_end = rowptr[n_cur + 1];
        if (row_end - row_start == 0) {
          e_cur = -1;
        } else {
          idx = int64_t(rand_data[n * walk_length + l] * (row_end - row_start));
          e_cur = row_start + idx;
          n_cur = col[e_cur];
        }
        n_out[n * (walk_length + 1) + (l + 1)] = n_cur;
        e_out[n * walk_length + l] = e_cur;
      }
    }
  });
}

template <typename IndexT>
bool inline is_neighbor(const IndexT *rowptr, const IndexT *col, int64_t v,
                        int64_t w) {
  int64_t row_start = rowptr[v], row_end = rowptr[v + 1];
  for (auto i = row_start; i < row_end; i++) {
    if (col[i] == w)
      return true;
  }
  return false;
}

// See: https://louisabraham.github.io/articles/node2vec-sampling.html
template <typename IndexT>
void rejection_sampling(const IndexT *rowptr, const IndexT *col,
                        IndexT *start, IndexT *n_out, IndexT *e_out,
                        const int64_t numel, const int64_t walk_length,
                        const double p, const double q) {

  double max_prob = fmax(fmax(1. / p, 1.), 1. / q);
  double prob_0 = 1. / p / max_prob;
  double prob_1 = 1. / max_prob;
  double prob_2 = 1. / q / max_prob;

  int64_t grain_size = at::internal::GRAIN_SIZE / walk_length;
  at::parallel_for(0, numel, grain_size, [&](int64_t begin, int64_t end) {
    for (auto n = begin; n < end; n++) {
      int64_t t = start[n], v, x, e_cur, row_start, row_end;

      n_out[n * (walk_length + 1)] = t;

      row_start = rowptr[t], row_end = rowptr[t + 1];
      if (row_end - row_start == 0) {
        e_cur = -1;
        v = t;
      } else {
        e_cur = row_start + (rand() % (row_end - row_start));
        v = col[e_cur];
      }
      n_out[n * (walk_length + 1) + 1] = v;
      e_out[n * walk_length] = e_cur;

      for (auto l = 1; l < walk_length; l++) {
        row_start = rowptr[v], row_end = rowptr[v + 1];

        if (row_end - row_start == 0) {
          e_cur = -1;
          x = v;
        } else if (row_end - row_start == 1) {
          e_cur = row_start;
          x = col[e_cur];
        } else {
          while (true) {
            e_cur = row_start + (rand() % (row_end - row_start));
            x = col[e_cur];

            auto r = ((double)rand() / (RAND_MAX)); // [0, 1)

            if (x == t && r < prob_0)
              break;
            else if (is_neighbor(rowptr, col, x, t) && r < prob_1)
              break;
            else if (r < prob_2)
              break;
          }
        }

        n_out[n * (walk_length + 1) + (l + 1)] = x;
        e_out[n * walk_length + l] = e_cur;
        t = v;
        v = x;
      }
    }
  });
}

template <typename IndexT>
std::tuple<torch::Tensor, torch::Tensor>
random_walk_cpu_impl(torch::Tensor rowptr, torch::Tensor col, torch::Tensor start,
                int64_t walk_length, double p, double q) {
  TORCH_CHECK((rowptr.scalar_type() == at::kInt || rowptr.scalar_type() == at::kLong) &&
                  col.scalar_type() == rowptr.scalar_type() && start.scalar_type() == rowptr.scalar_type(),
              "random_walk: inputs must have the same int32 or int64 dtype");
  CHECK_CPU(rowptr);
  CHECK_CPU(col);
  CHECK_CPU(start);

  CHECK_INPUT(rowptr.dim() == 1);
  CHECK_INPUT(col.dim() == 1);
  CHECK_INPUT(start.dim() == 1);

  auto n_out = torch::empty({start.size(0), walk_length + 1}, start.options());
  auto e_out = torch::empty({start.size(0), walk_length}, start.options());

  CHECK_INPUT(walk_length >= 0);
  if (walk_length == 0) {
    n_out.select(1, 0).copy_(start);
    return std::make_tuple(n_out, e_out);
  }

  auto rowptr_data = rowptr.data_ptr<IndexT>();
  auto col_data = col.data_ptr<IndexT>();
  auto start_data = start.data_ptr<IndexT>();
  auto n_out_data = n_out.data_ptr<IndexT>();
  auto e_out_data = e_out.data_ptr<IndexT>();

  if (p == 1. && q == 1.) {
    uniform_sampling(rowptr_data, col_data, start_data, n_out_data, e_out_data,
                     start.numel(), walk_length);
  } else {
    rejection_sampling(rowptr_data, col_data, start_data, n_out_data,
                       e_out_data, start.numel(), walk_length, p, q);
  }

  return std::make_tuple(n_out, e_out);
}

std::tuple<torch::Tensor, torch::Tensor>
random_walk_cpu(torch::Tensor rowptr, torch::Tensor col, torch::Tensor start,
                int64_t walk_length, double p, double q) {
  if (rowptr.scalar_type() == at::kInt)
    return random_walk_cpu_impl<int32_t>(rowptr, col, start, walk_length, p, q);
  return random_walk_cpu_impl<int64_t>(rowptr, col, start, walk_length, p, q);
}
