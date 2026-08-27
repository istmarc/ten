#ifndef TEN_MATHS_SIM_HXX
#define TEN_MATHS_SIM_HXX

#include <ten/random>
#include <ten/types.hxx>

namespace ten {
// Inverse transform method / Inverse sampling
template <typename T, class F>
tensor<T> inv_sample(const std::size_t size, F Finv) {
  tensor<T> u = rand_unif<T>({size});
  tensor<T> x({size});
  for (std::size_t i = 0; i < size; i++) {
    x[i] = Finv(u[i]);
  }
  return x;
}
} // namespace ten

#endif
