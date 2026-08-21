#ifndef TEN_MATHS_SIM_HXX
#define TEN_MATHS_SIM_HXX

#include <functional>

#include <ten/random>
#include <ten/types.hxx>

namespace ten {
// Inverse transform method
template <typename T>
tensor<T> inv_transform(std::function<T(T)> Finv, const std::size_t size) {
  tensor<T> u = rand_unif<T>({size});
  tensor<T> x({size});
  for (std::size_t i = 0; i < size; i++) {
    x[i] = Finv(u[i]);
  }
  return x;
}
} // namespace ten

#endif
