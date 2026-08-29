#ifndef TEN_RANDOM_HXX
#define TEN_RANDOM_HXX

#include <ten/distributions.hxx>
#include <ten/types.hxx>

#include <initializer_list>

namespace ten {

/// Random normal tensor
/// rand_norm<tensor<...>, ...>(shape, mean, std, requires_grad, order)
template <Tensor T>
auto rand_norm(const std::vector<std::size_t> &shape,
               const typename T::value_type mean = 0.,
               const typename T::value_type std = 1.,
               const bool requires_grad = false,
               const storage_order order = ten::storage_order::col_major) {
  T x(shape, requires_grad, order);
  using value_type = typename T::value_type;

  ten::normal<value_type> dist(mean, std);
  // Fill with random data
  for (size_t i = 0; i < x.size(); i++) {
    x[i] = dist.sample();
  }
  return x;
}
// rand_norm<tensor<...>>(dims, mean, std, requires_grad, order)
template <Tensor T>
auto rand_norm(std::initializer_list<size_t> &&dims,
               const typename T::value_type mean = 0.,
               const typename T::value_type std = 1.,
               const bool requires_grad = false,
               const storage_order order = ten::storage_order::col_major) {
  std::vector<std::size_t> shape(std::move(dims));
  return rand_norm<T>(shape, mean, std, requires_grad, order);
}

// rand_norm<T>(dims, mean, std, requires_grad, order)
template <typename T = float>
  requires(::ten::is_float<T>::value || ::ten::is_double<T>::value)
auto rand_norm(const std::vector<std::size_t> &shape, const T mean = 0.,
               const T std = 1., const bool requires_grad = false,
               const storage_order order = ten::storage_order::col_major) {
  return rand_norm<ten::tensor<T>>(shape, mean, std, requires_grad, order);
}
template <typename T = float>
  requires(::ten::is_float<T>::value || ::ten::is_double<T>::value)
auto rand_norm(std::initializer_list<std::size_t> &&dims, const T mean = 0.,
               const T std = 1., const bool requires_grad = false,
               const storage_order order = ten::storage_order::col_major) {
  std::vector<std::size_t> shape(std::move(dims));
  return rand_norm<ten::tensor<T>>(shape, mean, std, requires_grad, order);
}

/// Random uniform tensor
/// rand_unif<tensor<...>>(shape, lower_bound, upper_bound, requires_grad,
/// order)
template <Tensor T>
auto rand_unif(const std::vector<std::size_t> &shape,
               const typename T::value_type lower_bound = 0.,
               const typename T::value_type upper_bound = 1.,
               const bool requires_grad = false,
               const storage_order order = ten::storage_order::col_major) {
  T x(shape, requires_grad, order);
  using value_type = typename T::value_type;

  ten::uniform<value_type> dist(lower_bound, upper_bound);
  // Fill with random data
  for (size_t i = 0; i < x.size(); i++) {
    x[i] = dist.sample();
  }
  return x;
}
// rand_unif<tensor<...>>(dims, lower_bound, upper_bound, requires_grad, order)
template <Tensor T>
auto rand_unif(std::initializer_list<size_t> &&dims,
               const typename T::value_type lower_bound = 0.,
               const typename T::value_type upper_bound = 1.,
               const bool requires_grad = false,
               const storage_order order = ten::storage_order::col_major) {
  std::vector<std::size_t> shape(std::move(dims));
  return rand_unif<T>(shape, lower_bound, upper_bound, requires_grad, order);
}

// rand_unif<T>(dims, lower_bound, upper_bound, requires_grad, order)
template <typename T = float>
  requires(::ten::is_float<T>::value || ::ten::is_double<T>::value)
auto rand_unif(const std::vector<std::size_t> &shape, const T lower_bound = 0.,
               const T upper_bound = 1., const bool requires_grad = false,
               const storage_order order = ten::storage_order::col_major) {
  return rand_unif<ten::tensor<T>>(shape, lower_bound, upper_bound,
                                   requires_grad, order);
}
template <typename T = float>
  requires(::ten::is_float<T>::value || ::ten::is_double<T>::value)
auto rand_unif(std::initializer_list<std::size_t> &&dims,
               const T lower_bound = 0., const T upper_bound = 1.,
               const bool requires_grad = false,
               const storage_order order = ten::storage_order::col_major) {

  std::vector<std::size_t> shape(std::move(dims));
  return rand_unif<ten::tensor<T>>(shape, lower_bound, upper_bound,
                                   requires_grad, order);
}

} // namespace ten

#endif
