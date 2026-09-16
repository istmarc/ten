#ifndef TEN_PROCESSES_BROWNIAN_MOTION
#define TEN_PROCESSES_BROWNIAN_MOTION

#include <ten/distributions.hxx>
#include <ten/processes/random_walk.hxx>
#include <ten/tensor>

namespace ten {

/// Brownian motion from a random walk s
template <typename T = float>
std::tuple<tensor<T>, tensor<T>> brownian_motion(const tensor<T> &s, std::size_t t) {
  // s.size() % t must be equal to zero
  T deltat = 1. / t;
  std::size_t n = s.size();
  // Cumulative sum of s
  tensor<T> cums({n});
  cums[0] = s[0];
  for (std::size_t i = 1; i < n; i++) {
    cums[i] = cums[i - 1] + s[i];
  }
  // Brownian motion of s
  tensor<T> w({t});
  tensor<T> index({t});
  T ts = 0.;
  std::size_t i = 0;
  T sqrtdeltat = std::sqrt(deltat);
  T sqrtn = std::sqrt(n);
  while (std::size_t(std::floor(n * ts)) < n) {
    w[i] = cums[std::size_t(std::floor(n * ts))] * sqrtdeltat / sqrtn;
    index[i] = ts;
    ts += deltat;
    i++;
  }
  return std::make_tuple(index, w);
}

/// Brownian motion paths
/// S of shape [n x n]
template <typename T = float>
std::tuple<tensor<T>, tensor<T>> brownian_motion(const tensor<T> &S, std::size_t n, std::size_t t) {
  tensor<T> W({n, t});
  T deltat = 1. / t;
  tensor<T> index({t});
  index[0] = 0.;
  for (std::size_t i = 1; i < t; i++) {
    index[i] = index[i - 1] + deltat;
  }
  T sqrtdeltat = std::sqrt(deltat);
  T sqrtn = std::sqrt(n);
  // A column S(:, j) is a random walk S(:,j) -> W(j,:)
  for (std::size_t j = 0; j < n; j++) {
    // Compute brownian motion of S(:, j)
    tensor<T> cums({n});
    cums[0] = S(0, j);
    for (std::size_t i = 1; i < n; i++) {
      cums[i] = cums[i - 1] + S(i, j);
    }
    // Save the brownian motion in W(j,:)
    T ts = 0.;
    std::size_t i = 0;
    while (std::size_t(std::floor(n * ts)) < n) {
      W(j, i) = cums[std::size_t(std::floor(n * ts))] * sqrtdeltat / sqrtn;
      ts += deltat;
      i++;
    }
  }
  return std::make_tuple(index, W);
}

// Brownian motion paths from a random walk
template <typename T = float>
std::tuple<tensor<T>, tensor<T>> brownian_motion(std::size_t n, std::size_t t) {
  // Random walk S
  uniform<T> dist;
  random_walk_options<T, T> options;
  tensor<T> S({n, n});
  // Each column S(:,j) is a random walk
  for (std::size_t j = 0; j < n; j++) {
    S(0, j) = 0.;
    for (std::size_t i = 1; i < n; i++) {
      S(i, j) = details::random_walk_step(S(i - 1, j), dist, options);
    }
  }
  // Call
  return brownian_motion(S, n, t);
}

} // namespace ten

#endif
