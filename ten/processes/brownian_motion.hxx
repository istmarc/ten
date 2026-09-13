#ifndef TEN_PROCESSES_BROWNIAN_MOTION
#define TEN_PROCESSES_BROWNIAN_MOTION

#include "ten/distributions.hxx"
#include <ten/processes/random_walk.hxx>
#include <ten/tensor>

namespace ten {

/// Brownian motion from a random walk s
template <typename T = float>
tensor<T> brownian_motion(const tensor<T> &s, std::size_t t) {
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
  T ts = 0.;
  std::size_t i = 0;
  T sqrtdeltat = std::sqrt(deltat);
  T sqrtn = std::sqrt(n);
  while (size_t(std::floor(n * ts)) < n) {
    w[i] = cums[size_t(std::floor(n * ts))] * sqrtdeltat / sqrtn;
    ts += deltat;
    i++;
  }
  return w;
}

/// Brownian motion paths
template <typename T = float>
tensor<T> brownian_motion(const tensor<T> &S, std::size_t n, std::size_t t) {
  tensor<T> W({n, t});
  T deltat = 1. / t;
  T sqrtdeltat = std::sqrt(deltat);
  T sqrtn = std::sqrt(n);
  // S(:, j) is a random walk S(:,j) -> W(j,:)
  for (std::size_t j = 0; j < n; j++) {
    // Compute brownian motion of S(:, j)
    tensor<T> cums({n});
    cums[0] = S(0, j);
    for (std::size_t i = 1; i < n; i++) {
      cums[i] = cums[i - 1] + S(i, j);
    }
    // Save the brownian motion in W(i,:)
    T ts = 0.;
    std::size_t i = 0;
    while (size_t(std::floor(n * ts)) < n) {
      W(j, i) = cums[size_t(std::floor(n * ts))] * sqrtdeltat / sqrtn;
      ts += deltat;
      i++;
    }
  }
  return W;
}

// Brownian motion paths from a random walk
template <typename T = float>
tensor<T> brownian_motion(std::size_t n, std::size_t t) {
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
