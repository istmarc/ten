#ifndef TEN_PROCESSES_RANDOM_WALK
#define TEN_PROCESSES_RANDOM_WALK

#include <ten/distributions.hxx>
#include <ten/tensor>

namespace ten {
// Random walk options
template <typename T = float, typename Prob = float>
struct random_walk_options {
  T k = .0;
  Prob prob = .5;
  T inc = 1.;
};

namespace details {
// Next step in a random walk
template <typename T, class Prob, class Dist>
T random_walk_step(T x, Dist &dist,
                   const random_walk_options<T, Prob> &options) {
  if (dist.sample() < options.prob) {
    return x + options.inc;
  } else {
    return x - options.inc;
  }
}
} // namespace details

// Simulate a random walk of size n
template <typename T = float, typename Prob = float, class Dist>
tensor<T> random_walk(
    std::size_t n, Dist dist,
    const random_walk_options<T, Prob> &options = random_walk_options()) {
  tensor<T> steps({n});
  steps[0] = options.k;
  for (std::size_t i = 1; i < n; i++) {
    steps[i] = details::random_walk_step(steps[i - 1], dist, options);
  }
  return steps;
}

// Simulate a random walk by default uniform distribution
template <typename T = float, typename Prob = float>
tensor<T> random_walk(
    std::size_t n,
    const random_walk_options<T, Prob> &options = random_walk_options()) {
  uniform dist;
  return random_walk(n, dist, options);
}

/// Continuous random walk
/// s is a random walk
template<typename T = float>
tensor<T> continuous_random_walk(const tensor<T>& s, std::size_t t, std::size_t n) {
  T deltat = t / T(n);
  tensor<T> x({size_t(t / deltat)});
  std::size_t i = 0;
  T ts = 0.;
  while (ts < t) {
    x[i] = std::sqrt(deltat) * s[size_t(std::floor(ts / deltat))];
    ts += deltat;
    i++;
  }
  return x;
}

} // namespace ten

#endif
