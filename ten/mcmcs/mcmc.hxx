#ifndef TEN_MCMCS_MCMC
#define TEN_MCMCS_MCMC

#include "ten/distributions.hxx"
#include <ten/tensor>

namespace ten {

/// MCMC step
template <typename T, class F, class G, class Distribution>
T mcmc_step(T xt, F f, G g, Distribution &dist) {
  // Sample x ~ g(x / xt)
  T x = g(xt);
  // Compute the acceptance probability
  T a = f(x) / f(xt);
  // Accept or reject
  T u = dist.sample();
  if (u <= a) {
    return x;
  }
  return xt;
}

/// Markov Chain Monte Carlo
template <typename T, class F, class G>
ten::tensor<T> mcmc(T xt, std::size_t n, F f, G g, std::size_t burn = 0) {
  ten::tensor<T> sample({n});

  ten::uniform<T> dist;
  T x;

  for (std::size_t i = 0; i < burn; i++) {
    x = mcmc_step(xt, f, g, dist);
    if (xt != x) {
      xt = x;
    }
  }

  std::size_t k = 0;
  for (std::size_t i = 0; i < n; i++) {
    x = mcmc_step(xt, f, g, dist);
    if (xt != x) {
      sample[k] = x;
      k++;
      xt = x;
    }
  }

  if (k == n) {
    return sample;
  } else {
    ten::tensor<T> new_sample({k});
    for (std::size_t i = 0; i < k; i++) {
      new_sample[i] = sample[i];
    }
    return new_sample;
  }
}

/// Markov Chain Monte Carlo without specifying g
template <typename T, class F>
ten::tensor<T> mcmc(T xt, std::size_t n, F f, std::size_t burn = 0) {
  ten::uniform<T> dist(-1., 1.);
  std::function<T(T)> g = [&dist](T y) { return y + dist.sample(); };
  return mcmc(xt, n, f, g, burn);
}

} // namespace ten

#endif
