#ifndef TEN_FFTS_FTT_HXX
#define TEN_FFTS_FTT_HXX

#include <numbers>

#include <ten/types.hxx>
#include <ten/tensor>

namespace ten{

/// FFT slow
template<typename T>
tensor<std::complex<T>> fft(const tensor<T>& x) {
  std::size_t n = x.size();
  ten::tensor<std::complex<T>> y({n});
  for (std::size_t k = 0; k < n; k++) {
    y[k] = T(0);
    for (std::size_t m = 0; m < n; m++) {
      T angle = -2.0 * std::numbers::pi_v<T> * k * m / T(n);
      std::complex<T> root(std::cos(angle), std::sin(angle));
      y[k] += x[m] * root;
    }
  }
  return y;
}

}

#endif
