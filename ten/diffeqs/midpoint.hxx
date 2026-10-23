#ifndef TEN_DIFFEQS_MIDPOINT_HXX
#define TEN_DIFFEQS_MIDPOINT_HXX

#include <ten/tensor>

namespace ten{

/// Midpoint method
template<typename T, class DF>
std::tuple<tensor<T>, tensor<T>> midpoint(T t0, T y0, DF df, T h, std::size_t n) {
  std::size_t m = std::size_t(n/h) + 1;
  tensor<T> t({m});
  t[0] = t0;
  for (std::size_t i = 1; i < m; i++) {
    t[i] = t[i-1] + h;
  }
  tensor<T> y({m});
  y[0] = y0;
  for (std::size_t i = 1; i < m; i++) {
    y[i] = y[i-1] + h * df(t[i-1] + h/T(2), y[i-1] + (h / T(2)) * df(t[i-1], y[i-1]));
  }
  return std::make_tuple(t, y);
}

}

#endif
