#ifndef TEN_LEARNING_POLYREG
#define TEN_LEARNING_POLYREG

#include <ten/linalg>
#include <ten/tensor>

namespace ten::details {
template <typename T>
[[nodiscard]] ::ten::tensor<T> polymatrix(::ten::tensor<T> &x,
                                          ::std::size_t n) {
  ::std::size_t m = x.size(0);
  ::ten::tensor<T> X({m, n + 1});
  X.col(0) = T(1);
  for (::std::size_t i = 0; i < m; i++) {
    X(i, 1) = x[i];
  }
  for (::std::size_t i = 2; i < n + 1; i++) {
    for (::std::size_t j = 0; j < m; j++) {
      X(j, i) = X(j, i - 1) * x[j];
    }
  }
  return X;
}
} // namespace ten::details

namespace ten::ml {
/// Polynomial regression
template <typename T = float> class polyreg {
private:
  ::std::size_t _n;
  tensor<T> _beta = tensor<T>::make_default();
  tensor<T> _yhat = tensor<T>::make_default();

public:
  polyreg(std::size_t n) : _n(n) {}

  void fit(tensor<T> &x, tensor<T> &y, bool fitted = true) {
    tensor<T> X = ::ten::details::polymatrix(x, _n);
    _beta = linalg::lsqr(X, y);
    if (fitted) {
      _yhat = X * _beta;
    }
  }

  tensor<T> predict(tensor<T> &x) {
    tensor<T> X = ::ten::details::polymatrix(x, _n);
    tensor<T> yhat = X * _beta;
    return yhat;
  }

  tensor<T> coef() const { return _beta; }

  tensor<T> fitted() const { return _yhat; }
};

} // namespace ten::ml

#endif
