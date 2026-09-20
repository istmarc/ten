#ifndef TEN_LEARNING_LINEAR_MODEL
#define TEN_LEARNING_LINEAR_MODEL

#include <ten/linalg>
#include <ten/tensor>

namespace ten::ml {

enum class lm_method { qr = 1, lu = 2, svd = 3, gd = 4};

/// Linear model
/// Currently fit only linear regression
template <typename T = float> class linear_model {
private:
  lm_method _method = lm_method::qr;
  tensor<T> _beta = tensor<T>::make_default();
  tensor<T> _yhat = tensor<T>::make_default();

public:
  linear_model(lm_method method = lm_method::qr) : _method(method) {}

  void fit(tensor<T> &x, tensor<T> &y, bool fitted = true) {
    if (x.rank() == 1) {
      std::size_t n = x.size(0);
      tensor<T> X({n, 2});
      X.col(0) = T(1);
      for (std::size_t i = 0; i < n; i++) {
        X(i, 1) = x[i];
      }
      _beta = linalg::lsqr(X, y);
      if (fitted) {
        _yhat = X * _beta;
      }
    } else if (x.rank() == 2) {
      std::size_t n = x.size(0);
      std::size_t k = x.size(1);
      tensor<T> X({n, k + 1});
      X.col(0) = T(1);
      for (std::size_t i = 0; i < n; i++) {
        for (std::size_t j = 0; j < k; j++) {
          X(i, j + 1) = x(i, j);
        }
      }
      _beta = linalg::lsqr(X, y);
      if (fitted) {
        _yhat = X * _beta;
      }
    }
  }

  tensor<T> coef() const { return _beta; }

  tensor<T> fitted() const { return _yhat; }
};

} // namespace ten::ml

#endif
