#ifndef TEN_LINALGEBRA_LEAST_SQUARES_HXX
#define TEN_LINALGEBRA_LEAST_SQUARES_HXX

#include <ten/linalgebra/factorization.hxx>
#include <ten/linalgebra/linalgebra.hxx>
#include <ten/linalgebra/subtitution.hxx>
#include <ten/random>
#include <ten/tensor>
#include <ten/types.hxx>

#include <type_traits>

namespace ten::linalg {

enum class ls_method { qr = 1, lu = 2, svd = 3 };

struct ls_options {
  ls_method _method = ls_method::qr;

  ls_options(ls_method method) : _method(method) {}
};

/// Linear system
template <class T = float> class linear_system {
  static_assert(std::is_floating_point_v<T>, "T must be floating point");

private:
  ten::tensor<T> _x = ten::tensor<T>::make_default();
  ls_options _options;

public:
  explicit linear_system(const ls_options &options) : _options(options) {}
  explicit linear_system(ls_options &&options) : _options(std::move(options)) {}

  /// Solve Ax=b
  void solve(ten::tensor<T> &A, ten::tensor<T> &b) {
    if (_options._method == ls_method::qr) {
      auto [q, r] = qr(A);
      ten::tensor<T> z = ten::transposed(q) * b;
      std::size_t n = b.size();
      _x = ten::tensor<T>({n});
      backward_subtitution(r, z, _x);
    } else if (_options._method == ls_method::lu) {
      auto [P, L, U] = lu(A);
      // Solve Lz = t using forward subtitution where z = Ux and t = P^T b
      ten::tensor<T> t = ten::transposed(P) * b;
      std::size_t n = b.size();
      auto z = ten::tensor<T>({n});
      forward_subtitution(L, t, z);
      // Solve Ux = z using backward subtitution
      _x = ten::tensor<T>({n});
      backward_subtitution(U, z, _x);
    } else if (_options._method == ls_method::svd) {
      auto [U, Sigma, Vt] = svd(A);
      std::size_t n = b.size();
      // TODO Make invSigma = ten::fill<diagonal<T>>({n, n}, 1) / Sigma
      ten::diagonal<T> invSigma({n, n});
      for (std::size_t i = 0; i < n; i++) {
        invSigma[i] = T(1) / Sigma[i];
      }
      // FIXME Make this work _x = ::ten::transposed(Vt) *
      // ::ten::dense(invSigma) * ::ten::transposed(U) * b;
      ten::tensor<T> m =
          ten::transposed(Vt) * ten::dense(invSigma) * ten::transposed(U);
      _x = m * b;
    }
  }

  ten::tensor<T> solution() { return _x; }
};

/// Solve Ax=b
template <Tensor T>
auto solve(T &&A, T &&b, const ls_method method = ls_method::qr)
    -> decltype(auto) {
  if (A.rank() != 2) {
    std::cerr << "ten::linalg::solve, A must be a matrix.\n";
  }
  if (b.rank() != 1) {
    std::cerr << "ten::linalg::solve, b must be a vector.\n";
  }
  using value_type = std::remove_cvref_t<T>::value_type;
  ls_options options(method);
  linear_system<value_type> ls(options);
  ls.solve(A, b);
  return ls.solution();
}

/// Linear least squares
/// min_beta ||y - X beta||2
template <Tensor T>
auto lsqr(T &&X, T &&y, ls_method method = ls_method::qr) -> decltype(auto) {
  using value_type = std::remove_cvref_t<T>::value_type;
  // We wan to find beta such that X beta = y
  ten::tensor<value_type> beta({X.dim(1)});
  if (method == ls_method::qr) {
    auto [q, r] = qr(X);
    // R beta = QT y, set z = QT y and solve Rbeta = z
    ten::tensor<value_type> z = ten::transposed(q) * y;
    backward_subtitution(r, z, beta);
  }
  return beta;
}

enum class nls_method {
  gauss_newton = 1,
  mardquardt = 2,
  qr = 3,
  svd = 4,
  newton = 5
};

/// Non linear least squares options
template <class T> struct nls_options {
  using value_type = T::value_type;

  nls_method method = nls_method::gauss_newton;

  std::size_t n;
  std::optional<T> beta0 = std::nullopt;
  std::optional<T> H = std::nullopt;
  std::optional<T> W = std::nullopt;
  std::size_t itermax = 1000;
  value_type eps = 1e-3;
};

/// Nonlinear least squares
/// min ||f(x,beta)-y||2
template <class F, class Jacobian, Tensor T, Tensor R>
auto nls_newton_gauss(F f, T &&x, T &&y, Jacobian Jr, nls_options<R> options)
    -> decltype(auto) {
  using value_type = std::remove_cvref_t<T>::value_type;
  static_assert(std::is_same_v<value_type, typename R::value_type>,
                "Tensors must have the same value type.");
  std::size_t m = x.size();
  std::size_t n = options.n;

  // Gauss newton algorithm (JtJ)Delta_beta = Jt Delta_y
  tensor<value_type> beta = options.beta0.has_value()
                                ? options.beta0.value().copy()
                                : ten::rand_norm({n});
  tensor<value_type> beta0 = beta.copy();
  tensor<value_type> res({m});
  tensor<value_type> diff({m});
  for (std::size_t i = 0; i < options.itermax; i++) {
    // Compute the jacobian
    tensor<value_type> J = Jr(x, beta);
    // Compute the residuals
    res = y - f(x, beta);
    tensor<value_type> JtJ = ten::transposed(J) * J;
    tensor<value_type> JtRes = ten::transposed(J) * res;
    tensor<value_type> delta = ten::linalg::solve(JtJ, JtRes);
    // Update beta
    for (std::size_t k = 0; k < m; k++) {
      beta[k] -= delta[k];
    }
    value_type s = .0;
    for (std::size_t k = 0; k < m; k++) {
      s += std::abs(delta[k]);
    }
    if (std::sqrt(s) < options.eps) {
      break;
    }
  }
  return beta;
}

} // namespace ten::linalg

#endif
