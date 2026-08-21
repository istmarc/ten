#ifndef TEN_LINALGEBRA_LEAST_SQUARES_HXX
#define TEN_LINALGEBRA_LEAST_SQUARES_HXX

#include <ten/linalgebra/factorization.hxx>
#include <ten/linalgebra/subtitution.hxx>
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
  ten::tensor<T> _x;
  ls_options _options;

public:
  explicit linear_system(const ls_options &options) : _options(options) {}

  /// Solve Ax=b
  void solve(ten::tensor<T> &A, ten::tensor<T> &b) {
    // TODO Check that A is a matrix and b is a vector
    if (_options._method == ls_method::qr) {
      auto [q, r] = ::ten::linalg::qr(A);
      ::ten::tensor<T> z = ::ten::transposed(q) * b;
      std::size_t n = b.size();
      _x = ten::tensor<T>({n});
      ::ten::linalg::backward_subtitution(r, z, _x);
    } else if (_options._method == ls_method::lu) {
      auto [P, L, U] = ::ten::linalg::lu(A);
      // Solve Lz = t using forward subtitution where z = Ux and t = P^T b
      ::ten::tensor<T> t = ::ten::transposed(P) * b;
      std::size_t n = b.size();
      auto z = ::ten::tensor<T>({n});
      ::ten::linalg::forward_subtitution(L, t, z);
      // Solve Ux = z using backward subtitution
      _x = ten::tensor<T>({n});
      ::ten::linalg::backward_subtitution(U, z, _x);
    } else if (_options._method == ls_method::svd) {
      auto [U, Sigma, Vt] = ::ten::linalg::svd(A);
      std::size_t n = b.size();
      // TODO Make invSigma = ten::fill<diagonal<T>>({n, n}, 1) / Sigma
      ::ten::diagonal<T> invSigma({n, n});
      for (std::size_t i = 0; i < n; i++) {
        invSigma[i] = T(1) / Sigma[i];
      }
      // FIXME Make this work _x = ::ten::transposed(Vt) *
      // ::ten::dense(invSigma) * ::ten::transposed(U) * b;
      ::ten::tensor<T> m =
          ::ten::transposed(Vt) * ::ten::dense(invSigma) * ::ten::transposed(U);
      _x = m * b;
    }
  }

  ::ten::tensor<T> solution() { return _x; }
};

/// Solve Ax=b
template <Tensor M, Tensor V>
  requires(std::is_same_v<typename M::value_type, typename V::value_type>)
auto solve(M &&A, V &&b, const ls_method method = ls_method::qr)
    -> decltype(auto) {
  // TODO Check that A is a matrix and b is a vector
  using value_type = M::value_type;
  ls_options options(method);
  ::ten::linalg::linear_system<value_type> ls(options);
  ls.solve(A, b);
  return ls.solution();
}

/// TODO Linear least squares

/// TODO Nonlinear least squares

} // namespace ten::linalg

#endif
