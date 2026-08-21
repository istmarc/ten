#ifndef TEN_LINALGEBRA_LINEARALGEBRA
#define TEN_LINALGEBRA_LINEARALGEBRA

#include <ten/kernels/blas_api.hxx>
#include <ten/kernels/lapack_api.hxx>
#include <ten/linalgebra/factorization.hxx>
#include <ten/types.hxx>

namespace ten::linalg {

/// Vector norms
enum class vector_norm { l2, l1, linf, lp };

/// Norm of a vector
template <Tensor V>
typename V::value_type norm(const V &v,
                            const vector_norm norm_type = vector_norm::l2) {
  using value_type = V::value_type;

  if (norm_type == vector_norm::l2) {
    value_type r = value_type(0);
    for (size_t i = 0; i < v.size(); i++) {
      r += v[i] * v[i];
    }
    return std::sqrt(r);
  }
  if (norm_type == vector_norm::l1) {
    value_type r = value_type(0);
    for (size_t i = 0; i < v.size(); i++) {
      r += std::abs(v[i]);
    }
    return r;
  }
  if (norm_type == vector_norm::linf) {
    value_type r = std::abs(v[0]);
    for (size_t i = 1; i < v.size(); i++) {
      r = std::max(r, std::abs(v[i]));
    }
    return r;
  }
}

template <Tensor V>
typename V::value_type pnorm(const V &v, const size_t p = 2) {
  using value_type = V::value_type;

  if (p == 0) {
    std::cerr << "Vector norm Lp: p is null" << std::endl;
  }
  value_type r = value_type(0);
  for (size_t i = 0; i < v.size(); i++) {
    r += std::pow(std::abs(v[i]), p);
  }
  r = std::pow(r, value_type(1) / value_type(p));
  return r;
}

/// Matrix norms
enum class matrix_norm {
  frobenius = 1,
  l1 = 2,
  linf = 3,
};

/// Norm of a matrix
template <Tensor M>
typename M::value_type
norm(const M &m, const matrix_norm norm_type = matrix_norm::frobenius) {
  using value_type = M::value_type;

  if (norm_type == matrix_norm::frobenius) {
    value_type r = value_type(0);
    for (size_t i = 0; i < m.size(); i++) {
      r += std::abs(m[i]) * std::abs(m[i]);
    }
    return std::sqrt(r);
  }

  if (norm_type == matrix_norm::l1) {
    size_t p = m.dim(0);
    size_t q = m.dim(1);
    value_type r = std::abs(m(0, 0));
    for (size_t i = 1; i < p; i++) {
      r += std::abs(m(i, 0));
    }
    for (size_t j = 1; j < q; j++) {
      value_type s = std::abs(m(0, j));
      for (size_t i = 1; i < p; i++) {
        s += std::abs(m(i, j));
      }
      r = std::max(r, s);
    }
    return r;
  }

  if (norm_type == matrix_norm::linf) {
    size_t p = m.dim(0);
    size_t q = m.dim(1);
    value_type r = std::abs(m(0, 0));
    for (size_t j = 1; j < q; j++) {
      r += std::abs(m(0, j));
    }
    for (size_t i = 1; i < p; i++) {
      value_type s = std::abs(m(i, 0));
      for (size_t j = 1; j < q; j++) {
        s += std::abs(m(i, j));
      }
      r = std::max(r, s);
    }
    return r;
  }
}

/// dot(a, b)
/// Dot porduct between two vectors
template <Tensor V> typename V::value_type dot(V &a, V &b) {
  size_t n = a.size();
  return ten::kernels::blas::dot(n, a.data(), 1, b.data(), 1);
}

/// outer(a, b)
/// Outer product between two vectors
template <Tensor V> typename V::value_type outer(const V &a, const V &b) {
  using value_type = typename V::value_type;
  std::size_t n = a.size();
  std::size_t m = b.size();
  if (n != m) {
    std::cerr << "Outer product, different vector sizes.";
  }
  ten::tensor<value_type> c({n, n});
  for (std::size_t i = 0; i < n; i++) {
    for (std::size_t j = 0; j < n; j++) {
      c(i, j) = a[i] * b[j];
    }
  }
  return c;
}

/// Compute the inverse of a matrix
template <Tensor M> M inv(const M &a) {
  if (a.dim(0) != a.dim(1)) {
    std::cerr << "Matrix inverse: input is not square" << std::endl;
  }

  using value_type = M::value_type;
  ten::tensor<value_type> x = a.copy();

  auto layout = x.storage_order();
  std::size_t m = x.dim(0);
  std::size_t n = x.dim(1);
  std::size_t lda = x.is_transposed() ? n : m;
  ten::tensor<int32_t> ipiv({n});

  ::ten::kernels::lapack::inv(layout, n, x.data(), lda, ipiv.data());

  return x;
}

// Determinant
template <Tensor M> typename M::value_type det(const M &m) {
  using value_type = M::value_type;

  auto [p, l, u] = ::ten::linalg::lu(m);

  value_type d = 1.;
  // Number of rows exchange
  std::size_t n = 0;
  // Find the determinant of p
  for (std::size_t i = 0; i < m.dim(0); i++) {
    if (p(i, i) == 0.) {
      n += 1;
    }
  }
  if (n % 2 == 1) {
    d = -1;
  }
  // Multiply by det(l) and det(u)
  for (std::size_t i = 0; i < m.dim(0); i++) {
    d *= l(i, i) * u(i, i);
  }
  return d;
}

// TODO pinv

// TODO Power

} // namespace ten::linalg

#endif
