#ifndef TA_KERNELS_MUL_HXX
#define TA_KERNELS_MUL_HXX

#include <ten/types.hxx>
#include <type_traits>

namespace ten::kernels {

template <Tensor A, Tensor B, Tensor C> static void mul(A &&a, B &&b, C &c) {
  const size_t ranka = a.rank();
  const size_t rankb = b.rank();
  const size_t rankc = c.rank();
  /// Matrix vector multiplication
  if (ranka == 2 && rankb == 1 && rankc == 1) {
    size_t m = a.dim(0);
    size_t n = a.dim(1);
    using blas::transop;
    using T = typename std::remove_cvref_t<A>::value_type;
    const transop transa = (a.is_transposed() ? transop::trans : transop::no);
    const size_t lda = (transa == transop::no ? m : n);
    const size_t incb = 1;
    const size_t incc = 1;
    ::ten::kernels::blas::gemv(transa, m, n, T(1.), a.data(), lda, b.data(),
                               incb, T(0.), c.data(), incc);
  } else if (ranka == 2 && rankb == 2 && rankc == 2) {
    // Multiply two dense matrices
    size_t m = a.dim(0);
    size_t k = a.dim(1);
    size_t n = b.dim(1);
    using blas::transop;
    using T = typename std::remove_cvref_t<A>::value_type;
    const transop transa = (a.is_transposed() ? transop::trans : transop::no);
    const transop transb = (b.is_transposed() ? transop::trans : transop::no);
    const size_t lda = (transa == transop::no ? m : k);
    const size_t ldb = (transa == transop::no ? k : n);
    ::ten::kernels::blas::gemm(transa, transb, m, n, k, T(1.), a.data(), lda,
                               b.data(), ldb, T(0.), c.data(), m);
  } else {
    std::cerr << "mul Incompatible dimensions.\n";
  }
}

/// Multiply and add two dense matrices
/// C <- alpha * A * B + beta * C
template <class T, Tensor A, Tensor B, Tensor C>
  requires(::ten::is_float<T>::value || ::ten::is_double<T>::value)
static void mul_add(const T alpha, A &&a, B &&b, const T beta, C &c) {
  const size_t ranka = a.rank();
  const size_t rankb = b.rank();
  const size_t rankc = c.rank();
  if (ranka == 2 && rankb == 2 && rankc == 2) {
    size_t m = a.dim(0);
    size_t k = a.dim(1);
    size_t n = b.dim(1);
    using blas::transop;
    const transop transa = (a.is_transposed() ? transop::trans : transop::no);
    const transop transb = (b.is_transposed() ? transop::trans : transop::no);
    const size_t lda = (transa == transop::no ? m : k);
    const size_t ldb = (transa == transop::no ? k : n);
    ::ten::kernels::blas::gemm(transa, transb, m, n, k, alpha, a.data(), lda,
                               b.data(), ldb, beta, c.data(), m);
  } else {
    std::cerr << "mul_add Incompatible dimensions.\n";
  }
}

/// scale and add a vector
template <Tensor X, Tensor Y, class T>
static void axpy(const T a, X &&x, Y &y) {
  const size_t rankx = x.rank();
  const size_t ranky = y.rank();
  if (rankx == 1 && ranky == 1) {
    size_t n = x.size();
    ::ten::kernels::blas::axpy(n, a, x.data(), 1, y.data(), 1);
  } else {
    std::cerr << "axpy Incompatible dimensions.\n";
  }
}

} // namespace ten::kernels

#endif
