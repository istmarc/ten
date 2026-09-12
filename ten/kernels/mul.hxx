#ifndef TA_KERNELS_MUL_HXX
#define TA_KERNELS_MUL_HXX

#include <ten/types.hxx>
#include <type_traits>

namespace ten::kernels {

template <Tensor A, Tensor B, Tensor C> static void mul(A &&a, B &&b, C &c) {
  const std::size_t ranka = a.rank();
  const std::size_t rankb = b.rank();
  const std::size_t rankc = c.rank();
  /// Matrix vector multiplication
  if (ranka == 2 && rankb == 1 && rankc == 1) {
    std::size_t m = a.dim(0);
    std::size_t n = a.dim(1);
    using blas::transop;
    using T = typename std::remove_cvref_t<A>::value_type;
    const transop transa = (a.is_transposed() ? transop::trans : transop::no);
    // mxn and if transposed nxm
    const std::size_t incb = 1;
    const std::size_t incc = 1;
    ::ten::kernels::blas::gemv(transa, m, n, T(1.), a.data(), m, b.data(),
                              incb, T(0.), c.data(), incc);
  } else if (ranka == 2 && rankb == 2 && rankc == 2) {
    // Multiply two dense matrices
    // [m,k] * [k,n] -> [m, n]
    using blas::transop;
    using T = typename std::remove_cvref_t<A>::value_type;
    const transop transa = (a.is_transposed() ? transop::trans : transop::no);
    const transop transb = (b.is_transposed() ? transop::trans : transop::no);
    if (transa == transop::no && transb == transop::no) {
      ::ten::kernels::blas::gemm(transa, transb, a.dim(0), b.dim(1), a.dim(1),
                                 T(1.), a.data(), a.dim(0), b.data(), b.dim(0),
                                 T(0.), c.data(), a.dim(0));
    } else if (transa == transop::no && transb == transop::trans) {
      ::ten::kernels::blas::gemm(transa, transb, a.dim(0), b.dim(0), b.dim(1),
                                 T(1.), a.data(), a.dim(0), b.data(), b.dim(0),
                                 T(0.), c.data(), a.dim(0));
    } else if (transa == transop::trans && transb == transop::no) {
      ::ten::kernels::blas::gemm(transa, transb, a.dim(1), b.dim(1), a.dim(0),
                                 T(1.), a.data(), a.dim(0), b.data(), b.dim(0),
                                 T(0.), c.data(), a.dim(1));
    } else if (transa == transop::trans && transb == transop::trans) {
      ::ten::kernels::blas::gemm(transa, transb, a.dim(1), b.dim(0), a.dim(0),
                                 T(1.), a.data(), a.dim(0), b.data(), b.dim(0),
                                 T(0.), c.data(), a.dim(1));
    }
  } else {
    std::cerr << "mul Incompatible dimensions.\n";
  }
}

/// Multiply and add two dense matrices
/// C <- alpha * A * B + beta * C
template <class T, Tensor A, Tensor B, Tensor C>
  requires(::ten::is_float<T>::value || ::ten::is_double<T>::value)
static void mul_add(const T alpha, A &&a, B &&b, const T beta, C &c) {
  const std::size_t ranka = a.rank();
  const std::size_t rankb = b.rank();
  const std::size_t rankc = c.rank();
  if (ranka == 2 && rankb == 2 && rankc == 2) {
    // Multiply two dense matrices
    // alpha * [m,k] * [k,n] + beta * [m, n]-> [m, n]
    using blas::transop;
    const transop transa = (a.is_transposed() ? transop::trans : transop::no);
    const transop transb = (b.is_transposed() ? transop::trans : transop::no);
    if (transa == transop::no && transb == transop::no) {
      ::ten::kernels::blas::gemm(transa, transb, a.dim(0), b.dim(1), a.dim(1),
                                 alpha, a.data(), a.dim(0), b.data(), b.dim(0),
                                 beta, c.data(), a.dim(0));
    } else if (transa == transop::no && transb == transop::trans) {
      ::ten::kernels::blas::gemm(transa, transb, a.dim(0), b.dim(0), b.dim(1),
                                 alpha, a.data(), a.dim(0), b.data(), b.dim(0),
                                 beta, c.data(), a.dim(0));
    } else if (transa == transop::trans && transb == transop::no) {
      ::ten::kernels::blas::gemm(transa, transb, a.dim(1), b.dim(1), a.dim(0),
                                 alpha, a.data(), a.dim(0), b.data(), b.dim(0),
                                 beta, c.data(), a.dim(1));
    } else if (transa == transop::trans && transb == transop::trans) {
      ::ten::kernels::blas::gemm(transa, transb, a.dim(1), b.dim(0), a.dim(0),
                                 alpha, a.data(), a.dim(0), b.data(), b.dim(0),
                                 beta, c.data(), a.dim(1));
    }
  } else {
    std::cerr << "mul_add Incompatible dimensions.\n";
  }
}

/// scale and add a vector
template <Tensor X, Tensor Y, class T>
static void axpy(const T a, X &&x, Y &y) {
  const std::size_t rankx = x.rank();
  const std::size_t ranky = y.rank();
  if (rankx == 1 && ranky == 1) {
    std::size_t n = x.size();
    ::ten::kernels::blas::axpy(n, a, x.data(), 1, y.data(), 1);
  } else {
    std::cerr << "axpy Incompatible dimensions.\n";
  }
}

} // namespace ten::kernels

#endif
