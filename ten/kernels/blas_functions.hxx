#ifndef TEN_KERNELS_BLAS_FUNCTIONS_HXX
#define TEN_KERNELS_BLAS_FUNCTIONS_HXX

#include <ten/kernels/blas_api.hxx>
#include <ten/types.hxx>

// Level 1 blas funcions
namespace ten::kernels {
/// asum
template <Tensor T> static auto asum(T &&x) -> decltype(auto) {
  int32_t n = x.size();
  int32_t incx = 1;
  return ::ten::kernels::blas::asum(n, x.data(), incx);
}

template <Column T> static auto asum(T &&x) -> decltype(auto) {
  int32_t n = x.dim(0);
  int32_t incx = 1;
  return ::ten::kernels::blas::asum(n, x.data(), incx);
}

template <Row T> static auto asum(T &&x) -> decltype(auto) {
  int32_t n = x.dim(1);
  int32_t incx = x.dim(0);
  return ::ten::kernels::blas::asum(n, x.data(), incx);
}

/// axpy
template <typename T, Tensor X, Tensor Y>
static void axpy(const T a, X &&x, Y &y) {
  int32_t n = x.size();
  ::ten::kernels::blas::axpy(n, a, x.data(), 1, y.data(), 1);
}

template <typename T, Tensor X, Column Y>
static void axpy(const T a, X &&x, Y &y) {
  int32_t n = x.size();
  ::ten::kernels::blas::axpy(n, a, x.data(), 1, y.data(), 1);
}

template <typename T, Tensor X, Row Y>
static void axpy(const T a, X &&x, Y &y) {
  int32_t n = x.size();
  int32_t incy = y.dim(0);
  ::ten::kernels::blas::axpy(n, a, x.data(), 1, y.data(), incy);
}

template <typename T, Column X, Tensor Y>
static void axpy(const T a, X &&x, Y &y) {
  int32_t n = x.size();
  ::ten::kernels::blas::axpy(n, a, x.data(), 1, y.data(), 1);
}

template <typename T, Column X, Row Y>
static void axpy(const T a, X &&x, Y &y) {
  int32_t n = x.size();
  int32_t incy = y.dim(0);
  ::ten::kernels::blas::axpy(n, a, x.data(), 1, y.data(), incy);
}

template <typename T, Column X, Column Y>
static void axpy(const T a, X &&x, Y &y) {
  int32_t n = x.size();
  ::ten::kernels::blas::axpy(n, a, x.data(), 1, y.data(), 1);
}

template <typename T, Row X, Tensor Y>
static void axpy(const T a, X &&x, Y &y) {
  int32_t n = x.size();
  int32_t incx = x.dim(0);
  ::ten::kernels::blas::axpy(n, a, x.data(), incx, y.data(), 1);
}

template <typename T, Row X, Column Y>
static void axpy(const T a, X &&x, Y &y) {
  int32_t n = x.size();
  int32_t incx = x.dim(0);
  ::ten::kernels::blas::axpy(n, a, x.data(), incx, y.data(), 1);
}

template <typename T, Row X, Row Y> static void axpy(const T a, X &&x, Y &y) {
  int32_t n = x.size();
  int32_t incx = x.dim(0);
  int32_t incy = y.dim(0);
  ::ten::kernels::blas::axpy(n, a, x.data(), incx, y.data(), incy);
}

// copy
template <Tensor X, Tensor Y> static void copy(const X &x, Y &y) {
  int32_t n = x.size();
  ::ten::kernels::blas::copy(n, x.data(), 1, y.data(), 1);
}

template <Tensor X, Column Y> static void copy(const X &x, Y &y) {
  int32_t n = x.size();
  ::ten::kernels::blas::copy(n, x.data(), 1, y.data(), 1);
}

template <Tensor X, Row Y> static void copy(const X &x, Y &y) {
  int32_t n = x.size();
  int32_t incy = y.dim(0);
  ::ten::kernels::blas::copy(n, x.data(), 1, y.data(), incy);
}

template <Column X, Tensor Y> static void copy(const X &x, Y &y) {
  int32_t n = x.size();
  ::ten::kernels::blas::copy(n, x.data(), 1, y.data(), 1);
}

template <Column X, Column Y> static void copy(const X &x, Y &y) {
  int32_t n = x.size();
  ::ten::kernels::blas::copy(n, x.data(), 1, y.data(), 1);
}

template <Column X, Row Y> static void copy(const X &x, Y &y) {
  int32_t n = x.size();
  int32_t incy = y.dim(0);
  ::ten::kernels::blas::copy(n, x.data(), 1, y.data(), incy);
}

template <Row X, Tensor Y> static void copy(const X &x, Y &y) {
  int32_t n = x.size();
  int32_t incx = x.dim(0);
  ::ten::kernels::blas::copy(n, x.data(), incx, y.data(), 1);
}

template <Row X, Column Y> static void copy(const X &x, Y &y) {
  int32_t n = x.size();
  int32_t incx = x.dim(0);
  ::ten::kernels::blas::copy(n, x.data(), incx, y.data(), 1);
}

template <Row X, Row Y> static void copy(const X &x, Y &y) {
  int32_t n = x.size();
  int32_t incx = x.dim(0);
  int32_t incy = y.dim(0);
  ::ten::kernels::blas::copy(n, x.data(), incx, y.data(), incy);
}

// iamax
template <Tensor X> static size_t iamax(const X &x) {
  int32_t n = x.size();
  return ::ten::kernels::blas::iamax(n, x.data(), 1);
}

template <Column X> static size_t iamax(const X &x) {
  int32_t n = x.size();
  return ::ten::kernels::blas::iamax(n, x.data(), 1);
}

template <Row X> static size_t iamax(const X &x) {
  int32_t n = x.size();
  int32_t incx = x.dim(0);
  return ::ten::kernels::blas::iamax(n, x.data(), incx);
}

// dot
template <Tensor X, Tensor Y>
static auto dot(const X &x, const Y &y) -> decltype(auto) {
  int32_t n = x.size();
  return ::ten::kernels::blas::dot(n, x.data(), 1, y.data(), 1);
}

template <Tensor X, Column Y>
static auto dot(const X &x, const Y &y) -> decltype(auto) {
  int32_t n = x.size();
  return ::ten::kernels::blas::dot(n, x.data(), 1, y.data(), 1);
}

template <Tensor X, Row Y>
static auto dot(const X &x, const Y &y) -> decltype(auto) {
  int32_t n = x.size();
  int32_t incy = y.dim(0);
  return ::ten::kernels::blas::dot(n, x.data(), 1, y.data(), incy);
}

template <Column X, Tensor Y>
static auto dot(const X &x, const Y &y) -> decltype(auto) {
  int32_t n = x.size();
  return ::ten::kernels::blas::dot(n, x.data(), 1, y.data(), 1);
}

template <Column X, Column Y>
static auto dot(const X &x, const Y &y) -> decltype(auto) {
  int32_t n = x.size();
  return ::ten::kernels::blas::dot(n, x.data(), 1, y.data(), 1);
}

template <Column X, Row Y>
static auto dot(const X &x, const Y &y) -> decltype(auto) {
  int32_t n = x.size();
  int32_t incy = y.dim(0);
  return ::ten::kernels::blas::dot(n, x.data(), 1, y.data(), incy);
}

template <Row X, Tensor Y>
static auto dot(const X &x, const Y &y) -> decltype(auto) {
  int32_t n = x.size();
  int32_t incx = x.dim(0);
  return ::ten::kernels::blas::dot(n, x.data(), incx, y.data(), 1);
}

template <Row X, Column Y>
static auto dot(const X &x, const Y &y) -> decltype(auto) {
  int32_t n = x.size();
  int32_t incx = x.dim(0);
  return ::ten::kernels::blas::dot(n, x.data(), incx, y.data(), 1);
}

template <Row X, Row Y>
static auto dot(const X &x, const Y &y) -> decltype(auto) {
  int32_t n = x.size();
  int32_t incx = x.dim(0);
  int32_t incy = y.dim(0);
  return ::ten::kernels::blas::dot(n, x.data(), incx, y.data(), incy);
}

// dotc
template <Tensor X, Tensor Y> static auto dotc(const X &x, const Y &y) {
  int32_t n = x.size();
  return ::ten::kernels::blas::dotc(n, x.data(), 1, y.data(), 1);
}

template <Tensor X, Column Y> static auto dotc(const X &x, const Y &y) {
  int32_t n = x.size();
  return ::ten::kernels::blas::dotc(n, x.data(), 1, y.data(), 1);
}

template <Tensor X, Row Y> static auto dotc(const X &x, const Y &y) {
  int32_t n = x.size();
  int32_t incy = y.dim(0);
  return ::ten::kernels::blas::dotc(n, x.data(), 1, y.data(), incy);
}

template <Column X, Tensor Y> static auto dotc(const X &x, const Y &y) {
  int32_t n = x.size();
  return ::ten::kernels::blas::dotc(n, x.data(), 1, y.data(), 1);
}

template <Column X, Column Y> static auto dotc(const X &x, const Y &y) {
  int32_t n = x.size();
  return ::ten::kernels::blas::dotc(n, x.data(), 1, y.data(), 1);
}

template <Column X, Row Y> static auto dotc(const X &x, const Y &y) {
  int32_t n = x.size();
  int32_t incy = y.dim(0);
  return ::ten::kernels::blas::dotc(n, x.data(), 1, y.data(), incy);
}

template <Row X, Tensor Y> static auto dotc(const X &x, const Y &y) {
  int32_t n = x.size();
  int32_t incx = x.dim(0);
  return ::ten::kernels::blas::dotc(n, x.data(), incx, y.data(), 1);
}

template <Row X, Column Y> static auto dotc(const X &x, const Y &y) {
  int32_t n = x.size();
  int32_t incx = x.dim(0);
  return ::ten::kernels::blas::dotc(n, x.data(), incx, y.data(), 1);
}

template <Row X, Row Y> static auto dotc(const X &x, const Y &y) {
  int32_t n = x.size();
  int32_t incx = x.dim(0);
  int32_t incy = y.dim(0);
  return ::ten::kernels::blas::dotc(n, x.data(), incx, y.data(), incy);
}

// nrm2
template <Tensor X> static auto nrm2(const X &x) {
  int32_t n = x.size();
  return ::ten::kernels::blas::nrm2(n, x.data(), 1);
}

template <Column X> static auto nrm2(const X &x) {
  int32_t n = x.size();
  return ::ten::kernels::blas::nrm2(n, x.data(), 1);
}

template <Row X> static auto nrm2(const X &x) {
  int32_t n = x.size();
  int32_t incx = x.dim(0);
  return ::ten::kernels::blas::nrm2(n, x.data(), incx);
}

// scal
template <typename T, Tensor X> static void scal(const T alpha, X &x) {
  int32_t n = x.size();
  ::ten::kernels::blas::scal(n, alpha, x.data(), 1);
}

template <typename T, Column X> static void scal(const T alpha, X &x) {
  int32_t n = x.size();
  ::ten::kernels::blas::scal(n, alpha, x.data(), 1);
}

template <typename T, Row X> static void scal(const T alpha, X &x) {
  int32_t n = x.size();
  int32_t incx = x.dim(0);
  ::ten::kernels::blas::scal(n, alpha, x.data(), incx);
}

// swap
template <Tensor X, Tensor Y> static void swap(X &x, Y &y) {
  int32_t n = x.size();
  ::ten::kernels::blas::swap(n, x.data(), 1, y.data(), 1);
}

template <Tensor X, Column Y> static void swap(X &x, Y &y) {
  int32_t n = x.size();
  ::ten::kernels::blas::swap(n, x.data(), 1, y.data(), 1);
}

template <Tensor X, Row Y> static void swap(X &x, Y &y) {
  int32_t n = x.size();
  int32_t incy = y.dim(0);
  ::ten::kernels::blas::swap(n, x.data(), 1, y.data(), incy);
}

template <Column X, Tensor Y> static void swap(X &x, Y &y) {
  int32_t n = x.size();
  ::ten::kernels::blas::swap(n, x.data(), 1, y.data(), 1);
}

template <Column X, Column Y> static void swap(X &x, Y &y) {
  int32_t n = x.size();
  ::ten::kernels::blas::swap(n, x.data(), 1, y.data(), 1);
}

template <Column X, Row Y> static void swap(X &x, Y &y) {
  int32_t n = x.size();
  int32_t incy = y.dim(0);
  ::ten::kernels::blas::swap(n, x.data(), 1, y.data(), incy);
}

template <Row X, Tensor Y> static void swap(X &x, Y &y) {
  int32_t n = x.size();
  int32_t incx = x.dim(0);
  ::ten::kernels::blas::swap(n, x.data(), incx, y.data(), 1);
}

template <Row X, Column Y> static void swap(X &x, Y &y) {
  int32_t n = x.size();
  int32_t incx = x.dim(0);
  ::ten::kernels::blas::swap(n, x.data(), incx, y.data(), 1);
}

template <Row X, Row Y> static void swap(X &x, Y &y) {
  int32_t n = x.size();
  int32_t incx = x.dim(0);
  int32_t incy = y.dim(1);
  ::ten::kernels::blas::swap(n, x.data(), incx, y.data(), incy);
}

// gemv
// Matrix vector multiplication
template <typename T, Tensor A, Tensor B, Tensor C>
static void gemv(const T alpha, const A &a, const B &b, const T beta, C &c) {
  size_t m = a.dim(0);
  size_t n = a.dim(1);
  using blas::transop;
  const transop transa = (a.is_transposed() ? transop::trans : transop::no);
  const size_t lda = (transa == transop::no ? m : n);
  const size_t incb = 1;
  const size_t incc = 1;
  ::ten::kernels::blas::gemv(transa, m, n, alpha, a.data(), lda, b.data(), incb,
                             beta, c.data(), incc);
}

template <typename T, Tensor A, Tensor B, Column C>
static void gemv(const T alpha, const A &a, const B &b, const T beta, C &c) {
  size_t m = a.dim(0);
  size_t n = a.dim(1);
  using blas::transop;
  const transop transa = (a.is_transposed() ? transop::trans : transop::no);
  const size_t lda = (transa == transop::no ? m : n);
  const size_t incb = 1;
  const size_t incc = 1;
  ::ten::kernels::blas::gemv(transa, m, n, alpha, a.data(), lda, b.data(), incb,
                             beta, c.data(), incc);
}

template <typename T, Tensor A, Tensor B, Row C>
static void gemv(const T alpha, const A &a, const B &b, const T beta, C &c) {
  size_t m = a.dim(0);
  size_t n = a.dim(1);
  using blas::transop;
  const transop transa = (a.is_transposed() ? transop::trans : transop::no);
  const size_t lda = (transa == transop::no ? m : n);
  const size_t incb = 1;
  const size_t incc = c.dim(0);
  ::ten::kernels::blas::gemv(transa, m, n, alpha, a.data(), lda, b.data(), incb,
                             beta, c.data(), incc);
}

template <typename T, Tensor A, Column B, Tensor C>
static void gemv(const T alpha, const A &a, const B &b, const T beta, C &c) {
  size_t m = a.dim(0);
  size_t n = a.dim(1);
  using blas::transop;
  const transop transa = (a.is_transposed() ? transop::trans : transop::no);
  const size_t lda = (transa == transop::no ? m : n);
  const size_t incb = 1;
  const size_t incc = 1;
  ::ten::kernels::blas::gemv(transa, m, n, alpha, a.data(), lda, b.data(), incb,
                             beta, c.data(), incc);
}

template <typename T, Tensor A, Column B, Row C>
static void gemv(const T alpha, const A &a, const B &b, const T beta, C &c) {
  size_t m = a.dim(0);
  size_t n = a.dim(1);
  using blas::transop;
  const transop transa = (a.is_transposed() ? transop::trans : transop::no);
  const size_t lda = (transa == transop::no ? m : n);
  const size_t incb = 1;
  const size_t incc = c.dim(0);
  ::ten::kernels::blas::gemv(transa, m, n, alpha, a.data(), lda, b.data(), incb,
                             beta, c.data(), incc);
}

template <typename T, Tensor A, Row B, Tensor C>
static void gemv(const T alpha, const A &a, const B &b, const T beta, C &c) {
  size_t m = a.dim(0);
  size_t n = a.dim(1);
  using blas::transop;
  const transop transa = (a.is_transposed() ? transop::trans : transop::no);
  const size_t lda = (transa == transop::no ? m : n);
  const size_t incb = b.dim(0);
  const size_t incc = 1;
  ::ten::kernels::blas::gemv(transa, m, n, alpha, a.data(), lda, b.data(), incb,
                             beta, c.data(), incc);
}

template <typename T, Tensor A, Row B, Column C>
static void gemv(const T alpha, const A &a, const B &b, const T beta, C &c) {
  size_t m = a.dim(0);
  size_t n = a.dim(1);
  using blas::transop;
  const transop transa = (a.is_transposed() ? transop::trans : transop::no);
  const size_t lda = (transa == transop::no ? m : n);
  const size_t incb = b.dim(0);
  const size_t incc = 1;
  ::ten::kernels::blas::gemv(transa, m, n, alpha, a.data(), lda, b.data(), incb,
                             beta, c.data(), incc);
}

template <typename T, Tensor A, Row B, Row C>
static void gemv(const T alpha, const A &a, const B &b, const T beta, C &c) {
  size_t m = a.dim(0);
  size_t n = a.dim(1);
  using blas::transop;
  const transop transa = (a.is_transposed() ? transop::trans : transop::no);
  const size_t lda = (transa == transop::no ? m : n);
  const size_t incb = b.dim(0);
  const size_t incc = c.dim(0);
  ::ten::kernels::blas::gemv(transa, m, n, alpha, a.data(), lda, b.data(), incb,
                             beta, c.data(), incc);
}

// ger
// Rank one update of a matrix
template <typename T, Tensor X, Tensor Y, Tensor A>
static void ger(const T alpha, const X &x, const Y &y, A &a) {
  size_t m = a.dim(0);
  size_t n = a.dim(1);
  using blas::transop;
  const transop transa = (a.is_transposed() ? transop::trans : transop::no);
  const size_t lda = (transa == transop::no ? m : n);
  const size_t incx = 1;
  const size_t incy = 1;
  ::ten::kernels::blas::ger(m, n, alpha, x.data(), incx, y.data(), incy,
                            a.data(), lda);
}

template <typename T, Tensor X, Column Y, Tensor A>
static void ger(const T alpha, const X &x, const Y &y, A &a) {
  size_t m = a.dim(0);
  size_t n = a.dim(1);
  using blas::transop;
  const transop transa = (a.is_transposed() ? transop::trans : transop::no);
  const size_t lda = (transa == transop::no ? m : n);
  const size_t incx = 1;
  const size_t incy = 1;
  ::ten::kernels::blas::ger(m, n, alpha, x.data(), incx, y.data(), incy,
                            a.data(), lda);
}

template <typename T, Tensor X, Row Y, Tensor A>
static void ger(const T alpha, const X &x, const Y &y, A &a) {
  size_t m = a.dim(0);
  size_t n = a.dim(1);
  using blas::transop;
  const transop transa = (a.is_transposed() ? transop::trans : transop::no);
  const size_t lda = (transa == transop::no ? m : n);
  const size_t incx = 1;
  const size_t incy = y.dim(0);
  ::ten::kernels::blas::ger(m, n, alpha, x.data(), incx, y.data(), incy,
                            a.data(), lda);
}

template <typename T, Column X, Tensor Y, Tensor A>
static void ger(const T alpha, const X &x, const Y &y, A &a) {
  size_t m = a.dim(0);
  size_t n = a.dim(1);
  using blas::transop;
  const transop transa = (a.is_transposed() ? transop::trans : transop::no);
  const size_t lda = (transa == transop::no ? m : n);
  const size_t incx = 1;
  const size_t incy = 1;
  ::ten::kernels::blas::ger(m, n, alpha, x.data(), incx, y.data(), incy,
                            a.data(), lda);
}

template <typename T, Column X, Column Y, Tensor A>
static void ger(const T alpha, const X &x, const Y &y, A &a) {
  size_t m = a.dim(0);
  size_t n = a.dim(1);
  using blas::transop;
  const transop transa = (a.is_transposed() ? transop::trans : transop::no);
  const size_t lda = (transa == transop::no ? m : n);
  const size_t incx = 1;
  const size_t incy = 1;
  ::ten::kernels::blas::ger(m, n, alpha, x.data(), incx, y.data(), incy,
                            a.data(), lda);
}

template <typename T, Column X, Row Y, Tensor A>
static void ger(const T alpha, const X &x, const Y &y, A &a) {
  size_t m = a.dim(0);
  size_t n = a.dim(1);
  using blas::transop;
  const transop transa = (a.is_transposed() ? transop::trans : transop::no);
  const size_t lda = (transa == transop::no ? m : n);
  const size_t incx = 1;
  const size_t incy = y.dim(0);
  ::ten::kernels::blas::ger(m, n, alpha, x.data(), incx, y.data(), incy,
                            a.data(), lda);
}

template <typename T, Row X, Tensor Y, Tensor A>
static void ger(const T alpha, const X &x, const Y &y, A &a) {
  size_t m = a.dim(0);
  size_t n = a.dim(1);
  using blas::transop;
  const transop transa = (a.is_transposed() ? transop::trans : transop::no);
  const size_t lda = (transa == transop::no ? m : n);
  const size_t incx = x.dim(0);
  const size_t incy = 1;
  ::ten::kernels::blas::ger(m, n, alpha, x.data(), incx, y.data(), incy,
                            a.data(), lda);
}

template <typename T, Row X, Column Y, Tensor A>
static void ger(const T alpha, const X &x, const Y &y, A &a) {
  size_t m = a.dim(0);
  size_t n = a.dim(1);
  using blas::transop;
  const transop transa = (a.is_transposed() ? transop::trans : transop::no);
  const size_t lda = (transa == transop::no ? m : n);
  const size_t incx = x.dim(0);
  const size_t incy = 1;
  ::ten::kernels::blas::ger(m, n, alpha, x.data(), incx, y.data(), incy,
                            a.data(), lda);
}

template <typename T, Row X, Row Y, Tensor A>
static void ger(const T alpha, const X &x, const Y &y, A &a) {
  size_t m = a.dim(0);
  size_t n = a.dim(1);
  using blas::transop;
  const transop transa = (a.is_transposed() ? transop::trans : transop::no);
  const size_t lda = (transa == transop::no ? m : n);
  const size_t incx = x.dim(0);
  const size_t incy = y.dim(0);
  ::ten::kernels::blas::ger(m, n, alpha, x.data(), incx, y.data(), incy,
                            a.data(), lda);
}

} // namespace ten::kernels

#endif
