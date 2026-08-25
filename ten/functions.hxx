#ifndef TEN_FUNCTIONS_HXX
#define TEN_FUNCTIONS_HXX

#include <optional>
#include <ten/types.hxx>

namespace ten {

// Blas functions
/// gemm
//template <class T, Expr X, Expr Y, Tensor C>
//   requires(std::is_same_v<T, typename C::value_type>)
//static void gemm(const T /*alpha*/, X && /*x*/, Y && /*y*/, const T /*beta*/,
//                 C & /*c*/);

// Like
template <class T>
ten::tensor<T> like(const ten::tensor<T> & /*x*/, std::optional<bool> requires_grad = std::nullopt);

// Like for scalar
template <class T>
ten::scalar<T> like(const ten::scalar<T> & /*x*/,
                    std::optional<bool> requires_grad = std::nullopt);

} // namespace ten

#endif
