/// \file Ten/Types.hxx

#ifndef TEN_TYPES_HXX
#define TEN_TYPES_HXX

#include <complex>
#include <concepts>
#include <iostream>
#include <limits>
#include <memory>
#include <type_traits>
#include <variant>

namespace ten {

/// \enum data_type
enum class data_type : uint16_t {
  float32 = 1,
  float64 = 2,
  boolean = 4,
  int32 = 8,
  int64 = 16,
  uint32 = 32,
  uint64 = 64,
  complexfloat32 = 128,
  complexfloat64 = 256,
  string = 512,
};

/// \enum format
/// Format type
enum class storage_format : uint16_t {
  /// Dense format
  dense = 1,
  /// Sparse coordinate format
  coo = 2,
  /// Compressed sparse colums
  csc = 4,
  /// Compressed sparse rows
  csr = 8,
  /// Diagonal format
  diagonal = 16,
  /// Lower triangular format
  lower_tr = 32,
  /// Upper triangular format
  upper_tr = 64,
  /// Symmetric
  symmetric = 128,
  /// Transposed
  transposed = 256,
  /// Hermitian
  hermitian = 512,
  /// Conjugated
  conj = 1024
};

bool operator&(storage_format a, storage_format b) {
  return static_cast<uint16_t>(a) & static_cast<uint16_t>(b);
}

storage_format operator|(storage_format a, storage_format b) {
  return static_cast<storage_format>(static_cast<uint16_t>(a) |
                                     static_cast<uint16_t>(b));
}

void operator|=(storage_format &a, const storage_format b) {
  a = static_cast<storage_format>(static_cast<uint16_t>(a) |
                                  static_cast<uint16_t>(b));
}

#ifndef TENSEUR_SIZE_TYPE
#define TENSEUR_SIZE_TYPE std::size_t
#endif

/// \typedef size_type
/// Type of the indices
using size_type = TENSEUR_SIZE_TYPE;

/// Traits for std::complex
template <class> struct is_complex : std::false_type {};
template <class T> struct is_complex<std::complex<T>> : std::true_type {};

// Traits for float
template <class> struct is_float : std::false_type {};
template <> struct is_float<float> : std::true_type {};

// Traits for double
template <class> struct is_double : std::false_type {};
template <> struct is_double<double> : std::true_type {};

// Traits for int32
template <class> struct is_int32 : std::false_type {};
template <> struct is_int32<int32_t> : std::true_type {};

// Traits for uint32
template <class> struct is_uint32 : std::false_type {};
template <> struct is_uint32<uint32_t> : std::true_type {};

// Traits for int64
template <class> struct is_int64 : std::false_type {};
template <> struct is_int64<int64_t> : std::true_type {};

// Traits for uint64
template <class> struct is_uint64 : std::false_type {};
template <> struct is_uint64<uint64_t> : std::true_type {};

// Traits for bool
template <class> struct is_bool : std::false_type {};
template <> struct is_bool<bool> : std::true_type {};

/// \enum storage_order
/// Storage order of a multidimentional array
enum class storage_order { col_major, row_major };

/// Default storage order
static constexpr storage_order default_order = storage_order::col_major;

/// \class tensor_base
/// Base class for tensor types
class tensor_base {
public:
  virtual ~tensor_base() {}
};

/// Tensor type
template <typename T> class tensor;

/// Column type
template <typename T> class column;

/// Row type
template <typename T> class row;

/// is_column
template <class> struct is_column : std::false_type {};
template <typename T> struct is_column<column<T>> : std::true_type {};

template <class T> static constexpr bool is_column_v = is_column<T>::value;

/// is_row
template <class> struct is_row : std::false_type {};
template <typename T> struct is_row<row<T>> : std::true_type {};

template <class T> static constexpr bool is_row_v = is_row<T>::value;

/// Concept Column
template <class T>
concept Column = is_column<std::remove_cvref_t<T>>::value;

/// Concept Row
template <class T>
concept Row = is_row<std::remove_cvref_t<T>>::value;

// Storage
template <typename T> class dense_storage;
template <typename T> class diagonal_storage;
template <typename T> class sparse_storage;

// Storage trait for special storages
template <class> struct is_diagonal_storage : std::false_type {};
template <class T>
struct is_diagonal_storage<diagonal_storage<T>> : std::true_type {};

////////////////////////////////////////////////////////////////////////////////
// Storage types

template <class> struct is_storage : std::false_type {};
template <typename T> struct is_storage<dense_storage<T>> : std::true_type {};

template <typename T>
struct is_storage<diagonal_storage<T>> : std::true_type {};

template <typename T> struct is_storage<sparse_storage<T>> : std::true_type {};

////////////////////////////////////////////////////////////////////////////////

// Scalar
template <typename T> class scalar;

// Scalar node
template <typename T> class scalar_node;

// tensor node
template <typename T> class tensor_node;

// scalar_node traits
template <class> struct is_scalar_node : std::false_type {};
template <typename T> struct is_scalar_node<scalar_node<T>> : std::true_type {};

// tensor_node traits
template <class> struct is_tensor_node : std::false_type {};
template <typename T> struct is_tensor_node<tensor_node<T>> : std::true_type {};

// Unary expr
template <typename T> class unary_expr;

template <class> struct is_unary_expr : std::false_type {};
template <typename T> struct is_unary_expr<unary_expr<T>> : std::true_type {};

template <typename T>
static constexpr bool is_unary_expr_v = is_unary_expr<T>::value;

// Binary expr
template <typename T> class binary_expr;

template <class> struct is_binary_expr : std::false_type {};
template <typename T> struct is_binary_expr<binary_expr<T>> : std::true_type {};

template <typename T>
static constexpr bool is_binary_expr_v = is_binary_expr<T>::value;

/////////////////////////////////////////////////////////////////////////////////
// Sparse tensors

template <class> struct is_sparse_storage : std::false_type {};
template <class T>
struct is_sparse_storage<sparse_storage<T>> : std::true_type {};

// Forward declaration of sparse tensor
template <typename T> class sparse_tensor;

template <class> struct is_sparse_tensor : std::false_type {};

template <typename T>
struct is_sparse_tensor<sparse_tensor<T>> : std::true_type {};

template <typename T>
static constexpr bool is_sparse_tensor_v = is_sparse_tensor<T>::value;

template <class T>
concept SparseTensor = is_sparse_tensor<std::remove_cvref_t<T>>::value;

// Traits for scalar
template <class> struct is_scalar : std::false_type {};
template <typename T> struct is_scalar<scalar<T>> : std::true_type {};

template <typename T> static constexpr bool is_scalar_v = is_scalar<T>::value;

// Scalar concept
template <class T>
concept Scalar = is_scalar<std::remove_cvref_t<T>>::value;

/////////////////////////////////////////////////////////////////////////////////
// Concepts

/// Unary node
template <class T>
concept UnaryExpr = is_unary_expr<std::remove_cvref_t<T>>::value;

/// Binary node
template <class T>
concept BinaryExpr = is_binary_expr<std::remove_cvref_t<T>>::value;

// Concepts

/// Scalar node
template <typename T>
concept ScalarNode = is_scalar_node<std::remove_cvref_t<T>>::value;

////////////////////////////////////////////////////////////////////////////////
/// Tensor node
// DenseStorage<T>
template <typename T>
concept TensorNode = is_tensor_node<std::remove_cvref_t<T>>::value;

/// trait is_tensor
template <typename> struct is_tensor : std::false_type {};
template <typename T> struct is_tensor<tensor<T>> {
  static constexpr bool value = true;
};

template <typename T> static constexpr bool is_tensor_v = is_tensor<T>::value;

// Concept Tensor
template <class T>
concept Tensor = is_tensor<std::remove_cvref_t<T>>::value;

////////////////////////////////////////////////////////////////////////////////
// Expr
class expr {
protected:
  expr() = default;
  expr(const expr &) = default;
  expr(expr &&) = default;
};

template <class T> struct is_expr {
  static constexpr bool value = std::is_base_of_v<::ten::expr, T>;
};

// Concept for Expression type
template <class T>
concept Expr = is_expr<std::remove_cvref_t<T>>::value;

////////////////////////////////////////////////////////////////////////////////
// Operations

// Binary operations
enum class binary_operation { add, sub, div, mul };
// Trigonometric operations
enum class trig_operation { sin, cos, tan };

} // namespace ten

#endif
