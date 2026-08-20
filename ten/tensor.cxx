#include <cmath>

#include <pybind11/attr.h>

#include <ten/types.hxx>

#include <ten/io>
#include <ten/random>
#include <ten/tensor>

#include <pybind11/operators.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

namespace py = pybind11;

// vector, matrix and tensor
using tensor_float = ten::tensor<float>;
using tensor_double = ten::tensor<double>;
using tensor_int32 = ten::tensor<int32_t>;
using tensor_int64 = ten::tensor<int64_t>;
using tensor_uint32 = ten::tensor<uint32_t>;
using tensor_uint64 = ten::tensor<uint64_t>;
using tensor_bool = ten::tensor<bool>;
using tensor_c = ten::tensor<std::complex<float>>;
using tensor_z = ten::tensor<std::complex<double>>;

using diagonal_float = ten::diagonal<float>;
using diagonal_double = ten::diagonal<double>;
using diagonal_int32 = ten::diagonal<int32_t>;
using diagonal_int64 = ten::diagonal<int64_t>;
;
using diagonal_uint32 = ten::diagonal<uint32_t>;
using diagonal_uint64 = ten::diagonal<uint64_t>;

// Scalars
using scalar_float = ten::scalar<float>;
using scalar_double = ten::scalar<double>;
using scalar_int32 = ten::scalar<int32_t>;
using scalar_int64 = ten::scalar<int64_t>;
using scalar_uint32 = ten::scalar<uint32_t>;
using scalar_uint64 = ten::scalar<uint64_t>;

// Scalar nodes
/*
using scalarnode_float = scalar_float::node_type;
using scalarnode_double = scalar_double::node_type;
using scalarnode_int32 = scalar_int32::node_type;
using scalarnode_int64 = scalar_int64::node_type;
using scalarnode_uint32 = scalar_uint32::node_type;
using scalarnode_uint64 = scalar_uint64::node_type;
*/

// Tensor nodes
using tensornode_float = tensor_float::node_type;
using tensornode_double = tensor_double::node_type;
using tensornode_int32 = tensor_int32::node_type;
using tensornode_int64 = tensor_int64::node_type;
using tensornode_uint32 = tensor_uint32::node_type;
using tensornode_uint64 = tensor_uint64::node_type;
using tensornode_bool = tensor_bool::node_type;

////////////////////////////////////////////////////////////////////////////////
// Functions
/*
template <typename T> auto min_py(T tensor) {
   using value_type = T::value_type;
   return ten::unary_expr<T, ten::scalar<value_type>, ten::functional::min>(
       tensor);
}

template <typename T> auto max_py(T tensor) {
   using value_type = T::value_type;
   return ten::unary_expr<T, ten::scalar<value_type>, ten::functional::max>(
       tensor);
}*/

// TODO Unary expr
using expr_sqrt_tensor_float =
    ten::unary_expr<tensor_float, tensor_float,
                    ten::functional::sqrt<tensor_float, tensor_float>>;
using expr_sqrt_tensor_double =
    ten::unary_expr<tensor_double, tensor_double,
                    ten::functional::sqrt<tensor_double, tensor_double>>;

/*
using expr_sqr_tensor_float =
    ten::unary_expr<tensor_float, tensor_float,
                    ten::functional::sqrt<tensor_float, tensor_float>>;
using expr_sqr_tensor_double =
    ten::unary_expr<tensor_double, tensor_double,
                    ten::functional::sqrt<tensor_double, tensor_double>>;
*/

/*
using expr_min_vector_float =
    ten::unary_expr<vector_float, ten::scalar<float>, ten::functional::min>;
using expr_min_vector_double =
    ten::unary_expr<vector_double, ten::scalar<double>, ten::functional::min>;
using expr_min_matrix_float =
    ten::unary_expr<matrix_float, ten::scalar<float>, ten::functional::min>;
using expr_min_matrix_double =
    ten::unary_expr<matrix_double, ten::scalar<double>, ten::functional::min>;
using expr_min_tensor3_float =
    ten::unary_expr<tensor3_float, ten::scalar<float>, ten::functional::min>;
using expr_min_tensor3_double =
    ten::unary_expr<tensor3_double, ten::scalar<double>, ten::functional::min>;
using expr_min_tensor4_float =
    ten::unary_expr<tensor4_float, ten::scalar<float>, ten::functional::min>;
using expr_min_tensor4_double =
    ten::unary_expr<tensor4_double, ten::scalar<double>, ten::functional::min>;
using expr_min_tensor5_float =
    ten::unary_expr<tensor5_float, ten::scalar<float>, ten::functional::min>;
using expr_min_tensor5_double =
    ten::unary_expr<tensor5_double, ten::scalar<double>, ten::functional::min>;
*/

// Reshape
/*using expr_reshape_vector_float_matrix_float =
    ten::unary_expr<vector_float, matrix_float,
                    ten::functional::dynamic_reshape<ten::shape<0, 0>>::func>;
using expr_reshape_vector_float_tensor3_float = ten::unary_expr<
    vector_float, tensor3_float,
    ten::functional::dynamic_reshape<ten::shape<0, 0, 0>>::func>;
using expr_reshape_vector_float_tensor4_float = ten::unary_expr<
    vector_float, tensor4_float,
    ten::functional::dynamic_reshape<ten::shape<0, 0, 0, 0>>::func>;
using expr_reshape_vector_float_tensor5_float = ten::unary_expr<
    vector_float, tensor5_float,
    ten::functional::dynamic_reshape<ten::shape<0, 0, 0, 0, 0>>::func>;
*/

// Binary expr
/*using add_vector_float = ten::binary_expr<
    vector_float, vector_float, vector_float,
    ten::functional::binary_func<ten::binary_operation::add>::template func>;
using sub_vector_float = ten::binary_expr<
    vector_float, vector_float, vector_float,
    ten::functional::binary_func<ten::binary_operation::sub>::template func>;
using mul_vector_float = ten::binary_expr<
    vector_float, vector_float, vector_float,
    ten::functional::binary_func<ten::binary_operation::mul>::template func>;
using div_vector_float = ten::binary_expr<
    vector_float, vector_float, vector_float,
    ten::functional::binary_func<ten::binary_operation::div>::template func>;
*/

/*
using add_diagonal_float = ten::binary_expr<
    diagonal_float, diagonal_float, diagonal_float,
    ten::functional::binary_func<ten::binary_operation::add>::template func>;
using sub_diagonal_float = ten::binary_expr<
    diagonal_float, diagonal_float, diagonal_float,
    ten::functional::binary_func<ten::binary_operation::sub>::template func>;
// using mul_diagonal_float = ten::binary_expr<diagonal_float,
// diagonal_float,
//    ten::functional::binary_func<ten::binary_operation::mul>::template func>;
using div_diagonal_float = ten::binary_expr<
    diagonal_float, diagonal_float, diagonal_float,
    ten::functional::binary_func<ten::binary_operation::div>::template func>;

using add_diagonal_double = ten::binary_expr<
    diagonal_double, diagonal_double, diagonal_double,
    ten::functional::binary_func<ten::binary_operation::add>::template func>;
using sub_diagonal_double = ten::binary_expr<
    diagonal_double, diagonal_double, diagonal_double,
    ten::functional::binary_func<ten::binary_operation::sub>::template func>;
// using mul_diagonal_double = ten::binary_expr<diagonal_double,
// diagonal_double,
//    ten::functional::binary_func<ten::binary_operation::mul>::template func>;
using div_diagonal_double = ten::binary_expr<
    diagonal_double, diagonal_double, diagonal_double,
    ten::functional::binary_func<ten::binary_operation::div>::template func>;
*/

// Binary func mul
/*
using mul_vector_float_vector_float =
    ten::binary_expr<vector_float, vector_float, vector_float,
                     ten::functional::mul>;
using mul_vector_double_vector_double =
    ten::binary_expr<vector_double, vector_double, vector_double,
                     ten::functional::mul>;

using mul_vector_int32_vector_int32 =
    ten::binary_expr<vector_int32, vector_int32, vector_int32,
                     ten::functional::mul>;
using mul_vector_int64_vector_int64 =
    ten::binary_expr<vector_int64, vector_int64, vector_int64,
                     ten::functional::mul>;

using mul_vector_uint32_vector_uint32 =
    ten::binary_expr<vector_uint32, vector_uint32, vector_uint32,
                     ten::functional::mul>;
using mul_vector_uint64_vector_uint64 =
    ten::binary_expr<vector_uint64, vector_uint64, vector_uint64,
                     ten::functional::mul>;

using mul_matrix_float_matrix_float =
    ten::binary_expr<matrix_float, matrix_float, matrix_float,
                     ten::functional::mul>;
using mul_matrix_double_matrix_double =
    ten::binary_expr<matrix_double, matrix_double, matrix_double,
                     ten::functional::mul>;

using mul_matrix_float_vector_float =
    ten::binary_expr<matrix_float, vector_float, vector_float,
                     ten::functional::mul>;
using mul_matrix_double_vector_double =
    ten::binary_expr<matrix_double, vector_double, vector_double,
                     ten::functional::mul>;

using mul_scalar_float_vector_float =
    ten::binary_expr<scalar_float, vector_float, vector_float,
                     ten::functional::scalar_left_binary_func<::ten::binary_operation::mul>::func>;
using mul_scalar_double_vector_double =
    ten::binary_expr<scalar_double, vector_double, vector_double,
                     ten::functional::scalar_left_binary_func<::ten::binary_operation::mul>::func>;

using mul_scalar_int32_vector_int32 =
    ten::binary_expr<scalar_int32, vector_int32, vector_int32,
                     ten::functional::scalar_left_binary_func<::ten::binary_operation::mul>::func>;
using mul_scalar_int64_vector_int64 =
    ten::binary_expr<scalar_int64, vector_int64, vector_int64,
                     ten::functional::scalar_left_binary_func<::ten::binary_operation::mul>::func>;

using mul_scalar_uint32_vector_uint32 =
    ten::binary_expr<scalar_uint32, vector_uint32, vector_uint32,
                     ten::functional::scalar_left_binary_func<::ten::binary_operation::mul>::func>;
using mul_scalar_uint64_vector_uint64 =
    ten::binary_expr<scalar_uint64, vector_uint64, vector_uint64,
                     ten::functional::scalar_left_binary_func<::ten::binary_operation::mul>::func>;

using mul_scalar_float_matrix_float =
    ten::binary_expr<scalar_float, matrix_float, matrix_float,
                     ten::functional::scalar_left_binary_func<::ten::binary_operation::mul>::func>;
using mul_scalar_double_matrix_double =
    ten::binary_expr<scalar_double, matrix_double, matrix_double,
                     ten::functional::scalar_left_binary_func<::ten::binary_operation::mul>::func>;

using mul_scalar_float_tensor3_float =
    ten::binary_expr<scalar_float, tensor3_float, tensor3_float,
                     ten::functional::scalar_left_binary_func<::ten::binary_operation::mul>::func>;
using mul_scalar_double_tensor3_double =
    ten::binary_expr<scalar_double, tensor3_double, tensor3_double,
                     ten::functional::scalar_left_binary_func<::ten::binary_operation::mul>::func>;

using mul_scalar_float_tensor4_float =
    ten::binary_expr<scalar_float, tensor4_float, tensor4_float,
                     ten::functional::scalar_left_binary_func<::ten::binary_operation::mul>::func>;
using mul_scalar_double_tensor4_double =
    ten::binary_expr<scalar_double, tensor4_double, tensor4_double,
                     ten::functional::scalar_left_binary_func<::ten::binary_operation::mul>::func>;

using mul_scalar_float_tensor5_float =
    ten::binary_expr<scalar_float, tensor5_float, tensor5_float,
                     ten::functional::scalar_left_binary_func<::ten::binary_operation::mul>::func>;
using mul_scalar_double_tensor5_double =
    ten::binary_expr<scalar_double, tensor5_double, tensor5_double,
                     ten::functional::scalar_left_binary_func<::ten::binary_operation::mul>::func>;

using mul_diagonal_float_diagonal_float =
    ten::binary_expr<diagonal_float, diagonal_float, matrix_float,
                     ten::functional::mul>;
using mul_diagonal_double_diagonal_double =
    ten::binary_expr<diagonal_double, diagonal_double, matrix_double,
                     ten::functional::mul>;
*/

// Others binary functions

////////////////////////////////////////////////////////////////////////////////
// Initialization

// fill<T>(shape, value, order)
template <class T>
[[nodiscard]] auto py_fill(const std::vector<std::size_t> &dims, T value,
                           const ten::storage_order order) {
  return ten::fill<ten::tensor<T>>(dims, value, false, order);
}

// zeros<T>(shape, order)
template <class T>
[[nodiscard]] auto py_zeros(const std::vector<std::size_t> &dims,
                            const ten::storage_order order) {
  return ten::zeros<ten::tensor<T>>(dims, false, order);
}

// ones<T>(shape, order)
template <class T>
[[nodiscard]] auto py_ones(const std::vector<std::size_t> &dims,
                           const ten::storage_order order) {
  return ten::ones<ten::tensor<T>>(dims, false, order);
}

// range<T>(shape, value)
template <class T>
[[nodiscard]] auto py_range(const std::vector<std::size_t> &dims, T value,
                            const ten::storage_order order) {
  return ten::range<ten::tensor<T>>(dims, value, false, order);
}

// linear<T>(shape, start, stop, order)
template <class T>
[[nodiscard]] auto py_linear(const std::vector<std::size_t> &dims, T start,
                             T stop, const ten::storage_order order) {
  return ten::linear<ten::tensor<T>>(dims, start, stop, false, order);
}

/*
// reshape
template <class T, class Shape>
   requires(::ten::is_tensor_v<T>)
auto py_reshape(T x, Shape &dims) {
   using expr_type = typename std::remove_cvref_t<T>;
   using value_type = expr_type::value_type;
   using output_type =
       ten::ranked_tensor<value_type, Shape, expr_type::storage_order(),
                          typename expr_type::storage_type,
                          typename expr_type::allocator_type>;
   // using output_type = typename ten::details::output_type<expr_type>::type;
   using shape_type = std::remove_cvref_t<Shape>;
   using reshape_type =
       typename ::ten::details::reshape_result<output_type, shape_type>::type;
   return ::ten::unary_expr<
       expr_type, reshape_type,
       ::ten::functional::dynamic_reshape<shape_type>::template func>(x, dims);
}*/

////////////////////////////////////////////////////////////////////////////////
// Distributions
using uniform_float = ten::uniform<float>;
using uniform_double = ten::uniform<double>;
using normal_float = ten::normal<float>;
using normal_double = ten::normal<double>;
using gamma_float = ten::gamma<float>;
using gamma_double = ten::gamma<double>;

template <typename T> auto py_make_uniform(T lower_bound, T upper_bound) {
  return ten::uniform<float>(lower_bound, upper_bound);
}

template <typename T> auto py_make_normal(T mean, T std) {
  return ten::normal<T>(mean, std);
}

template <typename T> auto py_make_gamma(T alpha, T beta) {
  return ten::gamma<T>(alpha, beta);
}

////////////////////////////////////////////////////////////////////////////////
// Random

////////////////////////////////////////////////////////////////////////////////
// learning
// using histogram_float = ten::ml::histogram<float>;
// using histogram_double = ten::ml::histogram<double>;
// using histogram_options = ten::ml::histogram_options;

// Create a new tensor
template <typename T>
ten::tensor<T>
py_make_tensor(const std::vector<std::size_t> &dims,
               const ten::storage_format format,
               const bool requires_grad = false,
               const ten::storage_order order = ten::storage_order::col_major) {
  return ten::tensor<T>(dims, format, requires_grad, order);
}

// Get item from vector
template <typename T>
T &py_get_value(ten::tensor<T> &t, const std::vector<std::size_t> &indices) {
  return t[indices];
}

// Set item from vector
template <typename T>
void py_set_value(ten::tensor<T> &t, const std::vector<std::size_t> &indices,
                  T value) {
  t[indices] = value;
}

// Random
template <typename T>
auto py_rand_norm(
    const std::vector<std::size_t> &dims, T mean, T std,
    const ten::storage_order order = ten::storage_order::col_major) {
  return ten::rand_norm<T>(dims, mean, std, false, order);
}

template <typename T>
auto py_rand_unif(
    const std::vector<std::size_t> &dims, T lower_bound, T upper_bound,
    const ten::storage_order order = ten::storage_order::col_major) {
  return ten::rand_unif<T>(dims, lower_bound, upper_bound, false, order);
}

////////////////////////////////////////////////////////////////////////////////
// Bindings for the core libray

PYBIND11_MODULE(tencore, m) {
  m.doc() = "Ten core: Bindings for the ten library";

  // Scalars
  py::class_<scalar_float>(m, "scalar_float").def(py::init<const float &>())
      //.def("value", &scalar_float::value)
      ; /*.def("__repr__", [](const scalar_float &s) {
         std::stringstream ss;
         ss << s;
         return ss.str();
       });*/

  py::class_<scalar_double>(m, "scalar_double").def(py::init<const double &>())
      //.def("value", &scalar_double::value)
      ; /*.def("__repr__", [](const scalar_double &s) {
         std::stringstream ss;
         ss << s;
         return ss.str();
       });*/
        /*
       py::class_<scalar_int32>(m, "scalar_int32")
           .def(py::init<const int32_t &>())
           //.def("value", &scalar_int32::value)
           .def("__repr__", [](const scalar_int32 &s) {
             std::stringstream ss;
             ss << s;
             return ss.str();
           });
      
       py::class_<scalar_int64>(m, "scalar_int64")
           .def(py::init<const int64_t &>())
           //.def("value", &scalar_int64::value)
           .def("__repr__", [](const scalar_int64 &s) {
             std::stringstream ss;
             ss << s;
             return ss.str();
           });*/

  // Data type
  py::enum_<ten::data_type>(m, "data_type", py::arithmetic())
      .value("float32", ten::data_type::float32)
      .value("float64", ten::data_type::float64)
      .value("boolean", ten::data_type::boolean)
      .value("int32", ten::data_type::int32)
      .value("int64", ten::data_type::int64)
      .value("uint32", ten::data_type::uint32)
      .value("uint64", ten::data_type::uint64)
      .value("complexfloat32", ten::data_type::complexfloat32)
      .value("complexfloat64", ten::data_type::complexfloat64)
      .value("string", ten::data_type::string);

  // Storage format
  py::enum_<ten::storage_format>(m, "storage_format", py::arithmetic())
      .value("dense", ten::storage_format::dense)
      .value("coo", ten::storage_format::coo)
      .value("csc", ten::storage_format::csc)
      .value("csr", ten::storage_format::csr)
      .value("diagonal", ten::storage_format::diagonal)
      .value("lower_tr", ten::storage_format::lower_tr)
      .value("upper_tr", ten::storage_format::upper_tr)
      .value("symmetric", ten::storage_format::symmetric)
      .value("transposed", ten::storage_format::transposed)
      .value("hermitian", ten::storage_format::hermitian);

  // Storage order
  py::enum_<ten::storage_order>(m, "storage_order", py::arithmetic())
      .value("col_major", ten::storage_order::col_major)
      .value("row_major", ten::storage_order::row_major);

  // Unary expr

  // ten::min
  /*
  py::class_<expr_min_vector_float>(m, "expr_min_vector_float")
      .def("value", &expr_min_vector_float::value)
      .def("eval", &expr_min_vector_float::eval);
  py::class_<expr_min_vector_double>(m, "expr_min_vector_double")
      .def("value", &expr_min_vector_double::value)
      .def("eval", &expr_min_vector_double::eval);
  py::class_<expr_min_matrix_float>(m, "expr_min_matrix_float")
      .def("value", &expr_min_matrix_float::value)
      .def("eval", &expr_min_matrix_float::eval);
  py::class_<expr_min_matrix_double>(m, "expr_min_matrix_double")
      .def("value", &expr_min_matrix_double::value)
      .def("eval", &expr_min_matrix_double::eval);
  py::class_<expr_min_tensor3_float>(m, "expr_min_tensor3_float")
      .def("value", &expr_min_tensor3_float::value)
      .def("eval", &expr_min_tensor3_float::eval);
  py::class_<expr_min_tensor3_double>(m, "expr_min_tensor3_double")
      .def("value", &expr_min_tensor3_double::value)
      .def("eval", &expr_min_tensor3_double::eval);
  py::class_<expr_min_tensor4_float>(m, "expr_min_tensor4_float")
      .def("value", &expr_min_tensor4_float::value)
      .def("eval", &expr_min_tensor4_float::eval);
  py::class_<expr_min_tensor4_double>(m, "expr_min_tensor4_double")
      .def("value", &expr_min_tensor4_double::value)
      .def("eval", &expr_min_tensor4_double::eval);
  py::class_<expr_min_tensor5_float>(m, "expr_min_tensor5_float")
      .def("value", &expr_min_tensor5_float::value)
      .def("eval", &expr_min_tensor5_float::eval);
  py::class_<expr_min_tensor5_double>(m, "expr_min_tensor5_double")
      .def("value", &expr_min_tensor5_double::value)
      .def("eval", &expr_min_tensor5_double::eval);
  */

  // ten::max
  /*
  py::class_<expr_max_vector_float>(m, "expr_max_vector_float")
      .def("value", &expr_max_vector_float::value)
      .def("eval", &expr_max_vector_float::eval);
  py::class_<expr_max_vector_double>(m, "expr_max_vector_double")
      .def("value", &expr_max_vector_double::value)
      .def("eval", &expr_max_vector_double::eval);
  py::class_<expr_max_matrix_float>(m, "expr_max_matrix_float")
      .def("value", &expr_max_matrix_float::value)
      .def("eval", &expr_max_matrix_float::eval);
  py::class_<expr_max_matrix_double>(m, "expr_max_matrix_double")
      .def("value", &expr_max_matrix_double::value)
      .def("eval", &expr_max_matrix_double::eval);
  py::class_<expr_max_tensor3_float>(m, "expr_max_tensor3_float")
      .def("value", &expr_max_tensor3_float::value)
      .def("eval", &expr_max_tensor3_float::eval);
  py::class_<expr_max_tensor3_double>(m, "expr_max_tensor3_double")
      .def("value", &expr_max_tensor3_double::value)
      .def("eval", &expr_max_tensor3_double::eval);
  py::class_<expr_max_tensor4_float>(m, "expr_max_tensor4_float")
      .def("value", &expr_max_tensor4_float::value)
      .def("eval", &expr_max_tensor4_float::eval);
  py::class_<expr_max_tensor4_double>(m, "expr_max_tensor4_double")
      .def("value", &expr_max_tensor4_double::value)
      .def("eval", &expr_max_tensor4_double::eval);
  py::class_<expr_max_tensor5_float>(m, "expr_max_tensor5_float")
      .def("value", &expr_max_tensor5_float::value)
      .def("eval", &expr_max_tensor5_float::eval);
  py::class_<expr_max_tensor5_double>(m, "expr_max_tensor5_double")
      .def("value", &expr_max_tensor5_double::value)
      .def("eval", &expr_max_tensor5_double::eval);
  */

  // sqrt
  py::class_<expr_sqrt_tensor_float>(m, "expr_sqrt_tensor_float")
      .def("value", &expr_sqrt_tensor_float::value)
      .def("eval", &expr_sqrt_tensor_float::eval);
  py::class_<expr_sqrt_tensor_double>(m, "expr_sqrt_tensor_double")
      .def("value", &expr_sqrt_tensor_double::value)
      .def("eval", &expr_sqrt_tensor_double::eval);

  // sqr
  /*
 py::class_<expr_sqr_tensor_float>(m, "expr_sqr_tensor_float")
     .def("value", &expr_sqr_tensor_float::value)
     .def("eval", &expr_sqr_tensor_float::eval);
 py::class_<expr_sqr_tensor_double>(m, "expr_sqr_tensor_double")
     .def("value", &expr_sqr_tensor_double::value)
     .def("eval", &expr_sqr_tensor_double::eval);
  */

  // Reshape
  /*
  // Reshape a vector to a tensor
  py::class_<expr_reshape_vector_float_matrix_float>(
      m, "expr_reshape_vector_float_matrix_float")
      .def("value", &expr_reshape_vector_float_matrix_float::value)
      .def("eval", &expr_reshape_vector_float_matrix_float::eval);
  */

  // Binar expr

  // add, sub, mul, div
  /*
  py::class_<add_vector_float>(m, "add_vector_float")
      .def("value", &add_vector_float::value)
      .def("eval", &add_vector_float::eval);
  py::class_<sub_vector_float>(m, "sub_vector_float")
      .def("value", &sub_vector_float::value)
      .def("eval", &sub_vector_float::eval);
  py::class_<mul_vector_float>(m, "mul_vector_float")
      .def("value", &mul_vector_float::value)
      .def("eval", &mul_vector_float::eval);
  py::class_<div_vector_float>(m, "div_vector_float")
      .def("value", &div_vector_float::value)
      .def("eval", &div_vector_float::eval);
  */

  // Binary expr mul
  /*
  py::class_<mul_vector_float_vector_float>(m, "mul_vector_float_vector_float")
      .def("value", &mul_vector_float_vector_float::value)
      .def("eval", &mul_vector_float_vector_float::eval);
  py::class_<mul_vector_double_vector_double>(
      m, "mul_vector_double_vector_double")
      .def("value", &mul_vector_double_vector_double::value)
      .def("eval", &mul_vector_double_vector_double::eval);

  py::class_<mul_vector_int32_vector_int32>(m, "mul_vector_int32_vector_int32")
      .def("value", &mul_vector_int32_vector_int32::value)
      .def("eval", &mul_vector_int32_vector_int32::eval);
  py::class_<mul_vector_int64_vector_int64>(m, "mul_vector_int64_vector_int64")
      .def("value", &mul_vector_int64_vector_int64::value)
      .def("eval", &mul_vector_int64_vector_int64::eval);
  */

  // Tensor float
  py::class_<tensor_float>(m, "tensor_float")
      .def("make", &py_make_tensor<float>)
      .def("rank", &tensor_float::rank)
      .def("size", &tensor_float::size)
      .def("shape", &tensor_float::shape)
      .def("dim", &tensor_float::dim)
      .def("strides", &tensor_float::strides)
      .def("data_type", &tensor_float::data_type)
      .def("format", &tensor_float::format)
      .def("storage_order", &tensor_float::storage_order)
      .def("__getitem__",
           [](const tensor_float &t, size_t index) { return t[index]; })
      .def("__setitem__",
           [](tensor_float &t, size_t index, float value) { t[index] = value; })
      .def("is_transposed", &tensor_float::is_transposed)
      .def("is_symmetric", &tensor_float::is_symmetric)
      .def("is_hermitian", &tensor_float::is_hermitian)
      .def("is_diagonal", &tensor_float::is_diagonal)
      .def("is_lower_tr", &tensor_float::is_lower_tr)
      .def("is_upper_tr", &tensor_float::is_upper_tr)
      .def("__repr__", [](const tensor_float &t) {
        std::stringstream ss;
        ss << t;
        return ss.str();
      });

  // Get and set values from vector
  m.def("tensor_float_get", &py_get_value<float>);
  m.def("tensor_float_set", &py_set_value<float>);

  // Tensor double
  py::class_<tensor_double>(m, "tensor_double")
      .def("make", &py_make_tensor<double>)
      .def("rank", &tensor_double::rank)
      .def("size", &tensor_double::size)
      .def("shape", &tensor_double::shape)
      .def("dim", &tensor_double::dim)
      .def("strides", &tensor_double::strides)
      .def("data_type", &tensor_double::data_type)
      .def("format", &tensor_double::format)
      .def("storage_order", &tensor_double::storage_order)
      .def("__getitem__",
           [](const tensor_double &t, size_t index) { return t[index]; })
      .def("__setitem__", [](tensor_double &t, size_t index,
                             double value) { t[index] = value; })
      .def("is_transposed", &tensor_double::is_transposed)
      .def("is_symmetric", &tensor_double::is_symmetric)
      .def("is_hermitian", &tensor_double::is_hermitian)
      .def("is_diagonal", &tensor_double::is_diagonal)
      .def("is_lower_tr", &tensor_double::is_lower_tr)
      .def("is_upper_tr", &tensor_double::is_upper_tr)
      .def("__repr__", [](const tensor_double &t) {
        std::stringstream ss;
        ss << t;
        return ss.str();
      });

  // Get and set values from vector
  m.def("tensor_double_get", &py_get_value<double>);
  m.def("tensor_double_set", &py_set_value<double>);

  // Initialization functions
  m.def("zeros_float", &py_zeros<float>);
  m.def("zeros_double", &py_zeros<double>);

  m.def("ones_float", &py_ones<float>);
  m.def("ones_double", &py_ones<double>);

  m.def("fill_float", &py_fill<float>);
  m.def("fill_double", &py_fill<double>);

  m.def("range_float", &py_range<float>);
  m.def("range_double", &py_range<double>);

  m.def("linear_float", &py_linear<float>);
  m.def("linear_double", &py_linear<double>);

  /*
  // Transform diagonal to dense
  m.def("dense_float", &ten::dense<diagonal_float>);
  m.def("dense_double", &ten::dense<diagonal_double>);

  // Transposed, symmetric, lower_tr and upper_tr
  m.def("transposed_float", &ten::transposed<matrix_float>);
  m.def("transposed_double", &ten::transposed<matrix_double>);
  m.def("symmetric_float", &ten::symmetric<matrix_float>);
  m.def("symmetric_double", &ten::symmetric<matrix_double>);
  m.def("lower_tr_float", &ten::lower_tr<matrix_float>);
  m.def("lower_tr_double", &ten::lower_tr<matrix_double>);
  m.def("upper_tr_float", &ten::upper_tr<matrix_float>);
  m.def("upper_tr_double", &ten::upper_tr<matrix_double>);

  // Cast
  m.def("cast_vector_float_double", &ten::cast<double, vector_float>);
  m.def("cast_vector_double_float", &ten::cast<float, vector_double>);
  m.def("cast_matrix_float_double", &ten::cast<double, matrix_float>);
  m.def("cast_matrix_double_float", &ten::cast<float, matrix_double>);

  // Reshape
  // vector to matrix
  m.def("reshape_vector_float_matrix_float",
        &py_reshape<vector_float, ten::shape<0, 0>>);
  m.def("reshape_vector_double_matrix_double",
        &py_reshape<vector_double, ten::shape<0, 0>>);
  // vector to tensor3
  m.def("reshape_vector_float_tensor3_float",
        &py_reshape<vector_float, ten::shape<0, 0, 0>>);
  m.def("reshape_vector_double_tensor3_double",
        &py_reshape<vector_double, ten::shape<0, 0, 0>>);
  // vector to tensor4
  m.def("reshape_vector_float_tensor4_float",
        &py_reshape<vector_float, ten::shape<0, 0, 0, 0>>);
  m.def("reshape_vector_double_tensor4_double",
        &py_reshape<vector_double, ten::shape<0, 0, 0, 0>>);
  // vector to tensor5
  m.def("reshape_vector_float_tensor5_float",
        &py_reshape<vector_float, ten::shape<0, 0, 0, 0, 0>>);
  m.def("reshape_vector_double_tensor5_double",
        &py_reshape<vector_double, ten::shape<0, 0, 0, 0, 0>>);
  // matrix to vector
  m.def("reshape_matrix_float_vector_float",
        &py_reshape<matrix_float, ten::shape<0>>);
  m.def("reshape_matrix_double_vector_double",
        &py_reshape<matrix_double, ten::shape<0>>);
  // matrix to tensor3
  m.def("reshape_matrix_float_tensor3_float",
        &py_reshape<vector_float, ten::shape<0, 0, 0>>);
  m.def("reshape_matrix_double_tensor3_double",
        &py_reshape<vector_float, ten::shape<0, 0, 0>>);
  // matrix to tensor4
  m.def("reshape_matrix_float_tensor4_float",
        &py_reshape<matrix_float, ten::shape<0, 0, 0, 0>>);
  m.def("reshape_matrix_double_tensor4_double",
        &py_reshape<matrix_double, ten::shape<0, 0, 0, 0>>);
  // matrix to tensor5
  m.def("reshape_matrix_float_tensor5_float",
        &py_reshape<matrix_float, ten::shape<0, 0, 0, 0, 0>>);
  m.def("reshape_matrix_double_tensor5_double",
        &py_reshape<matrix_double, ten::shape<0, 0, 0, 0, 0>>);
  // tensor3 to vector
  m.def("reshape_tensor3_float_vector_float",
        &py_reshape<tensor3_float, ten::shape<0>>);
  m.def("reshape_tensor3_double_vector_double",
        &py_reshape<tensor3_double, ten::shape<0>>);
  // tensor3 to matrix
  m.def("reshape_tensor3_float_matrix_float",
        &py_reshape<tensor3_float, ten::shape<0, 0>>);
  m.def("reshape_tensor3_double_matrix_double",
        &py_reshape<tensor3_double, ten::shape<0, 0>>);
  // tensor3 to tensor4
  m.def("reshape_tensor3_float_tensor4_float",
        &py_reshape<tensor3_float, ten::shape<0, 0, 0, 0>>);
  m.def("reshape_tensor3_double_tensor4_double",
        &py_reshape<tensor3_double, ten::shape<0, 0, 0, 0>>);
  // tensor3 to tensor5
  m.def("reshape_tensor3_float_tensor5_float",
        &py_reshape<tensor3_float, ten::shape<0, 0, 0, 0, 0>>);
  m.def("reshape_tensor3_double_tensor5_double",
        &py_reshape<tensor3_double, ten::shape<0, 0, 0, 0, 0>>);
  // tensor4 to vector
  m.def("reshape_tensor4_float_vector_float",
        &py_reshape<tensor4_float, ten::shape<0>>);
  m.def("reshape_tensor4_double_vector_double",
        &py_reshape<tensor4_double, ten::shape<0>>);
  // tensor4 to matrix
  m.def("reshape_tensor4_float_matrix_float",
        &py_reshape<tensor4_float, ten::shape<0, 0>>);
  m.def("reshape_tensor4_double_matrix_double",
        &py_reshape<tensor4_double, ten::shape<0, 0>>);
  // tensor4 to tensor3
  m.def("reshape_tensor4_float_tensor3_float",
        &py_reshape<tensor4_float, ten::shape<0, 0, 0>>);
  m.def("reshape_tensor4_double_tensor3_double",
        &py_reshape<tensor4_double, ten::shape<0, 0, 0>>);
  // tensor4 to tensor5
  m.def("reshape_tensor4_float_tensor5_float",
        &py_reshape<tensor4_float, ten::shape<0, 0, 0, 0, 0>>);
  m.def("reshape_tensor4_double_tensor5_double",
        &py_reshape<tensor4_double, ten::shape<0, 0, 0, 0, 0>>);
  // tensor5 to vector
  m.def("reshape_tensor5_float_vector_float",
        &py_reshape<tensor5_float, ten::shape<0>>);
  m.def("reshape_tensor5_double_vector_double",
        &py_reshape<tensor5_double, ten::shape<0>>);
  // tensor5 to matrix
  m.def("reshape_tensor5_float_matrix_float",
        &py_reshape<tensor5_float, ten::shape<0, 0>>);
  m.def("reshape_tensor5_double_matrix_double",
        &py_reshape<tensor5_double, ten::shape<0, 0>>);
  // tensor5 to tensor3
  m.def("reshape_tensor5_float_tensor3_float",
        &py_reshape<tensor5_float, ten::shape<0, 0, 0>>);
  m.def("reshape_tensor5_double_tensor3_double",
        &py_reshape<tensor5_double, ten::shape<0, 0, 0>>);
  // tensor5 to tensor4
  m.def("reshape_tensor5_float_tensor4_float",
        &py_reshape<tensor5_float, ten::shape<0, 0, 0, 0>>);
  m.def("reshape_tensor5_double_tensor4_double",
        &py_reshape<tensor5_double, ten::shape<0, 0, 0, 0>>);

  // Flatten
  m.def("flatten_matrix_float", &ten::flatten<matrix_float>);
  m.def("flatten_matrix_double", &ten::flatten<matrix_double>);
  m.def("flatten_tensor3_float", &ten::flatten<tensor3_float>);
  m.def("flatten_tensor3_double", &ten::flatten<tensor3_double>);
  m.def("flatten_tensor4_float", &ten::flatten<tensor4_float>);
  m.def("flatten_tensor4_double", &ten::flatten<tensor4_double>);
  m.def("flatten_tensor5_float", &ten::flatten<tensor5_float>);
  m.def("flatten_tensor5_double", &ten::flatten<tensor5_double>);

  // Initializations
  // FIXME Valeur par défaut pour range
  m.def("fill_vector_float", &fill_py<vector_float>);
  m.def("zeros_vector_float", &zeros_py<vector_float>);
  m.def("ones_vector_float", &ones_py<vector_float>);
  m.def("range_vector_float", &range_py<vector_float>);
  m.def("fill_vector_double", &fill_py<vector_double>);
  m.def("zeros_vector_double", &zeros_py<vector_double>);
  m.def("ones_vector_double", &ones_py<vector_double>);
  m.def("range_vector_double", &range_py<vector_double>);

  m.def("fill_matrix_float", &fill_py<matrix_float>);
  m.def("zeros_matrix_float", &zeros_py<matrix_float>);
  m.def("ones_matrix_float", &ones_py<matrix_float>);
  m.def("range_matrix_float", &range_py<matrix_float>);
  m.def("fill_matrix_double", &fill_py<matrix_double>);
  m.def("zeros_matrix_double", &zeros_py<matrix_double>);
  m.def("ones_matrix_double", &ones_py<matrix_double>);
  m.def("range_matrix_double", &range_py<matrix_double>);

  m.def("fill_tensor3_float", &fill_py<tensor3_float>);
  m.def("zeros_tensor3_float", &zeros_py<tensor3_float>);
  m.def("ones_tensor3_float", &ones_py<tensor3_float>);
  m.def("range_tensor3_float", &range_py<tensor3_float>);
  m.def("fill_tensor3_double", &fill_py<tensor3_double>);
  m.def("zeros_tensor3_double", &zeros_py<tensor3_double>);
  m.def("ones_tensor3_double", &ones_py<tensor3_double>);
  m.def("range_tensor3_double", &range_py<tensor3_double>);

  m.def("fill_tensor4_float", &fill_py<tensor4_float>);
  m.def("zeros_tensor4_float", &zeros_py<tensor4_float>);
  m.def("ones_tensor4_float", &ones_py<tensor4_float>);
  m.def("range_tensor4_float", &range_py<tensor4_float>);
  m.def("fill_tensor4_double", &fill_py<tensor4_double>);
  m.def("zeros_tensor4_double", &zeros_py<tensor4_double>);
  m.def("ones_tensor4_double", &ones_py<tensor4_double>);
  m.def("range_tensor4_double", &range_py<tensor4_double>);

  m.def("fill_tensor5_float", &fill_py<tensor5_float>);
  m.def("zeros_tensor5_float", &zeros_py<tensor5_float>);
  m.def("ones_tensor5_float", &ones_py<tensor5_float>);
  m.def("range_tensor5_float", &range_py<tensor5_float>);
  m.def("fill_tensor5_double", &fill_py<tensor5_double>);
  m.def("zeros_tensor5_double", &zeros_py<tensor5_double>);
  m.def("ones_tensor5_double", &ones_py<tensor5_double>);
  m.def("range_tensor5_double", &range_py<tensor5_double>);
  */

  // Save to a binary file
  /*m.def("save_vector_float", &ten::save<vector_float>);
  m.def("save_vector_double", &ten::save<vector_double>);
  m.def("save_matrix_float", &ten::save<matrix_float>);
  m.def("save_matrix_double", &ten::save<matrix_double>);
  m.def("save_tensor3_float", &ten::save<tensor3_float>);
  m.def("save_tensor3_double", &ten::save<tensor3_double>);
  m.def("save_tensor4_float", &ten::save<tensor4_float>);
  m.def("save_tensor4_double", &ten::save<tensor4_double>);
  m.def("save_tensor5_float", &ten::save<tensor5_float>);
  m.def("save_tensor5_double", &ten::save<tensor5_double>);
  */

  // Load from binary file
  /*
  m.def("load_vector_float", &ten::load<vector_float>);
  m.def("load_vector_double", &ten::load<vector_double>);
  m.def("load_matrix_float", &ten::load<matrix_float>);
  m.def("load_matrix_double", &ten::load<matrix_double>);
  m.def("load_tensor3_float", &ten::load<tensor3_float>);
  m.def("load_tensor3_double", &ten::load<tensor3_double>);
  m.def("load_tensor4_float", &ten::load<tensor4_float>);
  m.def("load_tensor4_double", &ten::load<tensor4_double>);
  m.def("load_tensor5_float", &ten::load<tensor5_float>);
  m.def("load_tensor5_double", &ten::load<tensor5_double>);
  */

  // Save to a mtx file
  /*
  m.def("save_mtx_vector_float", &ten::io::save_mtx<vector_float>);
  m.def("save_mtx_vector_double", &ten::io::save_mtx<vector_double>);
  m.def("save_mtx_matrix_float", &ten::io::save_mtx<matrix_float>);
  m.def("save_mtx_matrix_double", &ten::io::save_mtx<matrix_double>);
  */

  /////////////////////////////////////////////////////////////////////////////
  // Functions
  /*
  // min
  m.def("min_vector_float", &min_py<vector_float>);
  m.def("min_vector_double", &min_py<vector_double>);
  m.def("min_matrix_float", &min_py<matrix_float>);
  m.def("min_matrix_double", &min_py<matrix_double>);
  m.def("min_tensor3_float", &min_py<tensor3_float>);
  m.def("min_tensor3_double", &min_py<tensor3_double>);
  m.def("min_tensor4_float", &min_py<tensor4_float>);
  m.def("min_tensor4_double", &min_py<tensor4_double>);
  m.def("min_tensor5_float", &min_py<tensor5_float>);
  m.def("min_tensor5_double", &min_py<tensor5_double>);
  // max
  m.def("max_vector_float", &max_py<vector_float>);
  m.def("max_vector_double", &max_py<vector_double>);
  m.def("max_matrix_float", &max_py<matrix_float>);
  m.def("max_matrix_double", &max_py<matrix_double>);
  m.def("max_tensor3_float", &max_py<tensor3_float>);
  m.def("max_tensor3_double", &max_py<tensor3_double>);
  m.def("max_tensor4_float", &max_py<tensor4_float>);
  m.def("max_tensor4_double", &max_py<tensor4_double>);
  m.def("max_tensor5_float", &max_py<tensor5_float>);
  m.def("max_tensor5_double", &max_py<tensor5_double>);
  */

  /////////////////////////////////////////////////////////////////////////////
  // Distributions
  m.def("set_seed", &ten::set_seed);

  /*
  py::class_<ten::uniform<float>>(m, "uniform_float")
     .def(py::init<float, float>)
     .def("sample", &ten::uniform<float>::sample);
  */

  /////////////////////////////////////////////////////////////////////////////
  // Distributions
  py::class_<uniform_float>(m, "uniform_float")
      .def("make", &py_make_uniform<float>)
      .def("sample", [](uniform_float &dist) { return dist.sample(); })
      .def("sample_tensor",
           [](uniform_float &dist, const std::vector<std::size_t> &dims,
              ten::storage_order order = ten::storage_order::col_major) {
             return dist.sample(dims, false, order);
           });

  py::class_<uniform_double>(m, "uniform_double")
      .def("make", &py_make_uniform<double>)
      .def("sample", [](uniform_double &dist) { return dist.sample(); })
      .def("sample_tensor",
           [](uniform_double &dist, const std::vector<std::size_t> &dims,
              ten::storage_order order = ten::storage_order::col_major) {
             return dist.sample(dims, false, order);
           });

  py::class_<normal_float>(m, "normal_float")
      .def("make", &py_make_normal<float>)
      .def("sample", [](normal_float &dist) { return dist.sample(); })
      .def("sample_tensor",
           [](normal_float &dist, const std::vector<std::size_t> &dims,
              ten::storage_order order = ten::storage_order::col_major) {
             return dist.sample(dims, false, order);
           });

  py::class_<normal_double>(m, "normal_double")
      .def("make", &py_make_normal<double>)
      .def("sample", [](normal_double &dist) { return dist.sample(); })
      .def("sample_tensor",
           [](normal_double &dist, const std::vector<std::size_t> &dims,
              ten::storage_order order = ten::storage_order::col_major) {
             return dist.sample(dims, false, order);
           });

  py::class_<gamma_float>(m, "gamma_float")
      .def("make", &py_make_gamma<float>)
      .def("sample", [](gamma_float &dist) { return dist.sample(); })
      .def("sample_tensor",
           [](gamma_float &dist, const std::vector<std::size_t> &dims,
              ten::storage_order order = ten::storage_order::col_major) {
             return dist.sample(dims, false, order);
           });

  py::class_<gamma_double>(m, "gamma_double")
      .def("make", &py_make_gamma<double>)
      .def("sample", [](gamma_double &dist) { return dist.sample(); })
      .def("sample_tensor",
           [](gamma_double &dist, const std::vector<std::size_t> &dims,
              ten::storage_order order = ten::storage_order::col_major) {
             return dist.sample(dims, false, order);
           });

  /////////////////////////////////////////////////////////////////////////////
  /// Random numbers generation

  m.def("rand_norm_float", &py_rand_norm<float>);
  m.def("rand_norm_double", &py_rand_norm<double>);

  m.def("rand_unif_float", &py_rand_unif<float>);
  m.def("rand_unif_double", &py_rand_unif<double>);

  /////////////////////////////////////////////////////////////////////////////
  // learning
  /*
  py::class_<histogram_options>(m, "histogram_options")
      .def(py::init<bool, bool, size_t>());
  py::class_<histogram_float>(m, "histogram_float")
      .def(py::init<histogram_options>())
      .def("fit", &histogram_float::fit)
      .def("hist", &histogram_float::hist);
  py::class_<histogram_double>(m, "histogram_double")
      .def(py::init<histogram_options>())
      .def("fit", &histogram_double::fit)
      .def("hist", &histogram_double::hist);
  */
}
