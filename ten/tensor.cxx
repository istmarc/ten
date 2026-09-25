#include <cmath>

#include <functional>

#include <ten/cmb>
#include <ten/graph>
#include <ten/io>
#include <ten/linalg>
#include <ten/math>
#include <ten/mcmc>
#include <ten/ml>
#include <ten/process>
#include <ten/random>
#include <ten/sort>
#include <ten/tensor>
#include <ten/types.hxx>

#include <pybind11/attr.h>
#include <pybind11/functional.h>
#include <pybind11/operators.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

namespace py = pybind11;

// Tensor
using tensor_float = ten::tensor<float>;
using tensor_double = ten::tensor<double>;
using tensor_int32 = ten::tensor<int32_t>;
using tensor_int64 = ten::tensor<int64_t>;
using tensor_uint32 = ten::tensor<uint32_t>;
using tensor_uint64 = ten::tensor<uint64_t>;
using tensor_bool = ten::tensor<bool>;
using tensor_c = ten::tensor<std::complex<float>>;
using tensor_z = ten::tensor<std::complex<double>>;

// Diagonal
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

// Column
using column_float = ten::column<float>;
using column_double = ten::column<double>;

// Row
using row_float = ten::row<float>;
using row_double = ten::row<double>;

////////////////////////////////////////////////////////////////////////////////
// Unary expr functions

template <typename T> auto py_min(ten::tensor<T> &x) {
  return ten::min(x).eval();
}

template <typename T> auto py_max(ten::tensor<T> &x) {
  return ten::max(x).eval();
}

template <typename T> auto py_mean(ten::tensor<T> &x) {
  return ten::mean(x).eval();
}

template <typename T> auto py_sum(ten::tensor<T> &x) {
  return ten::sum(x).eval();
}

template <typename T> auto py_cum_sum(ten::tensor<T> &x) {
  return ten::cum_sum(x).eval();
}

template <typename T> auto py_prod(ten::tensor<T> &x) {
  return ten::prod(x).eval();
}

template <typename T> auto py_abs(ten::tensor<T> &x) {
  return ten::abs(x).eval();
}

template <typename T> auto py_sqrt(ten::tensor<T> &x) {
  return ten::sqrt(x).eval();
}

template <typename T> auto py_sqr(ten::tensor<T> &x) {
  return ten::sqr(x).eval();
}

template <typename T> auto py_sin(ten::tensor<T> &x) {
  return ten::sin(x).eval();
}

template <typename T> auto py_sinh(ten::tensor<T> &x) {
  return ten::sinh(x).eval();
}

template <typename T> auto py_asin(ten::tensor<T> &x) {
  return ten::asin(x).eval();
}

template <typename T> auto py_cos(ten::tensor<T> &x) {
  return ten::cos(x).eval();
}

template <typename T> auto py_cosh(ten::tensor<T> &x) {
  return ten::cosh(x).eval();
}

template <typename T> auto py_acos(ten::tensor<T> &x) {
  return ten::acos(x).eval();
}

template <typename T> auto py_tan(ten::tensor<T> &x) {
  return ten::tan(x).eval();
}

template <typename T> auto py_tanh(ten::tensor<T> &x) {
  return ten::tanh(x).eval();
}

template <typename T> auto py_atan(ten::tensor<T> &x) {
  return ten::atan(x).eval();
}

template <typename T> auto py_exp(ten::tensor<T> &x) {
  return ten::exp(x).eval();
}

template <typename T> auto py_log(ten::tensor<T> &x) {
  return ten::log(x).eval();
}

template <typename T> auto py_log10(ten::tensor<T> &x) {
  return ten::log10(x).eval();
}

template <typename T> auto py_floor(ten::tensor<T> &x) {
  return ten::floor(x).eval();
}

template <typename T> auto py_ceil(ten::tensor<T> &x) {
  return ten::ceil(x).eval();
}

template <typename T>
auto py_reshape(ten::tensor<T> &x, const std::vector<std::size_t> &dims) {
  return ten::reshape(x, dims).eval();
}

template <typename T> auto py_flatten(ten::tensor<T> &x) {
  return ten::flatten(x).eval();
}

////////////////////////////////////////////////////////////////////////////////
// Parametric unary functions

template <typename T> auto py_pow(ten::tensor<T> &x, T n) {
  return ten::pow(x, n).eval();
}

////////////////////////////////////////////////////////////////////////////////
// Binary func mul

template <typename T> auto py_add(ten::tensor<T> &x, ten::tensor<T> &y) {
  return (x + y).eval();
}

template <typename T> auto py_sub(ten::tensor<T> &x, ten::tensor<T> &y) {
  return (x - y).eval();
}

template <typename T> auto py_mul(ten::tensor<T> &x, ten::tensor<T> &y) {
  return (x * y).eval();
}

template <typename T> auto py_div(ten::tensor<T> &x, ten::tensor<T> &y) {
  return (x / y).eval();
}

template <typename T> auto py_add_left(T s, ten::tensor<T> &x) {
  return (s + x).eval();
}

template <typename T> auto py_add_right(ten::tensor<T> &x, T s) {
  return (x + s).eval();
}

template <typename T> auto py_sub_left(T s, ten::tensor<T> &x) {
  return (s - x).eval();
}

template <typename T> auto py_sub_right(ten::tensor<T> &x, T s) {
  return (x - s).eval();
}

template <typename T> auto py_mul_left(T s, ten::tensor<T> &x) {
  return (s * x).eval();
}

template <typename T> auto py_mul_right(ten::tensor<T> &x, T s) {
  return (x * s).eval();
}

template <typename T> auto py_div_left(T s, ten::tensor<T> &x) {
  return (s / x).eval();
}

template <typename T> auto py_div_right(ten::tensor<T> &x, T s) {
  return (x / s).eval();
}

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
// Linear algebra

template <typename T> auto py_qr(const ten::tensor<T> &a) {
  return ten::linalg::qr(a);
}

template <typename T> auto py_lu(const ten::tensor<T> &a) {
  return ten::linalg::lu(a);
}

template <typename T> auto py_cholesky(const ten::tensor<T> &a) {
  return ten::linalg::cholesky(a);
}

template <typename T> auto py_svd(const ten::tensor<T> &a) {
  return ten::linalg::svd(a);
}

template <typename T> auto py_ls_qr(ten::tensor<T> &A, ten::tensor<T> &b) {
  return ten::linalg::solve(A, b, ten::linalg::ls_method::qr);
}

template <typename T> auto py_ls_lu(ten::tensor<T> &A, ten::tensor<T> &b) {
  return ten::linalg::solve(A, b, ten::linalg::ls_method::lu);
}

template <typename T> auto py_ls_svd(ten::tensor<T> &A, ten::tensor<T> &b) {
  return ten::linalg::solve(A, b, ten::linalg::ls_method::svd);
}

////////////////////////////////////////////////////////////////////////////////
// Least squares

template <typename T>
auto py_lsqr(ten::tensor<T> &X, ten::tensor<T> &y,
             ten::linalg::ls_method method) {
  return ten::linalg::lsqr(X, y, method);
}

////////////////////////////////////////////////////////////////////////////////
// Sort

template <typename T> auto py_sort(const ten::tensor<T> &x) {
  return ten::sort(x);
}

template <typename T> void py_sort_inplace(ten::tensor<T> &x) {
  ten::sort_inplace(x);
}

////////////////////////////////////////////////////////////////////////////////
// Machine learning

using histogram_options = ten::ml::histogram_options;
using histogram_float = ten::ml::histogram<float>;
using histogram_double = ten::ml::histogram<double>;

using lm_method = ten::ml::lm_method;

using linear_model_float = ten::ml::linear_model<float>;
using linear_model_double = ten::ml::linear_model<double>;

using polyreg_float = ten::ml::polyreg<float>;
using polyreg_double = ten::ml::polyreg<double>;

////////////////////////////////////////////////////////////////////////////////
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

// Create a new diagonal matrix
template <typename T>
ten::diagonal<T> py_make_diagonal(
    const std::vector<std::size_t> &dims, const bool requires_grad = false,
    const ten::storage_order order = ten::storage_order::col_major) {
  return ten::diagonal<T>(dims, requires_grad, order);
}

////////////////////////////////////////////////////////////////////////////////
// Random functions

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
// Maths

template <typename T>
auto py_inv_sample(std::size_t size, std::function<T(T)> Finv) {
  return ten::inv_sample<T>(size, Finv);
}

////////////////////////////////////////////////////////////////////////////////
// MCMC

template <typename T>
auto py_mcmc(T xt, std::size_t n, std::function<T(T)> f, std::function<T(T)> g,
             std::size_t burn) {
  return ten::mcmc(xt, n, f, g, burn);
}

template <typename T>
auto py_mcmc2(T xt, std::size_t n, std::function<T(T)> f, std::size_t burn) {
  return ten::mcmc(xt, n, f, burn);
}

////////////////////////////////////////////////////////////////////////////////
// Process

template <typename T, typename Prob>
auto py_random_walk(std::size_t n, T k, Prob prob, T inc) {
  ten::random_walk_options<T, Prob> options{.k = k, .prob = prob, .inc = inc};
  return ten::random_walk<T, Prob>(n, options);
}

template <typename T, typename Prob>
auto py_random_walk_paths(std::size_t n, std::size_t t, T k, Prob prob, T inc) {
  ten::random_walk_options<T, Prob> options{.k = k, .prob = prob, .inc = inc};
  return ten::random_walk_paths<T, Prob>(n, t, options);
}

template <typename T>
auto py_continuous_random_walk(ten::tensor<T> &s, std::size_t t,
                               std::size_t n) {
  return ten::continuous_random_walk(s, t, n);
}

template <typename T>
auto py_brownian_motion(ten::tensor<T> &s, std::size_t t) {
  return ten::brownian_motion(s, t);
}

template <typename T>
auto py_brownian_motion_paths(ten::tensor<T> &s, std::size_t n, std::size_t t) {
  return ten::brownian_motion(s, n, t);
}

template <typename T>
auto py_brownian_motion_all_paths(std::size_t n, std::size_t t) {
  return ten::brownian_motion(n, t);
}

////////////////////////////////////////////////////////////////////////////////
// Graphs

using graph_type = ten::graph::graph_type;

using glist_int32 = ten::graph::glist<int32_t>;
using glist_uint32 = ten::graph::glist<uint32_t>;
using glist_int64 = ten::graph::glist<int64_t>;
using glist_uint64 = ten::graph::glist<uint64_t>;
using glist_str = ten::graph::glist<std::string>;

using gedge_int32 = ten::graph::gedge<int32_t>;
using gedge_uint32 = ten::graph::gedge<uint32_t>;
using gedge_int64 = ten::graph::gedge<int64_t>;
using gedge_uint64 = ten::graph::gedge<uint64_t>;
using gedge_str = ten::graph::gedge<std::string>;

using gmatrix_float = ten::graph::gmatrix<float>;
using gmatrix_double = ten::graph::gmatrix<double>;

using gweighted_int32_int32 = ten::graph::gweighted<int32_t, int32_t>;
using gweighted_int32_float = ten::graph::gweighted<int32_t, float>;
using gweighted_int32_double = ten::graph::gweighted<int32_t, double>;
using gweighted_int64_int64 = ten::graph::gweighted<int64_t, int64_t>;
using gweighted_int64_float = ten::graph::gweighted<int64_t, float>;
using gweighted_int64_double = ten::graph::gweighted<int64_t, double>;
using gweighted_uint32_uint32 = ten::graph::gweighted<uint32_t, uint32_t>;
using gweighted_uint32_float = ten::graph::gweighted<uint32_t, float>;
using gweighted_uint32_double = ten::graph::gweighted<uint32_t, double>;
using gweighted_uint64_uint64 = ten::graph::gweighted<uint64_t, uint64_t>;
using gweighted_uint64_float = ten::graph::gweighted<uint64_t, float>;
using gweighted_uint64_double = ten::graph::gweighted<uint64_t, double>;
using gweighted_str_float = ten::graph::gweighted<std::string, float>;
using gweighted_str_double = ten::graph::gweighted<std::string, double>;

////////////////////////////////////////////////////////////////////////////////
// Compute linear index

std::size_t py_linear_index(const std::vector<std::size_t> &strides,
                            const std::vector<std::size_t> &indices) {
  std::size_t index = 0;
  for (std::size_t i = 0; i < strides.size(); i++) {
    index += indices[i] * strides[i];
  }
  return index;
}

////////////////////////////////////////////////////////////////////////////////
// Bindings for the core libray

PYBIND11_MODULE(tencore, m) {
  m.doc() = "Ten core: Bindings for the ten library";

  // Scalars
  py::class_<scalar_float>(m, "scalar_float")
      .def(py::init<const float &>())
      .def("value", [](const scalar_float &s) { return s.value(); })
      .def("__repr__", [](const scalar_float &s) {
        std::stringstream ss;
        ss << s;
        return ss.str();
      });

  py::class_<scalar_double>(m, "scalar_double")
      .def(py::init<const double &>())
      .def("value", [](const scalar_double &s) { return s.value(); })
      .def("__repr__", [](const scalar_double &s) {
        std::stringstream ss;
        ss << s;
        return ss.str();
      });

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

  // Linear index from indices and strides
  m.def("linear_index", &py_linear_index);

  // Tensor float
  py::class_<tensor_float>(m, "tensor_float")
      .def("make", &py_make_tensor<float>)
      .def("rank", &tensor_float::rank)
      .def("size", [](const tensor_float &t) { return t.size(); })
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
      .def("col", &tensor_float::col)
      .def("row", &tensor_float::row)
      .def("__repr__", [](const tensor_float &t) {
        std::stringstream ss;
        ss << t;
        return ss.str();
      });

  // Tensor double
  py::class_<tensor_double>(m, "tensor_double")
      .def("make", &py_make_tensor<double>)
      .def("rank", &tensor_double::rank)
      .def("size", [](const tensor_double &t) { return t.size(); })
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
      .def("col", &tensor_double::col)
      .def("row", &tensor_double::row)
      .def("__repr__", [](const tensor_double &t) {
        std::stringstream ss;
        ss << t;
        return ss.str();
      });

  // Tensor int32
  py::class_<tensor_int32>(m, "tensor_int32")
      .def("make", &py_make_tensor<int32_t>)
      .def("rank", &tensor_int32::rank)
      .def("size", [](const tensor_int32 &t) { return t.size(); })
      .def("shape", &tensor_int32::shape)
      .def("dim", &tensor_int32::dim)
      .def("strides", &tensor_int32::strides)
      .def("data_type", &tensor_int32::data_type)
      .def("format", &tensor_int32::format)
      .def("storage_order", &tensor_int32::storage_order)
      .def("__getitem__",
           [](const tensor_int32 &t, std::size_t index) { return t[index]; })
      .def("__setitem__", [](tensor_int32 &t, std::size_t index,
                             int32_t value) { t[index] = value; })
      .def("is_transposed", &tensor_int32::is_transposed)
      .def("is_symmetric", &tensor_int32::is_symmetric)
      .def("is_hermitian", &tensor_int32::is_hermitian)
      .def("is_diagonal", &tensor_int32::is_diagonal)
      .def("is_lower_tr", &tensor_int32::is_lower_tr)
      .def("is_upper_tr", &tensor_int32::is_upper_tr)
      .def("col", &tensor_int32::col)
      .def("row", &tensor_int32::row)
      .def("__repr__", [](const tensor_int32 &t) {
        std::stringstream ss;
        ss << t;
        return ss.str();
      });

  // Tensor int64
  py::class_<tensor_int64>(m, "tensor_int64")
      .def("make", &py_make_tensor<int64_t>)
      .def("rank", &tensor_int64::rank)
      .def("size", [](const tensor_int64 &t) { return t.size(); })
      .def("shape", &tensor_int64::shape)
      .def("dim", &tensor_int64::dim)
      .def("strides", &tensor_int64::strides)
      .def("data_type", &tensor_int64::data_type)
      .def("format", &tensor_int64::format)
      .def("storage_order", &tensor_int64::storage_order)
      .def("__getitem__",
           [](const tensor_int64 &t, std::size_t index) { return t[index]; })
      .def("__setitem__", [](tensor_int64 &t, std::size_t index,
                             int64_t value) { t[index] = value; })
      .def("is_transposed", &tensor_int64::is_transposed)
      .def("is_symmetric", &tensor_int64::is_symmetric)
      .def("is_hermitian", &tensor_int64::is_hermitian)
      .def("is_diagonal", &tensor_int64::is_diagonal)
      .def("is_lower_tr", &tensor_int64::is_lower_tr)
      .def("is_upper_tr", &tensor_int64::is_upper_tr)
      .def("col", &tensor_int64::col)
      .def("row", &tensor_int64::row)
      .def("__repr__", [](const tensor_int64 &t) {
        std::stringstream ss;
        ss << t;
        return ss.str();
      });

  // Tensor uint32
  py::class_<tensor_uint32>(m, "tensor_uint32")
      .def("make", &py_make_tensor<uint32_t>)
      .def("rank", &tensor_uint32::rank)
      .def("size", [](const tensor_uint32 &t) { return t.size(); })
      .def("shape", &tensor_uint32::shape)
      .def("dim", &tensor_uint32::dim)
      .def("strides", &tensor_uint32::strides)
      .def("data_type", &tensor_uint32::data_type)
      .def("format", &tensor_uint32::format)
      .def("storage_order", &tensor_uint32::storage_order)
      .def("__getitem__",
           [](const tensor_uint32 &t, std::size_t index) { return t[index]; })
      .def("__setitem__", [](tensor_uint32 &t, std::size_t index,
                             uint32_t value) { t[index] = value; })
      .def("is_transposed", &tensor_uint32::is_transposed)
      .def("is_symmetric", &tensor_uint32::is_symmetric)
      .def("is_hermitian", &tensor_uint32::is_hermitian)
      .def("is_diagonal", &tensor_uint32::is_diagonal)
      .def("is_lower_tr", &tensor_uint32::is_lower_tr)
      .def("is_upper_tr", &tensor_uint32::is_upper_tr)
      .def("col", &tensor_uint32::col)
      .def("row", &tensor_uint32::row)
      .def("__repr__", [](const tensor_uint32 &t) {
        std::stringstream ss;
        ss << t;
        return ss.str();
      });

  // Tensor uint64
  py::class_<tensor_uint64>(m, "tensor_uint64")
      .def("make", &py_make_tensor<uint64_t>)
      .def("rank", &tensor_uint64::rank)
      .def("size", [](const tensor_uint64 &t) { return t.size(); })
      .def("shape", &tensor_uint64::shape)
      .def("dim", &tensor_uint64::dim)
      .def("strides", &tensor_uint64::strides)
      .def("data_type", &tensor_uint64::data_type)
      .def("format", &tensor_uint64::format)
      .def("storage_order", &tensor_uint64::storage_order)
      .def("__getitem__",
           [](const tensor_uint64 &t, std::size_t index) { return t[index]; })
      .def("__setitem__", [](tensor_uint64 &t, std::size_t index,
                             int64_t value) { t[index] = value; })
      .def("is_transposed", &tensor_uint64::is_transposed)
      .def("is_symmetric", &tensor_uint64::is_symmetric)
      .def("is_hermitian", &tensor_uint64::is_hermitian)
      .def("is_diagonal", &tensor_uint64::is_diagonal)
      .def("is_lower_tr", &tensor_uint64::is_lower_tr)
      .def("is_upper_tr", &tensor_uint64::is_upper_tr)
      .def("col", &tensor_uint64::col)
      .def("row", &tensor_uint64::row)
      .def("__repr__", [](const tensor_uint64 &t) {
        std::stringstream ss;
        ss << t;
        return ss.str();
      });

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

  // Diagonal float
  py::class_<diagonal_float>(m, "diagonal_float")
      .def("make", &py_make_diagonal<float>)
      .def("rank", &diagonal_float::rank)
      .def("size", &diagonal_float::size)
      .def("shape", &diagonal_float::shape)
      .def("dim", &diagonal_float::dim)
      .def("strides", &diagonal_float::strides)
      .def("data_type", &diagonal_float::data_type)
      .def("format", &diagonal_float::format)
      .def("storage_order", &diagonal_float::storage_order)
      .def("__getitem__",
           [](const diagonal_float &t, std::size_t index) { return t[index]; })
      .def("__getitem__",
           [](const diagonal_float &t,
              std::tuple<std::size_t, std::size_t> index) {
             return t(std::get<0>(index), std::get<1>(index));
           })
      .def("__setitem__", [](diagonal_float &t, std::size_t index,
                             double value) { t[index] = value; })
      .def("__setitem__",
           [](diagonal_float &t, std::tuple<std::size_t, std::size_t> index,
              double value) {
             t(std::get<0>(index), std::get<1>(index)) = value;
           })
      .def("is_diagonal", &diagonal_float::is_diagonal)
      .def("__repr__", [](const diagonal_float &t) {
        std::stringstream ss;
        ss << t;
        return ss.str();
      });

  // Transform diagonal to dense
  m.def("dense_float", &ten::dense<diagonal_float>);
  m.def("dense_double", &ten::dense<diagonal_double>);

  // Column
  py::class_<column_float>(m, "column_float")
      .def("data_type", &column_float::data_type)
      .def("size", &column_float::size)
      .def("shape", &column_float::shape)
      .def("__getitem__",
           [](const column_float &t, std::size_t index) { return t[index]; })
      .def("__setitem__", [](column_float &t, std::size_t index,
                             float value) { t[index] = value; })
      .def("__repr__", [](const column_float &t) {
        std::stringstream ss;
        ss << t;
        return ss.str();
      });

  py::class_<column_double>(m, "column_double")
      .def("data_type", &column_double::data_type)
      .def("size", &column_double::size)
      .def("shape", &column_double::shape)
      .def("__getitem__",
           [](const column_double &t, std::size_t index) { return t[index]; })
      .def("__setitem__", [](column_double &t, std::size_t index,
                             double value) { t[index] = value; })
      .def("__repr__", [](const column_double &t) {
        std::stringstream ss;
        ss << t;
        return ss.str();
      });

  // Row
  py::class_<row_float>(m, "row_float")
      .def("data_type", &row_float::data_type)
      .def("size", &row_float::size)
      .def("shape", &row_float::shape)
      .def("__getitem__",
           [](const row_float &t, std::size_t index) { return t[index]; })
      .def("__setitem__", [](row_float &t, std::size_t index,
                             float value) { t[index] = value; })
      .def("__repr__", [](const row_float &t) {
        std::stringstream ss;
        ss << t;
        return ss.str();
      });

  py::class_<row_double>(m, "row_double")
      .def("data_type", &row_double::data_type)
      .def("size", &row_double::size)
      .def("shape", &row_double::shape)
      .def("__getitem__",
           [](const row_double &t, std::size_t index) { return t[index]; })
      .def("__setitem__", [](row_double &t, std::size_t index,
                             double value) { t[index] = value; })
      .def("__repr__", [](const row_double &t) {
        std::stringstream ss;
        ss << t;
        return ss.str();
      });

  /*
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
  m.def("cast_tensor_float_to_double", &ten::cast<double, vector_float>);
  m.def("cast_tensor_double_to_float", &ten::cast<float, vector_double>);

  // Reshape
  // vector to matrix
  m.def("reshape_vector_float_matrix_float",
        &py_reshape<vector_float, ten::shape<0, 0>>);
  m.def("reshape_vector_double_matrix_double",
        &py_reshape<vector_double, ten::shape<0, 0>>);

  // Flatten
  m.def("flatten_matrix_float", &ten::flatten<matrix_float>);
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

  //////////////////////////////////////////////////////////////////////////////
  // Functions

  m.def("min_float", &py_min<float>);
  m.def("min_double", &py_min<double>);

  m.def("max_float", &py_max<float>);
  m.def("max_double", &py_max<double>);

  m.def("mean_float", &py_mean<float>);
  m.def("mean_double", &py_mean<double>);

  m.def("sum_float", &py_sum<float>);
  m.def("sum_double", &py_sum<double>);

  m.def("cum_sum_float", &py_cum_sum<float>);
  m.def("cum_sum_double", &py_cum_sum<float>);

  m.def("prod_float", &py_prod<float>);
  m.def("prod_double", &py_prod<double>);

  m.def("abs_float", &py_abs<float>);
  m.def("abs_double", &py_abs<double>);

  m.def("sqrt_float", &py_sqrt<float>);
  m.def("sqrt_double", &py_sqrt<double>);

  m.def("sqr_float", &py_sqr<float>);
  m.def("sqr_double", &py_sqr<double>);

  m.def("sin_float", &py_sin<float>);
  m.def("sin_double", &py_sin<double>);

  m.def("sinh_float", &py_sinh<float>);
  m.def("sinh_double", &py_sinh<double>);

  m.def("asin_float", &py_asin<float>);
  m.def("asin_double", &py_asin<double>);

  m.def("cos_float", &py_cos<float>);
  m.def("cos_double", &py_cos<double>);

  m.def("cosh_float", &py_cosh<float>);
  m.def("cosh_double", &py_cosh<double>);

  m.def("acos_float", &py_acos<float>);
  m.def("acos_double", &py_acos<double>);

  m.def("tan_float", &py_tan<float>);
  m.def("tan_double", &py_tan<double>);

  m.def("tanh_float", &py_tanh<float>);
  m.def("tanh_double", &py_tanh<double>);

  m.def("atan_float", &py_atan<float>);
  m.def("atan_double", &py_atan<double>);

  m.def("exp_float", &py_exp<float>);
  m.def("exp_double", &py_exp<double>);

  m.def("log_float", &py_log<float>);
  m.def("log_double", &py_log<double>);

  m.def("log10_float", &py_log10<float>);
  m.def("log10_double", &py_log10<double>);

  m.def("floor_float", &py_floor<float>);
  m.def("floor_double", &py_floor<double>);

  m.def("ceil_float", &py_ceil<float>);
  m.def("ceil_double", &py_ceil<double>);

  m.def("reshape_float", &py_reshape<float>);
  m.def("reshape_double", &py_reshape<double>);

  m.def("flatten_float", &py_flatten<float>);
  m.def("flatten_double", &py_flatten<double>);

  m.def("pow_float", &py_pow<float>);
  m.def("pow_double", &py_pow<double>);

  /////////////////////////////////////////////////////////////////////////////
  // Binary functions
  m.def("add_float", &py_add<float>);
  m.def("add_double", &py_add<double>);

  m.def("sub_float", &py_sub<float>);
  m.def("sub_double", &py_sub<double>);

  m.def("mul_float", &py_mul<float>);
  m.def("mul_double", &py_mul<double>);

  m.def("div_float", &py_div<float>);
  m.def("div_double", &py_div<double>);

  m.def("add_left_float", &py_add_left<float>);
  m.def("add_left_double", &py_add_left<double>);

  m.def("add_right_float", &py_add_right<float>);
  m.def("add_right_double", &py_add_right<double>);

  m.def("sub_left_float", &py_sub_left<float>);
  m.def("sub_left_double", &py_sub_left<double>);

  m.def("sub_right_float", &py_sub_right<float>);
  m.def("sub_right_double", &py_sub_right<double>);

  m.def("mul_left_float", &py_mul_left<float>);
  m.def("mul_left_double", &py_mul_left<double>);

  m.def("mul_right_float", &py_mul_right<float>);
  m.def("mul_right_double", &py_mul_right<double>);

  m.def("div_left_float", &py_div_left<float>);
  m.def("div_left_double", &py_div_left<double>);

  m.def("div_right_float", &py_div_right<float>);
  m.def("div_right_double", &py_div_right<double>);

  /////////////////////////////////////////////////////////////////////////////
  // Distributions
  m.def("set_seed", &ten::set_seed);

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

  //////////////////////////////////////////////////////////////////////////////
  // Linear algebra

  m.def("qr_float", &py_qr<float>);
  m.def("qr_double", &py_qr<double>);

  m.def("lu_float", &py_lu<float>);
  m.def("lu_double", &py_lu<double>);

  m.def("cholesky_float", &py_cholesky<float>);
  m.def("cholesky_double", &py_cholesky<double>);

  m.def("svd_float", &py_svd<float>);
  m.def("svd_double", &py_svd<double>);

  m.def("ls_qr_float", &py_ls_qr<float>);
  m.def("ls_qr_double", &py_ls_qr<double>);

  m.def("ls_lu_float", &py_ls_lu<float>);
  m.def("ls_lu_double", &py_ls_lu<double>);

  m.def("ls_svd_float", &py_ls_svd<float>);
  m.def("ls_svd_double", &py_ls_svd<double>);

  // linear system method
  py::enum_<ten::linalg::ls_method>(m, "ls_method", py::arithmetic())
      .value("qr", ten::linalg::ls_method::qr)
      .value("lu", ten::linalg::ls_method::lu)
      .value("svd", ten::linalg::ls_method::svd);

  // Least squares
  m.def("lsqr_float", &py_lsqr<float>);
  m.def("lsqr_double", &py_lsqr<double>);

  //////////////////////////////////////////////////////////////////////////////
  // Sort

  m.def("sort_float", &py_sort<float>);
  m.def("sort_double", &py_sort<double>);

  m.def("sort_inplace_float", &py_sort_inplace<float>);
  m.def("sort_inplace_double", &py_sort_inplace<double>);

  /////////////////////////////////////////////////////////////////////////////
  // Maths

  m.def("inv_sample_float", &py_inv_sample<float>);
  m.def("inv_sample_double", &py_inv_sample<double>);

  /////////////////////////////////////////////////////////////////////////////
  // MCMC

  m.def("mcmc_float", &py_mcmc<float>);
  m.def("mcmc_double", &py_mcmc<double>);

  m.def("mcmc2_float", &py_mcmc2<float>);
  m.def("mcmc2_double", &py_mcmc2<double>);

  /////////////////////////////////////////////////////////////////////////////
  // Process

  m.def("random_walk_float_float", &py_random_walk<float, float>);
  m.def("random_walk_int64_float", &py_random_walk<int64_t, float>);

  m.def("random_walk_double_double", &py_random_walk<double, double>);
  m.def("random_walk_int64_double", &py_random_walk<int64_t, double>);

  m.def("random_walk_paths_float_float", &py_random_walk_paths<float, float>);
  m.def("random_walk_paths_int64_float", &py_random_walk_paths<int64_t, float>);

  m.def("random_walk_paths_double_double",
        &py_random_walk_paths<double, double>);
  m.def("random_walk_paths_int64_double",
        &py_random_walk_paths<int64_t, double>);

  m.def("continuous_random_walk_float", &py_continuous_random_walk<float>);
  m.def("continuous_random_walk_double", &py_continuous_random_walk<double>);

  m.def("brownian_motion_float", &py_brownian_motion<float>);
  m.def("brownian_motion_double", &py_brownian_motion<double>);

  m.def("brownian_motion_paths_float", &py_brownian_motion_paths<float>);
  m.def("brownian_motion_paths_double", &py_brownian_motion_paths<double>);

  m.def("brownian_motion_all_paths_float",
        &py_brownian_motion_all_paths<float>);
  m.def("brownian_motion_all_paths_double",
        &py_brownian_motion_all_paths<double>);

  /////////////////////////////////////////////////////////////////////////////
  // Combinatorics

  m.def("factorial_uint64", &ten::factorial<uint64_t>);
  m.def("perm_uint64", &ten::perm<uint64_t>);
  m.def("comb_uint64", &ten::comb<uint64_t>);
  m.def("pascal_triangle_uint64", &ten::pascal_triangle<uint64_t>);
  m.def("nth_pascal_triangle_uint64", &ten::nth_pascal_triangle<uint64_t>);

  /////////////////////////////////////////////////////////////////////////////
  // learning

  py::class_<histogram_options>(m, "histogram_options")
      .def(py::init<bool, bool, bool, std::size_t>());

  py::class_<histogram_float>(m, "histogram_float")
      .def(py::init<histogram_options>())
      .def("fit", &histogram_float::fit)
      .def("hist", &histogram_float::hist);

  py::class_<histogram_double>(m, "histogram_double")
      .def(py::init<histogram_options>())
      .def("fit", &histogram_double::fit)
      .def("hist", &histogram_double::hist);

  // linear model method
  py::enum_<lm_method>(m, "lm_method", py::arithmetic())
      .value("qr", ten::ml::lm_method::qr)
      .value("lu", ten::ml::lm_method::lu)
      .value("svd", ten::ml::lm_method::svd)
      .value("gd", ten::ml::lm_method::gd);

  // Linear model
  py::class_<linear_model_float>(m, "linear_model_float")
      .def(py::init<lm_method>())
      .def("fit", &linear_model_float::fit)
      .def("coef", &linear_model_float::coef)
      .def("fitted", &linear_model_float::fitted)
      .def("predict", &linear_model_float::predict);

  py::class_<linear_model_double>(m, "linear_model_double")
      .def(py::init<lm_method>())
      .def("fit", &linear_model_double::fit)
      .def("coef", &linear_model_double::coef)
      .def("fitted", &linear_model_double::fitted)
      .def("predict", &linear_model_double::predict);


  // Polyreg
  py::class_<polyreg_float>(m, "polyreg_float")
      .def(py::init<std::size_t>())
      .def("fit", &polyreg_float::fit)
      .def("coef", &polyreg_float::coef)
      .def("fitted", &polyreg_float::fitted)
      .def("predict", &polyreg_float::predict);

  py::class_<polyreg_double>(m, "polyreg_double")
      .def(py::init<std::size_t>())
      .def("fit", &polyreg_double::fit)
      .def("coef", &polyreg_double::coef)
      .def("fitted", &polyreg_double::fitted)
      .def("predict", &polyreg_double::predict);

  /////////////////////////////////////////////////////////////////////////////
  // Graphs

  py::enum_<graph_type>(m, "graph_type", py::arithmetic())
      .value("undirected", ten::graph::graph_type::undirected)
      .value("directed", ten::graph::graph_type::directed);

  // List
  py::class_<glist_int32>(m, "glist_int32")
      .def(py::init<graph_type>())
      .def("empty", &glist_int32::empty)
      .def("add_vertex", &glist_int32::add_vertex)
      .def("add_edge", &glist_int32::add_edge)
      .def("has_edge", &glist_int32::has_edge)
      .def("to_matrix_float", &glist_int32::to_matrix<float>)
      .def("to_matrix_double", &glist_int32::to_matrix<double>)
      .def("to_matrix_int32", &glist_int32::to_matrix<int32_t>)
      .def("to_matrix_uint32", &glist_int32::to_matrix<uint32_t>)
      .def("to_matrix_int64", &glist_int32::to_matrix<int64_t>)
      .def("to_matrix_uint64", &glist_int32::to_matrix<uint64_t>)
      .def("dfs", [](glist_int32 &g, int32_t s,
                     std::function<void(int32_t)> F) { return g.dfs(s, F); })
      .def("bfs",
           [](glist_int32 &g, int32_t s, std::function<void(int32_t)> F) {
             return g.bfs(s, F);
           });

  py::class_<glist_uint32>(m, "glist_uint32")
      .def(py::init<graph_type>())
      .def("empty", &glist_uint32::empty)
      .def("add_vertex", &glist_uint32::add_vertex)
      .def("add_edge", &glist_uint32::add_edge)
      .def("has_edge", &glist_uint32::has_edge)
      .def("to_matrix_float", &glist_uint32::to_matrix<float>)
      .def("to_matrix_double", &glist_uint32::to_matrix<double>)
      .def("to_matrix_int32", &glist_uint32::to_matrix<int32_t>)
      .def("to_matrix_uint32", &glist_uint32::to_matrix<uint32_t>)
      .def("to_matrix_int64", &glist_uint32::to_matrix<int64_t>)
      .def("to_matrix_uint64", &glist_uint32::to_matrix<uint64_t>)
      .def("dfs",
           [](glist_uint32 &g, uint32_t s,
              std::function<void(uint32_t)> F) { return g.dfs(s, F); })
      .def("bfs",
           [](glist_uint32 &g, uint32_t s,
              std::function<void(uint32_t)> F) { return g.bfs(s, F); });

  py::class_<glist_int64>(m, "glist_int64")
      .def(py::init<graph_type>())
      .def("empty", &glist_int64::empty)
      .def("add_vertex", &glist_int64::add_vertex)
      .def("add_edge", &glist_int64::add_edge)
      .def("has_edge", &glist_int64::has_edge)
      .def("to_matrix_float", &glist_int64::to_matrix<float>)
      .def("to_matrix_double", &glist_int64::to_matrix<double>)
      .def("to_matrix_int32", &glist_int64::to_matrix<int32_t>)
      .def("to_matrix_uint32", &glist_int64::to_matrix<uint32_t>)
      .def("to_matrix_int64", &glist_int64::to_matrix<int64_t>)
      .def("to_matrix_uint64", &glist_int64::to_matrix<uint64_t>)
      .def("dfs", [](glist_int64 &g, int64_t s,
                     std::function<void(int64_t)> F) { return g.dfs(s, F); })
      .def("bfs",
           [](glist_int64 &g, int64_t s, std::function<void(int64_t)> F) {
             return g.bfs(s, F);
           });

  py::class_<glist_uint64>(m, "glist_uint64")
      .def(py::init<graph_type>())
      .def("empty", &glist_uint64::empty)
      .def("add_vertex", &glist_uint64::add_vertex)
      .def("add_edge", &glist_uint64::add_edge)
      .def("has_edge", &glist_uint64::has_edge)
      .def("to_matrix_float", &glist_uint64::to_matrix<float>)
      .def("to_matrix_double", &glist_uint64::to_matrix<double>)
      .def("to_matrix_int32", &glist_uint64::to_matrix<int32_t>)
      .def("to_matrix_uint32", &glist_uint64::to_matrix<uint32_t>)
      .def("to_matrix_int64", &glist_uint64::to_matrix<int64_t>)
      .def("to_matrix_uint64", &glist_uint64::to_matrix<uint64_t>)
      .def("dfs",
           [](glist_uint64 &g, uint64_t s,
              std::function<void(uint64_t)> F) { return g.dfs(s, F); })
      .def("bfs",
           [](glist_uint64 &g, uint64_t s,
              std::function<void(uint64_t)> F) { return g.bfs(s, F); });

  py::class_<glist_str>(m, "glist_str")
      .def(py::init<graph_type>())
      .def("empty", &glist_str::empty)
      .def("add_vertex", &glist_str::add_vertex)
      .def("add_edge", &glist_str::add_edge)
      .def("has_edge", &glist_str::has_edge)
      .def("to_matrix_float", &glist_str::to_matrix<float>)
      .def("to_matrix_double", &glist_str::to_matrix<double>)
      .def("to_matrix_int32", &glist_str::to_matrix<int32_t>)
      .def("to_matrix_uint32", &glist_str::to_matrix<uint32_t>)
      .def("to_matrix_int64", &glist_str::to_matrix<int64_t>)
      .def("to_matrix_uint64", &glist_str::to_matrix<uint64_t>)
      .def(
          "dfs",
          [](glist_str &g, std::string s,
             std::function<void(std::string)> F) { return g.dfs(s, F); })
      .def("bfs", [](glist_str &g, std::string s,
                     std::function<void(std::string)> F) {
        return g.bfs(s, F);
      });

  // Edge list
  py::class_<gedge_int32>(m, "gedge_int32")
      .def(py::init<graph_type>())
      .def("empty", &gedge_int32::empty)
      .def("add_edge", &gedge_int32::add_edge);

  py::class_<gedge_uint32>(m, "gedge_uint32")
      .def(py::init<graph_type>())
      .def("empty", &gedge_uint32::empty)
      .def("add_edge", &gedge_uint32::add_edge);

  py::class_<gedge_int64>(m, "gedge_int64")
      .def(py::init<graph_type>())
      .def("empty", &gedge_int64::empty)
      .def("add_edge", &gedge_int64::add_edge);

  py::class_<gedge_uint64>(m, "gedge_uint64")
      .def(py::init<graph_type>())
      .def("empty", &gedge_uint64::empty)
      .def("add_edge", &gedge_uint64::add_edge);

  py::class_<gedge_str>(m, "gedge_str")
      .def(py::init<graph_type>())
      .def("empty", &gedge_str::empty)
      .def("add_edge", &gedge_str::add_edge);

  // Matrix
  py::class_<gmatrix_float>(m, "gmatrix_float")
      .def(py::init<std::size_t, graph_type>())
      .def("add_edge", &gmatrix_float::add_edge)
      .def("has_edge", &gmatrix_float::has_edge);

  py::class_<gmatrix_double>(m, "gmatrix_double")
      .def(py::init<std::size_t, graph_type>())
      .def("add_edge", &gmatrix_double::add_edge)
      .def("has_edge", &gmatrix_double::has_edge);

  // Weighted
  py::class_<gweighted_int32_int32>(m, "gweighted_int32_int32")
      .def(py::init<graph_type>())
      .def("empty", &gweighted_int32_int32::empty)
      .def("add_vertex", &gweighted_int32_int32::add_vertex)
      .def("add_edge", &gweighted_int32_int32::add_edge)
      .def("has_edge", &gweighted_int32_int32::has_edge)
      .def("weight", &gweighted_int32_int32::weight)
      .def("to_matrix", &gweighted_int32_int32::to_matrix);

  py::class_<gweighted_int32_float>(m, "gweighted_int32_float")
      .def(py::init<graph_type>())
      .def("empty", &gweighted_int32_float::empty)
      .def("add_vertex", &gweighted_int32_float::add_vertex)
      .def("add_edge", &gweighted_int32_float::add_edge)
      .def("has_edge", &gweighted_int32_float::has_edge)
      .def("weight", &gweighted_int32_float::weight)
      .def("to_matrix", &gweighted_int32_float::to_matrix);

  py::class_<gweighted_int32_double>(m, "gweighted_int32_double")
      .def(py::init<graph_type>())
      .def("empty", &gweighted_int32_double::empty)
      .def("add_vertex", &gweighted_int32_double::add_vertex)
      .def("add_edge", &gweighted_int32_double::add_edge)
      .def("has_edge", &gweighted_int32_double::has_edge)
      .def("weight", &gweighted_int32_double::weight)
      .def("to_matrix", &gweighted_int32_double::to_matrix);

  py::class_<gweighted_int64_int64>(m, "gweighted_int64_int64")
      .def(py::init<graph_type>())
      .def("empty", &gweighted_int64_int64::empty)
      .def("add_vertex", &gweighted_int64_int64::add_vertex)
      .def("add_edge", &gweighted_int64_int64::add_edge)
      .def("has_edge", &gweighted_int64_int64::has_edge)
      .def("weight", &gweighted_int64_int64::weight)
      .def("to_matrix", &gweighted_int64_int64::to_matrix);

  py::class_<gweighted_int64_float>(m, "gweighted_int64_float")
      .def(py::init<graph_type>())
      .def("empty", &gweighted_int64_float::empty)
      .def("add_vertex", &gweighted_int64_float::add_vertex)
      .def("add_edge", &gweighted_int64_float::add_edge)
      .def("has_edge", &gweighted_int64_float::has_edge)
      .def("weight", &gweighted_int64_float::weight)
      .def("to_matrix", &gweighted_int64_float::to_matrix);

  py::class_<gweighted_uint32_uint32>(m, "gweighted_uint32_uint32")
      .def(py::init<graph_type>())
      .def("empty", &gweighted_uint32_uint32::empty)
      .def("add_vertex", &gweighted_uint32_uint32::add_vertex)
      .def("add_edge", &gweighted_uint32_uint32::add_edge)
      .def("has_edge", &gweighted_uint32_uint32::has_edge)
      .def("weight", &gweighted_uint32_uint32::weight)
      .def("to_matrix", &gweighted_uint32_uint32::to_matrix);

  py::class_<gweighted_uint32_float>(m, "gweighted_uint32_float")
      .def(py::init<graph_type>())
      .def("empty", &gweighted_uint32_float::empty)
      .def("add_vertex", &gweighted_uint32_float::add_vertex)
      .def("add_edge", &gweighted_uint32_float::add_edge)
      .def("has_edge", &gweighted_uint32_float::has_edge)
      .def("weight", &gweighted_uint32_float::weight)
      .def("to_matrix", &gweighted_uint32_float::to_matrix);

  py::class_<gweighted_uint32_double>(m, "gweighted_uint32_double")
      .def(py::init<graph_type>())
      .def("empty", &gweighted_uint32_double::empty)
      .def("add_vertex", &gweighted_uint32_double::add_vertex)
      .def("add_edge", &gweighted_uint32_double::add_edge)
      .def("has_edge", &gweighted_uint32_double::has_edge)
      .def("weight", &gweighted_uint32_double::weight)
      .def("to_matrix", &gweighted_uint32_double::to_matrix);

  py::class_<gweighted_uint64_uint64>(m, "gweighted_uint64_uint64")
      .def(py::init<graph_type>())
      .def("empty", &gweighted_uint64_uint64::empty)
      .def("add_vertex", &gweighted_uint64_uint64::add_vertex)
      .def("add_edge", &gweighted_uint64_uint64::add_edge)
      .def("has_edge", &gweighted_uint64_uint64::has_edge)
      .def("weight", &gweighted_uint64_uint64::weight)
      .def("to_matrix", &gweighted_uint64_uint64::to_matrix);

  py::class_<gweighted_uint64_float>(m, "gweighted_uint64_float")
      .def(py::init<graph_type>())
      .def("empty", &gweighted_uint64_float::empty)
      .def("add_vertex", &gweighted_uint64_float::add_vertex)
      .def("add_edge", &gweighted_uint64_float::add_edge)
      .def("has_edge", &gweighted_uint64_float::has_edge)
      .def("weight", &gweighted_uint64_float::weight)
      .def("to_matrix", &gweighted_uint64_float::to_matrix);

  py::class_<gweighted_uint64_double>(m, "gweighted_uint64_double")
      .def(py::init<graph_type>())
      .def("empty", &gweighted_uint64_double::empty)
      .def("add_vertex", &gweighted_uint64_double::add_vertex)
      .def("add_edge", &gweighted_uint64_double::add_edge)
      .def("has_edge", &gweighted_uint64_double::has_edge)
      .def("weight", &gweighted_uint64_double::weight)
      .def("to_matrix", &gweighted_uint64_double::to_matrix);

  py::class_<gweighted_str_float>(m, "gweighted_str_float")
      .def(py::init<graph_type>())
      .def("empty", &gweighted_str_float::empty)
      .def("add_vertex", &gweighted_str_float::add_vertex)
      .def("add_edge", &gweighted_str_float::add_edge)
      .def("has_edge", &gweighted_str_float::has_edge)
      .def("weight", &gweighted_str_float::weight)
      .def("to_matrix", &gweighted_str_float::to_matrix);

  py::class_<gweighted_str_double>(m, "gweighted_str_double")
      .def(py::init<graph_type>())
      .def("empty", &gweighted_str_double::empty)
      .def("add_vertex", &gweighted_str_double::add_vertex)
      .def("add_edge", &gweighted_str_double::add_edge)
      .def("has_edge", &gweighted_str_double::has_edge)
      .def("weight", &gweighted_str_double::weight)
      .def("to_matrix", &gweighted_str_double::to_matrix);
}
