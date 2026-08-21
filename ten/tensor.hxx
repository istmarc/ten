// \file ten/tensor.hxx

#ifndef TEN_TENSOR_HXX
#define TEN_TENSOR_HXX

#include <algorithm>
// #include <complex>
//  FIXME move fstream and ostream to io
// #include <fstream>
#include <ostream>

#include <initializer_list>
#include <memory>
#include <optional>
// #include <stdexcept>
// #include <tuple>
#include <type_traits>
#include <utility>
// #include <vector>

#include <ten/expr.hxx>
#include <ten/functional.hxx>
#include <ten/types.hxx>
#include <ten/utils.hxx>

#include <ten/storage/dense_storage.hxx>
// #include <ten/storage/sparse_storage.hxx>

// For expression matching
// #include <ten/matching.hxx>

namespace ten {

// Last element of a sequence
size_t last = 0;

// Sequence
struct seq {
  std::size_t _start;
  std::size_t _end;

  explicit seq(std::size_t start, std::size_t end) : _start(start), _end(end) {}

  explicit seq(std::size_t start) : _start(start), _end(start + 1) {}
};

/*
/// Multidimentional sequence
template <size_t Rank> struct mdseq {
   std::array<size_t, Rank> _start;
   std::array<size_t, Rank> _end;

   mdseq(auto... sequences)
      requires(std::is_same_v<decltype(sequences...[0]), seq> &&
               sizeof...(sequences) == Rank)
   {
      std::array<seq, Rank> arr{sequences...};
      for (size_t i = 0; i < Rank; i++) {
         _start[i] = arr[i]._start;
         _end[i] = arr[i]._end;
      }
   }
};
*/

/*
// Add two expr
template <Expr LeftExpr, Expr RightExpr>
   requires(!ten::is_scalar_v<std::remove_cvref_t<LeftExpr>> &&
            !ten::is_scalar_v<std::remove_cvref<RightExpr>>)
auto operator+(LeftExpr &&left, RightExpr &&right) {
   using L = std::remove_cvref_t<LeftExpr>;
   using R = std::remove_cvref_t<RightExpr>;
   using left_input = ::ten::details::output_type<L>::type;
   using right_input = ::ten::details::output_type<R>::type;
   using output_type = ::ten::details::common_type_t<left_input, right_input>;

   return ::ten::binary_expr<
       L, R, output_type,
       ::ten::functional::binary_func<::ten::binary_operation::add>::func>(
       left, right);
}

template <typename T, Expr E>
   requires(::std::is_floating_point_v<std::remove_cvref_t<T>> ||
            ten::is_complex<std::remove_cvref_t<T>>::value ||
            std::is_integral_v<std::remove_cvref_t<T>>)
auto operator+(T &&scalar, E &&expr) {
   using R = std::remove_cvref_t<E>;
   using L = ::ten::scalar<std::remove_cvref_t<T>>;
   using left_input = ::ten::details::output_type<L>::type;
   using right_input = ::ten::details::output_type<R>::type;
   using output_type = ::ten::details::common_type_t<left_input, right_input>;
   auto s = L(scalar);
   return ::ten::binary_expr<L, R, output_type,
                             ::ten::functional::scalar_left_binary_func<
                                 ::ten::binary_operation::add>::func>(s, expr);
}

template <Expr E, typename T>
   requires(::std::is_floating_point_v<std::remove_cvref_t<T>> ||
            ten::is_complex<std::remove_cvref_t<T>>::value ||
            std::is_integral_v<std::remove_cvref_t<T>>)
auto operator+(E &&expr, T &&scalar) {
   using L = std::remove_cvref_t<E>;
   using R = ::ten::scalar<std::remove_cvref_t<T>>;
   using left_input = ::ten::details::output_type<L>::type;
   using right_input = ::ten::details::output_type<R>::type;
   using output_type = ::ten::details::common_type_t<left_input, right_input>;
   auto s = R(scalar);
   return ::ten::binary_expr<L, R, output_type,
                             ::ten::functional::scalar_right_binary_func<
                                 ::ten::binary_operation::add>::func>(expr, s);
}

// Add an expression an a ten::scalar
template <typename T, Expr E>
   requires(::ten::is_float<T>::value || ::ten::is_double<T>::value ||
::ten::is_complex<T>::value ||
            ::std::is_integral_v<T>)
auto operator+(ten::scalar<T> &scalar, E &&expr) {
   using R = std::remove_cvref_t<E>;
   using L = ten::scalar<T>;
   using left_input = ::ten::details::output_type<L>::type;
   using right_input = ::ten::details::output_type<R>::type;
   using output_type = ::ten::details::common_type_t<left_input, right_input>;
   return ::ten::binary_expr<L, R, output_type,
                             ::ten::functional::scalar_left_binary_func<
                                 ::ten::binary_operation::add>::func>(scalar,
                                                                      expr);
}

template <typename T, Expr E>
   requires(::ten::is_float<T>::value || ::ten::is_double<T>::value ||
::ten::is_complex<T>::value ||
            ::std::is_integral_v<T>)
auto operator+(ten::scalar<T> &&scalar, E &&expr) {
   using R = std::remove_cvref_t<E>;
   using L = ten::scalar<T>;
   using left_input = ::ten::details::output_type<L>::type;
   using right_input = ::ten::details::output_type<R>::type;
   using output_type = ::ten::details::common_type_t<left_input, right_input>;
   return ::ten::binary_expr<L, R, output_type,
                             ::ten::functional::scalar_left_binary_func<
                                 ::ten::binary_operation::add>::func>(scalar,
                                                                      expr);
}

template <typename T, Expr E>
   requires(::ten::is_float<T>::value || ::ten::is_complex<T>::value ||
            ::std::is_integral_v<T>)
auto operator+(E &&expr, ten::scalar<T> &scalar) {
   using L = std::remove_cvref_t<E>;
   using R = ten::scalar<T>;
   using left_input = ::ten::details::output_type<L>::type;
   using right_input = ::ten::details::output_type<R>::type;
   using output_type = ::ten::details::common_type_t<left_input, right_input>;
   return ::ten::binary_expr<L, R, output_type,
                             ::ten::functional::scalar_right_binary_func<
                                 ::ten::binary_operation::add>::func>(expr,
                                                                      scalar);
}

template <typename T, Expr E>
   requires(::ten::is_float<T>::value || ::ten::is_complex<T>::value ||
            ::std::is_integral_v<T>)
auto operator+(E &&expr, ten::scalar<T> &&scalar) {
   using L = std::remove_cvref_t<E>;
   using R = ten::scalar<T>;
   using left_input = ::ten::details::output_type<L>::type;
   using right_input = ::ten::details::output_type<R>::type;
   using output_type = ::ten::details::common_type_t<left_input, right_input>;
   return ::ten::binary_expr<L, R, output_type,
                             ::ten::functional::scalar_right_binary_func<
                                 ::ten::binary_operation::add>::func>(expr,
                                                                      scalar);
}

// Substract two expressions
template <Expr LeftExpr, Expr RightExpr>
   requires(!::ten::is_scalar_v<std::remove_cvref_t<LeftExpr>> &&
            !::ten::is_scalar_v<std::remove_cvref_t<RightExpr>>)
auto operator-(LeftExpr &&left, RightExpr &&right) {
   using L = std::remove_cvref_t<LeftExpr>;
   using R = std::remove_cvref_t<RightExpr>;
   using left_input = ::ten::details::output_type<L>::type;
   using right_input = ::ten::details::output_type<R>::type;
   using output_type = ::ten::details::common_type_t<left_input, right_input>;

   return ::ten::binary_expr<
       L, R, output_type,
       ::ten::functional::binary_func<::ten::binary_operation::sub>::func>(
       left, right);
}

template <typename T, Expr E>
   requires(::std::is_floating_point_v<std::remove_cvref_t<T>> ||
            ten::is_complex<std::remove_cvref_t<T>>::value ||
            std::is_integral_v<std::remove_cvref_t<T>>)
auto operator-(T &&scalar, E &&expr) {
   using R = std::remove_cvref_t<E>;
   using L = ::ten::scalar<std::remove_cvref_t<T>>;
   using left_input = ::ten::details::output_type<L>::type;
   using right_input = ::ten::details::output_type<R>::type;
   using output_type = ::ten::details::common_type_t<left_input, right_input>;
   auto s = L(scalar);
   return ::ten::binary_expr<L, R, output_type,
                             ::ten::functional::scalar_left_binary_func<
                                 ::ten::binary_operation::sub>::func>(s, expr);
}

template <Expr E, typename T>
   requires(::std::is_floating_point_v<std::remove_cvref_t<T>> ||
            ten::is_complex<std::remove_cvref_t<T>>::value ||
            std::is_integral_v<std::remove_cvref_t<T>>)
auto operator-(E &&expr, T &&scalar) {
   using L = std::remove_cvref_t<E>;
   using R = ::ten::scalar<std::remove_cvref_t<T>>;
   using left_input = ::ten::details::output_type<L>::type;
   using right_input = ::ten::details::output_type<R>::type;
   using output_type = ::ten::details::common_type_t<left_input, right_input>;
   auto s = R(scalar);
   return ::ten::binary_expr<L, R, output_type,
                             ::ten::functional::scalar_right_binary_func<
                                 ::ten::binary_operation::sub>::func>(expr, s);
}

// Substract an expression an a ten::scalar
template <typename T, Expr E>
   requires(::ten::is_float<T>::value || ::ten::is_double<T>::value ||
::ten::is_complex<T>::value ||
            ::std::is_integral_v<T>)
auto operator-(ten::scalar<T> &scalar, E &&expr) {
   using R = std::remove_cvref_t<E>;
   using L = ten::scalar<T>;
   using left_input = ::ten::details::output_type<L>::type;
   using right_input = ::ten::details::output_type<R>::type;
   using output_type = ::ten::details::common_type_t<left_input, right_input>;
   return ::ten::binary_expr<L, R, output_type,
                             ::ten::functional::scalar_left_binary_func<
                                 ::ten::binary_operation::sub>::func>(scalar,
                                                                      expr);
}

template <typename T, Expr E>
   requires(::ten::is_float<T>::value || ::ten::is_double<T>::value ||
::ten::is_complex<T>::value ||
            ::std::is_integral_v<T>)
auto operator-(ten::scalar<T> &&scalar, E &&expr) {
   using R = std::remove_cvref_t<E>;
   using L = ten::scalar<T>;
   using left_input = ::ten::details::output_type<L>::type;
   using right_input = ::ten::details::output_type<R>::type;
   using output_type = ::ten::details::common_type_t<left_input, right_input>;
   return ::ten::binary_expr<L, R, output_type,
                             ::ten::functional::scalar_left_binary_func<
                                 ::ten::binary_operation::sub>::func>(scalar,
                                                                      expr);
}

template <typename T, Expr E>
   requires(::ten::is_float<T>::value || ::ten::is_double<T>::value ||
::ten::is_complex<T>::value ||
            ::std::is_integral_v<T>)
auto operator-(E &&expr, ten::scalar<T> &scalar) {
   using L = std::remove_cvref_t<E>;
   using R = ten::scalar<T>;
   using left_input = ::ten::details::output_type<L>::type;
   using right_input = ::ten::details::output_type<R>::type;
   using output_type = ::ten::details::common_type_t<left_input, right_input>;
   return ::ten::binary_expr<L, R, output_type,
                             ::ten::functional::scalar_right_binary_func<
                                 ::ten::binary_operation::sub>::func>(expr,
                                                                      scalar);
}

template <typename T, Expr E>
   requires(::ten::is_float<T>::value || ::ten::is_double<T>::value ||
::ten::is_complex<T>::value ||
            ::std::is_integral_v<T>)
auto operator-(E &&expr, ten::scalar<T> &&scalar) {
   using L = std::remove_cvref_t<E>;
   using R = ten::scalar<T>;
   using left_input = ::ten::details::output_type<L>::type;
   using right_input = ::ten::details::output_type<R>::type;
   using output_type = ::ten::details::common_type_t<left_input, right_input>;
   return ::ten::binary_expr<L, R, output_type,
                             ::ten::functional::scalar_right_binary_func<
                                 ::ten::binary_operation::sub>::func>(expr,
                                                                      scalar);
}

// Multiply two expressions
template <Expr LeftExpr, Expr RightExpr>
   requires(!::ten::is_scalar_v<std::remove_cvref_t<LeftExpr>> &&
            !::ten::is_scalar_v<std::remove_cvref_t<RightExpr>>)
auto operator*(LeftExpr &&left, RightExpr &&right) {
   using L = std::remove_cvref_t<LeftExpr>;
   using R = std::remove_cvref_t<RightExpr>;
   using left_input = ::ten::details::output_type<L>::type;
   using right_input = ::ten::details::output_type<R>::type;

   using output_type = ::ten::details::mul_result_t<left_input, right_input>;

   return ::ten::binary_expr<L, R, output_type, ::ten::functional::mul>(left,
                                                                        right);
}

template <typename T, Expr E>
   requires(::std::is_floating_point_v<std::remove_cvref_t<T>> ||
            ten::is_complex<std::remove_cvref_t<T>>::value ||
            std::is_integral_v<std::remove_cvref_t<T>>)
auto operator*(T &&scalar, E &&expr) {
   using R = std::remove_cvref_t<E>;
   using L = ::ten::scalar<std::remove_cvref_t<T>>;
   using left_input = ::ten::details::output_type<L>::type;
   using right_input = ::ten::details::output_type<R>::type;
   using output_type = ::ten::details::common_type_t<left_input, right_input>;
   auto s = L(scalar);
   return ::ten::binary_expr<L, R, output_type,
                             ::ten::functional::scalar_left_binary_func<
                                 ::ten::binary_operation::mul>::func>(s, expr);
}
template <Expr E, typename T>
   requires(::std::is_floating_point_v<std::remove_cvref_t<T>> ||
            ten::is_complex<std::remove_cvref_t<T>>::value ||
            std::is_integral_v<std::remove_cvref_t<T>>)
auto operator*(E &&expr, T &&scalar) {
   using L = std::remove_cvref_t<E>;
   using R = ::ten::scalar<std::remove_cvref_t<T>>;
   using left_input = ::ten::details::output_type<L>::type;
   using right_input = ::ten::details::output_type<R>::type;
   using output_type = ::ten::details::common_type_t<left_input, right_input>;
   auto s = R(scalar);
   return ::ten::binary_expr<L, R, output_type,
                             ::ten::functional::scalar_right_binary_func<
                                 ::ten::binary_operation::mul>::func>(expr, s);
}

// Multiplyan expression an a ten::scalar
template <typename T, Expr E>
   requires(::ten::is_float<T>::value || ::ten::is_double<T>::value ||
::ten::is_complex<T>::value ||
            ::std::is_integral_v<T>)
auto operator*(ten::scalar<T> &scalar, E &&expr) {
   using R = std::remove_cvref_t<E>;
   using L = ten::scalar<T>;
   using left_input = ::ten::details::output_type<L>::type;
   using right_input = ::ten::details::output_type<R>::type;
   using output_type = ::ten::details::common_type_t<left_input, right_input>;
   return ::ten::binary_expr<L, R, output_type,
                             ::ten::functional::scalar_left_binary_func<
                                 ::ten::binary_operation::mul>::func>(scalar,
                                                                      expr);
}

template <typename T, Expr E>
   requires(::ten::is_float<T>::value || ::ten::is_double<T>::value ||
::ten::is_complex<T>::value ||
            ::std::is_integral_v<T>)
auto operator*(ten::scalar<T> &&scalar, E &&expr) {
   using R = std::remove_cvref_t<E>;
   using L = ten::scalar<T>;
   using left_input = ::ten::details::output_type<L>::type;
   using right_input = ::ten::details::output_type<R>::type;
   using output_type = ::ten::details::common_type_t<left_input, right_input>;
   return ::ten::binary_expr<L, R, output_type,
                             ::ten::functional::scalar_left_binary_func<
                                 ::ten::binary_operation::mul>::func>(scalar,
                                                                      expr);
}

template <typename T, Expr E>
   requires(::ten::is_float<T>::value || ::ten::is_double<T>::value ||
::ten::is_complex<T>::value ||
            ::std::is_integral_v<T>)
auto operator*(E &&expr, ten::scalar<T> &scalar) {
   using L = std::remove_cvref_t<E>;
   using R = ten::scalar<T>;
   using left_input = ::ten::details::output_type<L>::type;
   using right_input = ::ten::details::output_type<R>::type;
   using output_type = ::ten::details::common_type_t<left_input, right_input>;
   return ::ten::binary_expr<L, R, output_type,
                             ::ten::functional::scalar_right_binary_func<
                                 ::ten::binary_operation::mul>::func>(expr,
                                                                      scalar);
}

template <typename T, Expr E>
   requires(::ten::is_float<T>::value || ten::is_double<T>::value ||
::ten::is_complex<T>::value ||
            ::std::is_integral_v<T>)
auto operator*(E &&expr, ten::scalar<T> &&scalar) {
   using L = std::remove_cvref_t<E>;
   using R = ten::scalar<T>;
   using left_input = ::ten::details::output_type<L>::type;
   using right_input = ::ten::details::output_type<R>::type;
   using output_type = ::ten::details::common_type_t<left_input, right_input>;
   return ::ten::binary_expr<L, R, output_type,
                             ::ten::functional::scalar_right_binary_func<
                                 ::ten::binary_operation::mul>::func>(expr,
                                                                      scalar);
}

// Divide two expressions
template <Expr LeftExpr, Expr RightExpr>
   requires(!::ten::is_scalar_v<std::remove_cvref_t<LeftExpr>> &&
            !::ten::is_scalar_v<std::remove_cvref_t<RightExpr>>)
auto operator/(LeftExpr &&left, RightExpr &&right) {
   using L = std::remove_cvref_t<LeftExpr>;
   using R = std::remove_cvref_t<RightExpr>;
   using left_input = ::ten::details::output_type<L>::type;
   using right_input = ::ten::details::output_type<R>::type;
   using output_type = ::ten::details::common_type_t<left_input, right_input>;

   return ::ten::binary_expr<
       L, R, output_type,
       ::ten::functional::binary_func<::ten::binary_operation::div>::func>(
       left, right);
}

template <typename T, Expr E>
   requires(::std::is_floating_point_v<std::remove_cvref_t<T>> ||
            ten::is_complex<std::remove_cvref_t<T>>::value ||
            std::is_integral_v<std::remove_cvref_t<T>>)
auto operator/(T &&scalar, E &&expr) {
   using R = std::remove_cvref_t<E>;
   using L = ::ten::scalar<std::remove_cvref_t<T>>;
   using left_input = ::ten::details::output_type<L>::type;
   using right_input = ::ten::details::output_type<R>::type;
   using output_type = ::ten::details::common_type_t<left_input, right_input>;
   auto s = L(scalar);
   return ::ten::binary_expr<L, R, output_type,
                             ::ten::functional::scalar_left_binary_func<
                                 ::ten::binary_operation::div>::func>(s, expr);
}
template <Expr E, typename T>
   requires(::std::is_floating_point_v<std::remove_cvref_t<T>> ||
            ten::is_complex<std::remove_cvref_t<T>>::value ||
            std::is_integral_v<std::remove_cvref_t<T>>)
auto operator/(E &&expr, T &&scalar) {
   using L = std::remove_cvref_t<E>;
   using R = ::ten::scalar<std::remove_cvref_t<T>>;
   using left_input = ::ten::details::output_type<L>::type;
   using right_input = ::ten::details::output_type<R>::type;
   using output_type = ::ten::details::common_type_t<left_input, right_input>;
   auto s = R(scalar);
   return ::ten::binary_expr<L, R, output_type,
                             ::ten::functional::scalar_right_binary_func<
                                 ::ten::binary_operation::div>::func>(expr, s);
}

// Divide an expression an a ten::scalar
template <typename T, Expr E>
   requires(::ten::is_float<T>::value || ::ten::is_double<T>::value ||
::ten::is_complex<T>::value ||
            ::std::is_integral_v<T>)
auto operator/(ten::scalar<T> &scalar, E &&expr) {
   using R = std::remove_cvref_t<E>;
   using L = ten::scalar<T>;
   using left_input = ::ten::details::output_type<L>::type;
   using right_input = ::ten::details::output_type<R>::type;
   using output_type = ::ten::details::common_type_t<left_input, right_input>;
   return ::ten::binary_expr<L, R, output_type,
                             ::ten::functional::scalar_left_binary_func<
                                 ::ten::binary_operation::div>::func>(scalar,
                                                                      expr);
}

template <typename T, Expr E>
   requires(::ten::is_float<T>::value || ::ten::is_double<T>::value ||
::ten::is_complex<T>::value ||
            ::std::is_integral_v<T>)
auto operator/(ten::scalar<T> &&scalar, E &&expr) {
   using R = std::remove_cvref_t<E>;
   using L = ten::scalar<T>;
   using left_input = ::ten::details::output_type<L>::type;
   using right_input = ::ten::details::output_type<R>::type;
   using output_type = ::ten::details::common_type_t<left_input, right_input>;
   return ::ten::binary_expr<L, R, output_type,
                             ::ten::functional::scalar_left_binary_func<
                                 ::ten::binary_operation::div>::func>(scalar,
                                                                      expr);
}

template <typename T, Expr E>
   requires(::ten::is_float<T>::value || ::ten::is_double<T>::value ||
::ten::is_complex<T>::value ||
            ::std::is_integral_v<T>)
auto operator/(E &&expr, ten::scalar<T> &scalar) {
   using L = std::remove_cvref_t<E>;
   using R = ten::scalar<T>;
   using left_input = ::ten::details::output_type<L>::type;
   using right_input = ::ten::details::output_type<R>::type;
   using output_type = ::ten::details::common_type_t<left_input, right_input>;
   return ::ten::binary_expr<L, R, output_type,
                             ::ten::functional::scalar_right_binary_func<
                                 ::ten::binary_operation::div>::func>(expr,
                                                                      scalar);
}

template <typename T, Expr E>
   requires(::ten::is_float<T>::value || ::ten::is_double<T>::value ||
::ten::is_complex<T>::value ||
            ::std::is_integral_v<T>)
auto operator/(E &&expr, ten::scalar<T> &&scalar) {
   using L = std::remove_cvref_t<E>;
   using R = ten::scalar<T>;
   using left_input = ::ten::details::output_type<L>::type;
   using right_input = ::ten::details::output_type<R>::type;
   using output_type = ::ten::details::common_type_t<left_input, right_input>;
   return ::ten::binary_expr<L, R, output_type,
                             ::ten::functional::scalar_right_binary_func<
                                 ::ten::binary_operation::div>::func>(expr,
                                                                      scalar);
}
*/

/// TODO AxpyOne
/// A += B
/*template <typename left_expr, typename right_expr>
   requires ::ten::is_expr<std::remove_cvref_t<left_expr>> &&
            ::ten::is_expr<std::remove_cvref_t<right_expr>>
void operator+=(left_expr &&left, right_expr &&right) {
   using L = std::remove_cvref_t<left_expr>;
   using R = std::remove_cvref_t<RightExpr>;
   return ::ten::binary_expr<
       typename L::node_type, typename R::node_type,
       ::ten::functional::Axpy>(
       left.node(), right.node());
}*/

/// \class scalar
/// Hold a single value of type T.
template <typename T> class scalar : public expr {
private:
  bool _requires_grad = false;
  std::shared_ptr<T> _value = nullptr;
  std::shared_ptr<T> _grad = nullptr;

public:
  // Maybe for serialization
  // scalar() : _value(std::make_shared<T>()), _requires_grad(false) {}

  scalar(std::shared_ptr<T> value, bool requires_grad = false)
      : _value(value), _requires_grad(requires_grad) {
    if (requires_grad) {
      _grad = std::make_shared<T>();
    }
  }

  scalar(bool requires_grad = false)
      : _requires_grad(requires_grad), _value(std::make_shared<T>()) {
    if (requires_grad) {
      _grad = std::make_shared<T>();
    }
  }

  scalar(const T &value, bool requires_grad = false)
      : _value(std::make_shared<T>(value)), _requires_grad(requires_grad) {
    if (requires_grad) {
      _grad = std::make_shared<T>();
    }
  }

  scalar(T &&value, bool requires_grad = false)
      : _value(std::make_shared<T>(std::move(value))),
        _requires_grad(requires_grad) {
    if (requires_grad) {
      _grad = std::make_shared<T>();
    }
  }

  /// Asignment from an expression
  template <class Expr>
    requires(::ten::is_unary_expr<std::remove_cvref_t<Expr>>::value ||
             ::ten::is_binary_expr<std::remove_cvref_t<Expr>>::value)
  scalar(Expr &&expr) {
    static_assert(::ten::is_scalar_v<typename Expr::output_type>,
                  "Output type must be a scalar");
    auto value = expr.eval();
    _requires_grad = value.requires_grad();
    _value = value.value_node();
    _grad = value.grad_node();
  }

  /// Get or set  the value
  [[nodiscard]] T &value() { return *_value.get(); }
  [[nodiscard]] const T &value() const { return *_value.get(); }

  // Get tje node to the value
  [[nodiscard]] std::shared_ptr<T> value_node() const { return _value; }

  // Get or set the gradient value
  T &grad_value() { return *_grad.get(); }
  const T &grad_value() const { return *_grad.get(); }

  // Returns the node to the gradient
  [[nodiscard]] std::shared_ptr<T> grad_node() { return _grad; }

  // Returns the gradient in a scalar
  scalar<T> grad() const { return scalar<T>(_grad, false); }

  // Returns whether the scalar requires gradient or not
  [[nodiscard]] bool requires_grad() const { return _requires_grad; }

  /// Allocate the gradient scalar
  void allocate_gradient() {
    if (!_grad) {
      _grad = std::make_shared<T>();
    }
    _requires_grad = true;
  }

  scalar &operator=(const T &value) {
    *_value.get() = value;
    return *this;
  }

  scalar &operator=(const scalar &s) {
    _value = s._value;
    _grad = s._grad;
    return *this;
  }

  [[nodiscard]] inline static constexpr size_type rank() { return 0; }
};

/// \class tensor_node
/// Tensor node
template <typename T> class tensor_node {
public:
  using storage_type = dense_storage<T>;

private:
  /// storage
  std::unique_ptr<storage_type> _storage = nullptr;

public:
  /// Construct a tensor_node from storage
  explicit tensor_node(std::unique_ptr<storage_type> &&st) noexcept
      : _storage(std::move(st)) {}

  /// Move constructor
  tensor_node(tensor_node &&) = default;

  /// Copy constructor
  tensor_node(const tensor_node &) = default;

  /// Assignment operator
  tensor_node &operator=(tensor_node &&) = default;

  // Assignement operator
  tensor_node &operator=(const tensor_node &node) {
    this->_storage = node._storage;
    return *this;
  }

  /// Get the size
  [[nodiscard]] size_type size() const { return _storage.get()->size(); }

  /// Get the data
  [[nodiscard]] T *data() { return _storage.get()->data(); }

  /// Get the data
  [[nodiscard]] const T *data() const { return _storage.get()->data(); }

  /// Get the storage
  [[nodiscard]] storage_type &storage() const { return *_storage.get(); }

  /// Overloading the [] operator
  [[nodiscard]] inline const T &operator[](size_type index) const noexcept {
    return (*_storage.get())[index];
  }

  /// Overloading the [] operator
  [[nodiscard]] inline T &operator[](size_type index) noexcept {
    return (*_storage.get())[index];
  }

  /*
  friend bool
  serialize(std::ostream &os,
     requires(::ten::is_tensor_node<TensorNode>::value)
  friend TensorNode deserialize(std::istream &os);*/
};

/// \class tensor
///
/// Tensor represented by a multidimentional array.
template <typename T>
class tensor final : public ::ten::expr, public ::ten::tensor_base {
public:
  using value_type = T;

  /// \typedef node_type
  /// Node type
  using node_type = tensor_node<T>;

  using storage_type = node_type::storage_type;

  /// Tensor view type
  using view_type = view<T>;

private:
  /// Has gradient
  bool _requires_grad = false;
  /// Storage format
  storage_format _format = storage_format::dense;
  /// Storage order
  storage_order _order = storage_order::col_major;
  /// Size
  std::size_t _size = 0;
  /// Shape
  std::vector<std::size_t> _shape = {};
  /// Stride
  std::vector<std::size_t> _stride = {};
  /// Shared pointer to the node
  std::shared_ptr<node_type> _node = nullptr;
  /// Gradient information
  std::shared_ptr<node_type> _grad = nullptr;

private:
  /// Returns the value at the indices
  [[nodiscard]] inline T &at(size_type index, auto... tail) noexcept {
    const size_type rank = _shape.size();
    constexpr size_type tail_size = sizeof...(tail);
    // tail_size must be equal to 0 or rank-1
    if constexpr (tail_size == 0) {
      return (*_node.get())[index];
    }
    std::vector<size_type> indices = {index, static_cast<size_type>(tail)...};
    size_type idx = details::linear_index(_stride, indices);
    return (*_node.get())[idx];
  }

  /// Returns the value at the indices
  [[nodiscard]] inline const T &at(size_type index,
                                   auto... tail) const noexcept {
    const size_type rank = _shape.size();
    constexpr size_type tail_size = sizeof...(tail);
    if constexpr (tail_size == 0) {
      return (*_node.get())[index];
    }
    std::vector<size_type> indices = {index, static_cast<size_type>(tail)...};
    size_type idx = details::linear_index(_stride, indices);
    return (*_node.get())[idx];
  }

  /// TODO Returns the gradient value at the indices
  /*
  [[nodiscard]] inline typename base_type::value_type &
  grad_linear(size_type index, auto... tail) noexcept {
     static constexpr size_type rank = Shape::rank();
     constexpr size_type tail_size = sizeof...(tail);
     static_assert(tail_size == 0 || tail_size == (rank - 1),
                   "Invalid number of indices.");
     if constexpr (tail_size == 0) {
        return (*_grad.get())[index];
     }
     std::array<size_type, Shape::rank()> indices{
         index, static_cast<size_type>(tail)...};
     if constexpr (Shape::is_dynamic()) {
        size_type idx = details::linear_index(_stride.value(), indices);
        return (*_grad.get())[idx];
     } else {
        size_type idx = details::static_linear_index<stride_type>(indices);
        return (*_grad.get())[idx];
     }
  }

  /// TODO Returns the gradient value at the indices
  [[nodiscard]] inline const typename base_type::value_type &
  grad_linear(size_type index, auto... tail) const noexcept {
     static constexpr size_type rank = Shape::rank();
     constexpr size_type tail_size = sizeof...(tail);
     static_assert(tail_size == 0 || tail_size == (rank - 1),
                   "Invalid number of indices.");
     if constexpr (tail_size == 0) {
        return (*_grad.get())[index];
     }
     std::array<size_type, Shape::rank()> indices{
         index, static_cast<size_type>(tail)...};
     if constexpr (Shape::is_dynamic()) {
        size_type idx = details::linear_index(_stride.value(), indices);
        return (*_grad.get())[idx];
     } else {
        size_type idx = details::static_linear_index<stride_type>(indices);
        return (*_grad.get())[idx];
     }
  }*/

private:
  /// Private default constructor
  tensor(const bool requires_grad, const ten::storage_order order) noexcept
      : _requires_grad(requires_grad), _node(nullptr),
        _format(ten::storage_format::dense), _order(order), _size(0),
        _shape({}), _stride({}) {}

public:
  /// Make a tensor with default constructor
  static tensor
  make_default(const bool requires_grad = false,
               const ten::storage_order order = ten::storage_order::col_major) {
    return tensor(requires_grad, order);
  }

  /// Construct a tensor from node, shape, format and storage order
  explicit tensor(const std::shared_ptr<node_type> &node,
                  const std::vector<std::size_t> &dims,
                  const ::ten::storage_format format,
                  const bool requires_grad = false,
                  const ::ten::storage_order order =
                      ::ten::storage_order::col_major) noexcept
      : _requires_grad(requires_grad), _format(format), _order(order),
        _size(ten::details::compute_size(dims)), _shape(dims),
        _stride(ten::details::compute_strides(dims, order)), _node(node) {
    if (requires_grad) {
      auto grad_storage = std::make_unique<storage_type>(_size);
      _grad = std::make_shared<node_type>(std::move(grad_storage));
    }
  }

  /// Construct a tensor from node, shape, format and storage order
  explicit tensor(std::shared_ptr<node_type> &&node,
                  const std::vector<std::size_t> &dims,
                  const ::ten::storage_format format,
                  const bool requires_grad = false,
                  const ::ten::storage_order order =
                      ::ten::storage_order::col_major) noexcept
      : _requires_grad(requires_grad), _format(format), _order(order),
        _size(ten::details::compute_size(dims)), _shape(dims),
        _stride(ten::details::compute_strides(dims, order)),
        _node(std::move(node)) {
    if (requires_grad) {
      auto grad_storage = std::make_unique<storage_type>(_size);
      _grad = std::make_shared<node_type>(std::move(grad_storage));
    }
  }

  /// Construct a tensor_node from a list of shape and storage order
  explicit tensor(std::initializer_list<size_type> &&dims,
                  const bool requires_grad = false,
                  const ::ten::storage_order order =
                      ::ten::storage_order::col_major) noexcept
      : _requires_grad(requires_grad), _format(::ten::storage_format::dense),
        _order(order), _shape(std::move(dims)),
        _stride(ten::details::compute_strides(_shape, _order)) {
    _size = ten::details::compute_size(_shape);
    auto storage = std::make_unique<storage_type>(_size);
    _node = std::make_shared<node_type>(std::move(storage));
    if (requires_grad) {
      auto grad_storage = std::make_unique<storage_type>(_size);
      _grad = std::make_shared<node_type>(std::move(grad_storage));
    }
  }

  /// Construct a tensor_node from a list of shape, format and storage order
  explicit tensor(std::initializer_list<size_type> &&dims,
                  ten::storage_format format, const bool requires_grad = false,
                  const ::ten::storage_order order =
                      ::ten::storage_order::col_major) noexcept
      : _requires_grad(requires_grad), _format(format), _order(order),
        _shape(std::move(dims)),
        _stride(ten::details::compute_strides(_shape, _order)) {
    _size = ten::details::compute_size(_shape);
    auto storage = std::make_unique<storage_type>(_size);
    _node = std::make_shared<node_type>(std::move(storage));
    if (requires_grad) {
      auto grad_storage = std::make_unique<storage_type>(_size);
      _grad = std::make_shared<node_type>(std::move(grad_storage));
    }
  }

  /// Tensor node from shape and initialize with data
  explicit tensor(std::initializer_list<size_type> &&dims,
                  std::initializer_list<T> &&data,
                  const bool requires_grad = false,
                  const ::ten::storage_order order =
                      ::ten::storage_order::col_major) noexcept
      : _requires_grad(requires_grad), _format(ten::storage_format::dense),
        _order(order), _shape(std::move(dims)),
        _stride(ten::details::compute_strides(_shape, _order)) {
    _size = ten::details::compute_size(_shape);
    auto storage = std::make_unique<storage_type>(_size);
    size_type i = 0;
    for (auto x : data) {
      (*storage.get())[i] = x;
      i++;
    }
    _node = std::make_shared<node_type>(std::move(storage));
    if (requires_grad) {
      auto grad_storage = std::make_unique<storage_type>(_size);
      _grad = std::make_shared<node_type>(std::move(grad_storage));
    }
  }

  /// Construct a tensor from the shape
  explicit tensor(const std::vector<std::size_t> &dims,
                  const bool requires_grad = false,
                  const ::ten::storage_order order =
                      ::ten::storage_order::col_major) noexcept
      : _requires_grad(requires_grad), _format(ten::storage_format::dense),
        _order(order), _size(ten::details::compute_size(dims)), _shape(dims),
        _stride(ten::details::compute_strides(_shape, _order)) {
    auto storage = std::make_unique<storage_type>(_size);
    _node = std::make_shared<node_type>(std::move(storage));
    if (requires_grad) {
      auto grad_storage = std::make_unique<storage_type>(_size);
      _grad = std::make_shared<node_type>(std::move(grad_storage));
    }
  }

  /// Construct a tensor_node from a shape, format and storage order
  explicit tensor(const std::vector<std::size_t> &dims,
                  const ten::storage_format format,
                  const bool requires_grad = false,
                  const ::ten::storage_order order =
                      ::ten::storage_order::col_major) noexcept
      : _requires_grad(requires_grad), _format(format), _order(order),
        _shape(dims), _stride(ten::details::compute_strides(_shape, _order)) {
    _size = ten::details::compute_size(_shape);
    auto storage = std::make_unique<storage_type>(_size);
    _node = std::make_shared<node_type>(std::move(storage));
    if (requires_grad) {
      auto grad_storage = std::make_unique<storage_type>(_size);
      _grad = std::make_shared<node_type>(std::move(grad_storage));
    }
  }

  /// Tensor from shape and std::initializer_list data
  explicit tensor(const std::vector<std::size_t> &dims,
                  std::initializer_list<T> &&data,
                  const bool requires_grad = false,
                  const ::ten::storage_order order =
                      ::ten::storage_order::col_major) noexcept
      : _requires_grad(requires_grad), _format(ten::storage_format::dense),
        _order(order), _size(ten::details::compute_size(dims)), _shape(dims),
        _stride(ten::details::compute_strides(_shape, _order)) {
    auto storage = std::make_unique<storage_type>(_size);
    size_type i = 0;
    for (auto x : data) {
      (*storage.get())[i] = x;
      i++;
    }
    _node = std::make_shared<node_type>(std::move(storage));
    if (requires_grad) {
      auto grad_storage = std::make_unique<storage_type>(_size);
      _grad = std::make_shared<node_type>(std::move(grad_storage));
    }
  }

  /// Tensor node from shape and std::vector data
  explicit tensor(const std::vector<std::size_t> &dims, std::vector<T> &&data,
                  const bool requires_grad = false,
                  const ::ten::storage_order order =
                      ::ten::storage_order::col_major) noexcept
      : _requires_grad(requires_grad), _format(ten::storage_format::dense),
        _order(order), _size(ten::details::compute_size(dims)), _shape(dims),
        _stride(ten::details::compute_strides(dims, _order)) {
    auto storage = std::make_unique<storage_type>(_size);
    size_type i = 0;
    for (auto x : data) {
      (*storage.get())[i] = x;
      i++;
    }
    _node = std::make_shared<node_type>(std::move(storage));
    if (requires_grad) {
      auto grad_storage = std::make_unique<storage_type>(_size);
      _grad = std::make_shared<node_type>(std::move(grad_storage));
    }
  }

  /// Copy asignment from an expression
  template <class ExprType>
    requires(::ten::is_unary_expr_v<std::remove_cvref_t<ExprType>> ||
             ::ten::is_binary_expr_v<std::remove_cvref_t<ExprType>>)
  tensor(ExprType &&expr) noexcept {
    using expr_type = std::remove_cvref_t<ExprType>;
    using evaluated_type = typename expr_type::output_type;

    static_assert(::ten::is_tensor<evaluated_type>::value,
                  "Evaluated type must be a tensor.");
    auto value = expr.eval();
    _format = value.format();
    _order = value.storage_order();
    _size = value.size();
    _shape = value.shape();
    _stride = value.strides();
    _node = value.node();
    _grad = value.grad_node();
  }

  /// Asignment from an expression
  template <class ExprType>
    requires(::ten::is_unary_expr_v<std::remove_cvref_t<ExprType>> ||
             ::ten::is_binary_expr_v<std::remove_cvref_t<ExprType>>)
  tensor &operator=(ExprType &&expr) noexcept {
    using expr_type = std::remove_cvref_t<ExprType>;
    using evaluated_type = typename expr_type::output_type;

    static_assert(::ten::is_tensor<evaluated_type>::value,
                  "Evaluated type must be a tensor.");

    // TODO Match and fuse expression
    // bool matched_fused = ::ten::match_fuse(expr, *this);
    auto value = expr.eval();
    _format = value.format();
    _order = value.storage_order();
    _size = value.size();
    _shape = value.shape();
    _stride = value.strides();
    _node = value.node();
    _grad = value.grad_node();
    return *this;
  }

  // Assign a value of type T
  tensor &operator=(T value) noexcept {
    for (size_t i = 0; i < _size; i++) {
      (*_node.get())[i] = value;
    }
    return *this;
  }

  /// Copy constructor
  tensor(const tensor &t) {
    _requires_grad = t._requires_grad;
    _format = t._format;
    _order = t._order;
    _size = t._size;
    _shape = t._shape;
    _stride = t._stride;
    _node = t._node;
    _grad = t._grad;
  }

  /// Copy constructor
  tensor(tensor &&t) {
    _requires_grad = std::move(t._requires_grad);
    _format = std::move(t._format);
    _order = std::move(t._order);
    _size = std::move(t._size);
    _shape = std::move(t._shape);
    _stride = std::move(t._stride);
    _node = std::move(t._node);
    _grad = std::move(t._grad);
  }

  /// Assignment operator
  tensor &operator=(const tensor &t) {
    _requires_grad = t._requires_grad;
    _format = t._format;
    _order = t._order;
    _size = t._size;
    _shape = t._shape;
    _stride = t._stride;
    _node = t._node;
    _grad = t._grad;
    return *this;
  }

  /// Assignment operator
  tensor &operator=(tensor &&t) {
    _requires_grad = std::move(t._requires_grad);
    _format = std::move(t._format);
    _order = std::move(t._order);
    _size = std::move(t._size);
    _shape = std::move(t._shape);
    _stride = std::move(t._stride);
    _node = std::move(t._node);
    _grad = std::move(t._grad);
    return *this;
  }

  /// Returns the storage order
  [[nodiscard]] inline storage_order storage_order() const { return _order; }

  //// Returns the rank
  [[nodiscard]] inline std::size_t rank() const { return _shape.size(); }

  /// Get the dimension at index
  [[nodiscard]] std::size_t dim(std::size_t index) const {
    return _shape[index];
  }

  /// Returns the shape
  [[nodiscard]] inline const std::vector<std::size_t> &shape() const {
    return _shape;
  }

  /// Returns the strides
  [[nodiscard]] inline const std::vector<std::size_t> &strides() const {
    return _stride;
  }

  /// Returns the size (number of elements of the tensor)
  [[nodiscard]] inline std::size_t size() const { return _size; }

  /// Get the storage format
  [[nodiscard]] storage_format format() const { return _format; }

  /// Return whether the tensor is transposed
  [[nodiscard]] bool is_transposed() const {
    return _format & ::ten::storage_format::transposed;
  }

  /// Return whether the tensor is symmetric
  [[nodiscard]] bool is_symmetric() const {
    return _format & ::ten::storage_format::symmetric;
  }

  // Returns whether the tensor is conjugated
  [[nodiscard]] bool is_conj() const {
    return _format & ::ten::storage_format::conj;
  }

  /// Return whether the tensor is hermitian
  [[nodiscard]] bool is_hermitian() const {
    return _format & ::ten::storage_format::hermitian;
  }

  /// Return whether the tensor is diagonal matrix
  [[nodiscard]] constexpr bool is_diagonal() { return false; }

  /// Return whether the tensor is lower triangular
  [[nodiscard]] bool is_lower_tr() const {
    return _format == ::ten::storage_format::lower_tr;
  }

  /// Return whether the tensor is upper triangular
  [[nodiscard]] bool is_upper_tr() const {
    return _format == ::ten::storage_format::upper_tr;
  }

  // Returns the shared ptr to the node
  [[nodiscard]] std::shared_ptr<node_type> node() const { return _node; }
  /// Get the data
  [[nodiscard]] const T *data() const { return _node.get()->data(); }

  /// Get the data
  [[nodiscard]] T *data() { return _node.get()->data(); }

  /// Returns the storage
  [[nodiscard]] storage_type &storage() const { return _node.get()->storage(); }

  /// Overloading the [] operator
  [[nodiscard]] inline const T &operator[](std::size_t index) const noexcept {
    return (*_node.get())[index];
  }

  /// Overloading the [] operator
  [[nodiscard]] inline T &operator[](std::size_t index) noexcept {
    return (*_node.get())[index];
  }

  /// Overloading the () operator
  [[nodiscard]] inline const T &operator()(auto... index) const noexcept {
    return at(index...);
  }

  /// Overloading the () operator
  [[nodiscard]] inline T &operator()(auto... index) noexcept {
    return at(index...);
  }

  // Overload the [] operator for std::vector<std::size_t>
  [[nodiscard]] inline const T &
  operator[](const std::vector<std::size_t> &indices) const noexcept {
    size_t index = ten::details::linear_index(_stride, indices);
    return (*_node.get())[index];
  }

  // Overload the [] operator for std::vector<std::size_t>
  [[nodiscard]] inline T &
  operator[](const std::vector<std::size_t> &indices) noexcept {
    size_t index = ten::details::linear_index(_stride, indices);
    return (*_node.get())[index];
  }

  // TODO Overload the [] operator for mdseq
  /*
 [[nodiscard]] inline tensor_view_type
 operator[](mdseq<Shape::rank()> sequences) const noexcept {
   if constexpr (Shape::is_dynamic()) {
     for (size_t i = 0; i < Shape::rank(); i++) {
       if (sequences._end[i] == 0) {
         sequences._end[i] = _shape.value().dim(i);
       }
     }
   } else {
     for (size_t i = 0; i < Shape::rank(); i++) {
       if (sequences._end[i] == 0) {
         sequences._end[i] = Shape::static_dim(i);
       }
     }
   }
   return tensor_view_type(*this, sequences._start, sequences._end);
 }*/

  // Overload the ()  operator for seq
  [[nodiscard]] inline view<T> operator()(auto... index) noexcept
    requires(std::is_same_v<decltype(index...[0]), seq>)
  {
    std::size_t r = sizeof...(index);
    if (r != rank()) {
      std::cerr << "operator () requires" << rank() << "arguments.";
    }
    // TODO Check if its a valid sequence
    std::vector<std::size_t> starts(r);
    std::vector<std::size_t> ends(r);
    std::vector<seq> sequences{index...};
    for (size_t i = 0; i < r; i++) {
      starts[i] = sequences[i]._start;
      if (sequences[i]._end == 0) {
        ends[i] = dim(i);
      } else {
        ends[i] = sequences[i]._end;
      }
    }
    return view<T>(*this, starts, ends);
  }

  // TODO Get the column
  /*
 [[nodiscard]] auto col(const size_t index) const -> decltype(auto)
   requires(Shape::rank() == 2 && Shape::is_dynamic())
 {
   return ranked_column<T, Shape, order, Storage, Allocator>(
       index, _shape.value(), _node);
 }*/

  // TODO Get the row
  /*
 [[nodiscard]] auto row(const size_t index) -> decltype(auto)
   requires(Shape::rank() == 2 && Shape::is_dynamic())
 {
   return ranked_row<T, Shape, order, Storage, Allocator>(
       index, _shape.value(), _node);
 }*/

  // Returns the shared ptr to the gradient node
  [[nodiscard]] std::shared_ptr<node_type> grad_node() const { return _grad; }

  // Return the gradient tensor
  [[nodiscard]] tensor<T> grad() const {
    return tensor(_grad, _shape, _format, false, _order);
  }

  /// TODO Get the gradient data
  // [[nodiscard]] const T *grad_data() const { return _grad.get()->data(); }

  /// TODO Get the gradient data
  // [[nodiscard]] T *grad_data() { return _grad.get()->data(); }

  /// TODO MAYBE Returns the gradient storage
  ///[[nodiscard]] Storage &grad_storage() const { return
  ///_grad.get()->storage(); }

  /// Get the gradient at index
  [[nodiscard]] inline const T &grad_value(size_type index) const noexcept {
    return (*_grad.get())[index];
  }

  /// Get the gradient at index
  [[nodiscard]] inline T &grad_value(size_type index) noexcept {
    return (*_grad.get())[index];
  }

  /// TODO Get the gradient at index...
  /*
 [[nodiscard]] inline const typename base_type::value_type &
 grad_at(auto... index) const noexcept {
   static_assert(sizeof...(index) == Shape::rank());
   return grad_linear(index...);
 }*/

  /// TODO Get the gradient at index...
  /*
 [[nodiscard]] inline typename base_type::value_type &
 grad_at(auto... index) noexcept {
   static_assert(sizeof...(index) == Shape::rank());
   return grad_linear(index...);
 }*/

  /// Return whether it has gradient information
  [[nodiscard]] inline bool requires_grad() const { return _requires_grad; }

  // TODO Allocate the gradient tensor
  /*
 void allocate_gradient() {
   if (!_grad) {
     if constexpr (Shape::is_static()) {
       auto grad_storage = std::make_unique<storage_type>();
       _grad = std::make_shared<node_type>(std::move(grad_storage));
     } else {
       auto grad_storage = std::make_unique<storage_type>(_shape.value());
       _grad = std::make_shared<node_type>(std::move(grad_storage));
     }
   }
   _requires_grad = true;
 }*/

  // TODO Zero the gradient
  /*
 void zero_grad() {
   if (!_grad) {
     for (size_t i = 0; i < size(); i++) {
       (*_grad.get())[i] = 0;
     }
   }
 }*/

  /// Copy the tensor
  auto copy(bool copy_grad = false) const {
    auto st = std::make_unique<storage_type>(_size);
    auto node = std::make_shared<node_type>(std::move(st));
    tensor t(std::move(node), _shape, _format, copy_grad, _order);
    // Copy the data to the new tensor
    for (size_t i = 0; i < _size; i++) {
      t[i] = (*_node.get())[i];
    }
    // Copy gradient
    if (copy_grad && _requires_grad) {
      for (size_t i = 0; i < _size; i++) {
        t.grad_value(i) = (*_grad.get())[i];
      }
    }
    return t;
  }

  /// TODO Returns whether the tensor is sparse
  /*[[nodiscard]] bool is_sparse() const {
    return this->is_sparse_coo() || this->is_sparse_csc() ||
           this->is_sparse_csr();
  }*/

  // Data type
  [[nodiscard]] inline ten::data_type data_type() noexcept {
    return to_data_type<T>();
  }

  // Overload == operator
  template <class Type>
  bool operator==(const tensor<Type> &right) const noexcept {
    // Template parameters must be the same
    if (!std::is_same_v<T, Type>) {
      return false;
    }
    // Same storage order
    if (_order != right.storage_order()) {
      return false;
    }
    // TODO Test if same storage
    if (_format != right.format()) {
      return false;
    }
    // They must have the same shape
    if (!ten::details::same_shape(_shape, right.shape())) {
      return false;
    }
    // They must have the same node
    if (_node != right.node()) {
      return false;
    }
    return true;
  }

  // Overload != operator
  template <class Type> bool operator!=(const tensor<Type> &right) const {
    return !operator==(right);
  }

  // Overload the > operator for dynamic tensors
  tensor<bool> operator>(const tensor &right) const noexcept {
    if (_order != right.storage_order()) {
      std::cerr
          << "Error, > operator, inputs must have the same storage order.\n";
    }
    if (!ten::details::same_shape(_shape, right.shape())) {
      std::cerr << "Error, > operator, inputs must have the same shape.\n";
    }
    tensor<bool> res(_shape, false, _order);
    for (size_t i = 0; i < _size; i++) {
      res[i] = (*this)[i] > right[i];
    }
    return res;
  }

  tensor<bool> gt(const tensor &right) const noexcept {
    return operator>(right);
  }

  // Overload the < operator for dynamic tensors
  tensor<bool> operator<(const tensor &right) const noexcept {
    if (_order != right.storage_order()) {
      std::cerr
          << "Error, < operator, inputs must have the same storage order.\n";
    }
    if (!ten::details::same_shape(_shape, right.shape())) {
      std::cerr << "Error, < operator, inputs must have the same shape.\n";
    }
    tensor<bool> res(_shape, false, _order);
    for (size_t i = 0; i < size(); i++) {
      res[i] = (*this)[i] < right[i];
    }
    return res;
  }

  tensor<bool> lt(const tensor &right) const noexcept {
    return operator<(right);
  }

  // Overload the >= operator
  tensor<bool> operator>=(const tensor &right) const noexcept {

    if (_order != right.storage_order()) {
      std::cerr
          << "Error, >= operator, inputs must have the same storage order.\n";
    }
    if (!ten::details::same_shape(_shape, right.shape())) {
      std::cerr << "Error, >= operator, inputs must have the same shape.\n";
    }
    tensor<bool> res(_shape, false, _order);
    for (size_t i = 0; i < size(); i++) {
      res[i] = (*this)[i] >= right[i];
    }
    return res;
  }

  tensor<bool> ge(const tensor &right) const noexcept {
    return operator>=(right);
  }

  // Overload the >= operator for dynamic tensors
  tensor<bool> operator<=(const tensor &right) const noexcept {
    if (_order != right.storage_order()) {
      std::cerr
          << "Error, <= operator, inputs must have the same storage order.\n";
    }
    if (!ten::details::same_shape(_shape, right.shape())) {
      std::cerr << "Error, <= operator, inputs must have the same shape.\n";
    }

    tensor<bool> res(_shape, false, _order);
    for (size_t i = 0; i < size(); i++) {
      res[i] = (*this)[i] <= right[i];
    }
    return res;
  }

  tensor<bool> le(const tensor &right) const noexcept {
    return operator<=(right);
  }

  // Overload the == or eq operator
  tensor<bool> eq(const tensor &right) const noexcept {
    if (_order != right.storage_order()) {
      std::cerr
          << "Error, eq operator, inputs must have the same storage order.\n";
    }
    if (!ten::details::same_shape(_shape, right.shape())) {
      std::cerr << "Error, eq operator, inputs must have the same shape.\n";
    }
    tensor<bool> res(_shape, false, _order);
    for (size_t i = 0; i < size(); i++) {
      res[i] = (*this)[i] == right[i];
    }
    return res;
  }

  // Serialize friend function
  /*template <class Type>
  friend bool serialize(std::ostream &os, tensor<Type> &t);
  */

  // Deserialize friend function
  /*template <class TensorType>
     requires(ten::is_tensor<TensorType>::value)
  friend TensorType deserialize(std::istream &is);
  */
};

/*
template <class T, class Shape, storage_order StorageOder, class Storage,
          class Allocator>
bool serialize(std::ostream &os,
               ranked_tensor<T, Shape, StorageOder, Storage, Allocator> &t) {
   return serialize(os, *t._node.get());
}

template <class TensorType>
   requires(ten::is_tensor<TensorType>::value)
TensorType deserialize(std::istream &is) {
   using NodeType = TensorType::node_type;
   auto node =
       std::shared_ptr<NodeType>(new NodeType(deserialize<NodeType>(is)));
   return TensorType(std::move(node));
}*/

////////////////////////////////////////////////////////////////////////////////
// Diagonal matrix

/// \class diagonal
/// Diagonal square matrix
template <typename T> class diagonal {
public:
  using value_type = T;
  using node_type = tensor_node<T>;
  using storage_type = node_type::storage_type;

private:
  /// Requires gradient
  bool _requires_grad = false;
  /// Storage order
  storage_order _order = storage_order::col_major;
  /// Size (here number of elemens in the diagonal)
  std::size_t _size = 0;
  /// shape
  std::vector<std::size_t> _shape = {};
  /// Strides
  std::vector<std::size_t> _stride = {};
  /// Node
  std::shared_ptr<node_type> _node = nullptr;
  /// Gradient
  std::shared_ptr<node_type> _grad = nullptr;

public:
  explicit diagonal(
      const std::shared_ptr<node_type> &node, const std::vector<size_t> &dims,
      const bool requires_grad = false,
      const ::ten::storage_order order = ::ten::storage_order::col_major)
      : _requires_grad(requires_grad), _order(order), _shape(shape),
        _stride(ten::details::compute_strides(_shape, _order)), _node(node) {
    if (dims.size() != 2) {
      std::cerr << "diagonal must be a matrix.\n";
    } else if (dims[0] != dims[1]) {
      std::cerr << "diagonal matrix must be square.\n";
    }
    _size = std::min(dims[0], dims[1]);
    if (requires_grad) {
      auto st = std::make_unique<storage_type>(_size);
      _grad = std::make_shared<node_type>(std::move(st));
    }
  }

  explicit diagonal(
      const std::vector<std::size_t> &dims, const bool requires_grad = false,
      const ::ten::storage_order order = ::ten::storage_order::col_major)
      : _requires_grad(requires_grad), _order(order), _shape(dims),
        _stride(ten::details::compute_strides(_shape, _order)) {
    if (_shape.size() != 2) {
      std::cerr << "diagonal must be a matrix.\n";
    } else if (_shape[0] != _shape[1]) {
      std::cerr << "diagonal matrix must be square.\n";
    }
    _size = std::min(_shape[0], _shape[1]);
    auto storage = std::make_unique<storage_type>(_size);
    _node = std::make_shared<node_type>(std::move(storage));
    if (requires_grad) {
      auto st = std::make_unique<storage_type>(_size);
      _grad = std::make_shared<node_type>(std::move(st));
    }
  }

  explicit diagonal(
      std::initializer_list<std::size_t> &&dims,
      const bool requires_grad = false,
      const ::ten::storage_order order = ::ten::storage_order::col_major)
      : _requires_grad(requires_grad), _order(order), _shape(std::move(dims)),
        _stride(ten::details::compute_strides(_shape, _order)) {
    if (_shape.size() != 2) {
      std::cerr << "diagonal must be a matrix.\n";
    } else if (_shape[0] != _shape[1]) {
      std::cerr << "diagonal matrix must be square.\n";
    }
    _size = std::min(_shape[0], _shape[1]);
    auto storage = std::make_unique<storage_type>(_size);
    _node = std::make_shared<node_type>(std::move(storage));
    if (requires_grad) {
      auto st = std::make_unique<storage_type>(_size);
      _grad = std::make_shared<node_type>(std::move(st));
    }
  }

private:
  [[nodiscard]] const T &at(std::size_t row, std::size_t col) const {
    if (row != col) {
      std::cerr << "Cannot acces diagonal matrix at index (" << row << ","
                << col << ")\n";
    }
    return (*_node.get())[row];
  }

  [[nodiscard]] T &at(std::size_t row, std::size_t col) {
    if (row != col) {
      std::cerr << "Cannot acces diagonal matrix at index (" << row << ","
                << col << ")\n";
    }
    return (*_node.get())[row];
  }

public:
  // Assign a value of type T
  diagonal &operator=(T value) noexcept {
    for (std::size_t i = 0; i < _size; i++) {
      (*_node.get())[i] = value;
    }
    return *this;
  }

  /// Copy constructor
  diagonal(const diagonal &m) {
    _requires_grad = m._requires_grad;
    _order = m._order;
    _size = m._size;
    _shape = m._shape;
    _stride = m._stride;
    _node = m._node;
    _grad = m._grad;
  }

  /// Copy constructor
  diagonal(diagonal &&m) {
    _requires_grad = std::move(m._requires_grad);
    _order = std::move(m._order);
    _size = std::move(m._size);
    _shape = std::move(m._shape);
    _stride = std::move(m._stride);
    _node = std::move(m._node);
    _grad = std::move(m._grad);
  }

  /// Assignment operator
  diagonal &operator=(const diagonal &m) {
    _requires_grad = m._requires_grad;
    _order = m._order;
    _size = m._size;
    _shape = m._shape;
    _stride = m._stride;
    _node = m._node;
    _grad = m._grad;
    return *this;
  }

  /// Assignment operator
  diagonal &operator=(diagonal &&m) {
    _requires_grad = std::move(m._requires_grad);
    _order = std::move(m._order);
    _size = std::move(m._size);
    _shape = std::move(m._shape);
    _stride = std::move(m._stride);
    _node = std::move(m._node);
    _grad = std::move(m._grad);
    return *this;
  }

  //// Returns the rank
  [[nodiscard]] inline size_type rank() const { return _shape.size(); }

  /// Returns the storage order
  [[nodiscard]] inline ten::storage_order storage_order() const {
    return _order;
  }

  /// Get the dimension at index
  [[nodiscard]] size_type dim(size_type index) const { return _shape[index]; }

  /// Returns the shape
  [[nodiscard]] inline const std::vector<std::size_t> &shape() const {
    return _shape;
  }

  /// Returns the strides
  [[nodiscard]] inline const std::vector<std::size_t> &strides() const {
    return _stride;
  }

  // Data type
  [[nodiscard]] inline ten::data_type data_type() noexcept {
    return to_data_type<T>();
  }

  [[nodiscard]] constexpr ::ten::storage_format format() {
    return ::ten::storage_format::diagonal;
  }

  [[nodiscard]] std::size_t size() const { return _size; }

  /// Return whether the tensor is diagonal
  [[nodiscard]] constexpr bool is_diagonal() { return true; }

  // Returns the shared ptr to the node
  [[nodiscard]] std::shared_ptr<node_type> node() const { return _node; }

  /// Get the pointer to the data
  [[nodiscard]] const T *data() const { return _node.get()->data(); }

  /// Get the mutable pointer to the data
  [[nodiscard]] T *data() { return _node.get()->data(); }

  /// Returns the storage
  [[nodiscard]] storage_type &storage() const { return _node.get()->storage(); }

  /// Overload the [] operator
  [[nodiscard]] const T &operator[](std::size_t index) const {
    return (*_node.get())[index];
  }
  [[nodiscard]] T &operator[](std::size_t index) {
    return (*_node.get())[index];
  }

  /// Overload the () operator
  [[nodiscard]] const T &operator()(std::size_t row, std::size_t col) const {
    return at(row, col);
  }
  [[nodiscard]] T &operator()(std::size_t row, std::size_t col) {
    return at(row, col);
  }
};

////////////////////////////////////////////////////////////////////////////////
// TODO Sparse tensors
/*
template <class T> class sparse_tensor final : expr, tensor_base {
public:
  using value_type = T;
  using node_type = ::ten::tensor_node<T, Storage, Allocator>;
  using storage_type = Storage;
  using base_type = tensor_operations<T, Shape, order, Storage, Allocator>;
  /// stride type
  using stride_type = typename base_type::stride_type;

private:
  /// Storage format (must be coo, csc or csr)
  ::ten::storage_format _format = ::ten::storage_format::coo;
  /// Optional shape (only for dynamic tensors)
  std::optional<Shape> _shape = std::nullopt;
  /// Optional stride (only for dynamic tensors)
  std::optional<stride_type> _stride = std::nullopt;
  /// Shared pointer to the node
  std::shared_ptr<node_type> _node = nullptr;

private:
  /// Returns the value at the indices
  [[nodiscard]] inline typename base_type::value_type &
  at(size_type index, auto... tail) noexcept {
    static constexpr size_type rank = Shape::rank();
    constexpr size_type tail_size = sizeof...(tail);
    static_assert(tail_size == 0 || tail_size == (rank - 1),
                  "Invalid number of indices.");
    if constexpr (tail_size == 0) {
      return (*_node.get())[index];
    }
    std::array<size_type, Shape::rank()> indices{
        index, static_cast<size_type>(tail)...};
    if constexpr (Shape::is_dynamic()) {
      size_type idx = details::linear_index(_stride.value(), indices);
      return (*_node.get())[idx];
    } else {
      size_type idx = details::static_linear_index<stride_type>(indices);
      return (*_node.get())[idx];
    }
  }

  /// Returns the value at the indices
  [[nodiscard]] inline const typename base_type::value_type &
  at(size_type index, auto... tail) const noexcept {
    static constexpr size_type rank = Shape::rank();
    constexpr size_type tail_size = sizeof...(tail);
    static_assert(tail_size == 0 || tail_size == (rank - 1),
                  "Invalid number of indices.");
    if constexpr (tail_size == 0) {
      return (*_node.get())[index];
    }
    std::array<size_type, Shape::rank()> indices{
        index, static_cast<size_type>(tail)...};
    if constexpr (Shape::is_dynamic()) {
      size_type idx = details::linear_index(_stride.value(), indices);
      return (*_node.get())[idx];
    } else {
      size_type idx = details::static_linear_index<stride_type>(indices);
      return (*_node.get())[idx];
    }
  }

public:
  /// Constructor for static sparse tensor
  ranked_sparse_tensor(ten::storage_format format = storage_format::coo)
    requires(Shape::is_static())
      : _format(format), _stride(typename base_type::stride_type()) {
    auto storage = std::make_unique<storage_type>();
    _node = std::make_shared<node_type>(storage);
  }

  // Constructor for sparse tensor
  explicit ranked_sparse_tensor(
      std::initializer_list<size_t> &&dims,
      ten::storage_format format = storage_format::coo) noexcept
    requires(Shape::is_dynamic())
      : _format(format), _shape(std::move(dims)),
        _stride(typename base_type::stride_type(_shape.value())) {
    auto storage = std::make_unique<storage_type>(_shape.value());
    _node = std::make_shared<node_type>(std::move(storage));
  }

  /// Copy constructor
  ranked_sparse_tensor(const ranked_sparse_tensor &t) {
    _format = t._format;
    _shape = t._shape;
    _stride = t._stride;
    _node = t._node;
  }

  /// Move constructor
  ranked_sparse_tensor(ranked_sparse_tensor &&t) {
    _format = std::move(t._format);
    _shape = std::move(t._shape);
    _stride = std::move(t._stride);
    _node = std::move(t._node);
  }

  /// Assignment operator
  ranked_sparse_tensor &operator=(const ranked_sparse_tensor &t) {
    _format = t._format;
    _shape = t._shape;
    _stride = t._stride;
    _node = t._node;
    return *this;
  }

  /// Assignment operator
  ranked_sparse_tensor &operator=(ranked_sparse_tensor &&t) {
    _format = std::move(t._format);
    _shape = std::move(t._shape);
    _stride = std::move(t._stride);
    _node = std::move(t._node);
    return *this;
  }

  /// Get the dimension at index
  /// FIXME Requires only for dynamic dim
  [[nodiscard]] size_type dim(size_type index) const {
    return _shape.value().dim(index);
  }

  /// Returns the shape
  [[nodiscard]] inline const Shape &shape() const
    requires(Shape::is_dynamic())
  {
    return _shape.value();
  }

  /// Returns the strides
  [[nodiscard]] inline const typename base_type::stride_type &strides() const {
    return _stride.value();
  }

  /// Returns the dynamic size
  /// TODO For static size
  [[nodiscard]] inline size_type size() const { return _node.get()->size(); }

  /// Get the storage format
  [[nodiscard]] storage_format format() const { return _format; }

  /// Return whether the tensor is sparse coo
  [[nodiscard]] bool is_sparse_coo() const {
    return _format == ::ten::storage_format::coo;
  }

  /// Return whether the tensor is sparse csc
  [[nodiscard]] bool is_sparse_csc() const {
    return _format == ::ten::storage_format::csc;
  }

  /// Return whether the tensor is sparse csr
  [[nodiscard]] bool is_sparse_csr() const {
    return _format == ::ten::storage_format::csr;
  }

  // Returns the shared ptr to the node
  [[nodiscard]] std::shared_ptr<node_type> node() const { return _node; }

  /// Returns the storage
  [[nodiscard]] Storage &storage() const { return _node.get()->storage(); }

  /// Overloading the [] operator
  [[nodiscard]] inline const typename base_type::value_type &
  operator[](size_type index) const noexcept {
    return (*_node.get())[index];
  }

  /// Overloading the [] operator
  [[nodiscard]] inline typename base_type::value_type &
  operator[](size_type index) noexcept {
    return (*_node.get())[index];
  }

  /// Overloading the () operator
  [[nodiscard]] inline const typename base_type::value_type &
  operator()(auto... index) const noexcept {
    static_assert(sizeof...(index) == Shape::rank());
    return at(index...);
  }

  /// Overloading the () operator
  [[nodiscard]] inline typename base_type::value_type &
  operator()(auto... index) noexcept {
    static_assert(sizeof...(index) == Shape::rank());
    return at(index...);
  }
};*/

////////////////////////////////////////////////////////////////////////////////
// tensor view
template <class T> class view final {
public:
  using tensor_type = tensor<T>;

private:
  tensor_type _data;
  std::vector<std::size_t> _start = {};
  std::vector<std::size_t> _end = {};

public:
  explicit view(const tensor_type &data, const std::vector<std::size_t> &start,
                const std::vector<std::size_t> &end)
      : _data(data), _start(start), _end(end) {}

  explicit view(const tensor_type &data, std::vector<std::size_t> &&start,
                std::vector<std::size_t> &&end)
      : _data(data), _start(std::move(start)), _end(std::move(end)) {}

  /// Get the rank
  [[nodiscard]] std::size_t rank() const { return _data.rank(); }

  // Assign a value
  view &operator=(T value) noexcept {
    std::size_t r = rank();

    if (r > 5) {
      std::cerr << "Assignemnt is supported only for up to 5d tensors.\n";
    }

    if (r == 1) {
      for (size_t i = _start[0]; i < _end[0]; i++) {
        _data[i] = value;
      }
    } else if (r == 2) {
      for (size_t i = _start[0]; i < _end[0]; i++) {
        for (size_t j = _start[1]; j < _end[1]; j++) {
          _data(i, j) = value;
        }
      }
    } else if (r == 3) {
      for (size_t i = _start[0]; i < _end[0]; i++) {
        for (size_t j = _start[1]; j < _end[1]; j++) {
          for (size_t k = _start[2]; j < _end[2]; k++) {
            _data(i, j, k) = value;
          }
        }
      }
    } else if (r == 4) {
      for (size_t i = _start[0]; i < _end[0]; i++) {
        for (size_t j = _start[1]; j < _end[1]; j++) {
          for (size_t k = _start[2]; j < _end[2]; k++) {
            for (size_t l = _start[3]; l < _end[3]; l++) {
              _data(i, j, k, l) = value;
            }
          }
        }
      }
    } else if (r == 5) {
      for (size_t i = _start[0]; i < _end[0]; i++) {
        for (size_t j = _start[1]; j < _end[1]; j++) {
          for (size_t k = _start[2]; j < _end[2]; k++) {
            for (size_t l = _start[3]; l < _end[3]; l++) {
              for (size_t m = _start[4]; m < _end[4]; m++) {
                _data(i, j, k, l, m) = value;
              }
            }
          }
        }
      }
    }
    return *this;
  }

  // Assign a vector
  view &operator=(const ten::tensor<T> &values) noexcept {
    std::size_t values_rank = values.rank();

    if (values_rank == 1) {
      std::size_t r = rank();
      if (r > 5) {
        std::cerr << "Vector assignemnt is supported only for slices of up to "
                     "5d tensors.\n";
      }

      if (r == 1) {
        if (values.size() != (_end[0] - _start[0])) {
          std::cerr << "Error Assigning a vector to a vector slice, "
                       "incompatible sizes.\n";
        } else {
          for (std::size_t i = _start[0]; i < _end[0]; i++) {
            _data[i] = values[i - _start[0]];
          }
        }
      } else if (r == 2) {
        if (values.size() != (_end[0] - _start[0]) * (_end[1] - _start[1])) {
          std::cerr << "Error Assigning a vector to a matrix slice, "
                       "incompatible sizes.\n";
        } else {
          std::size_t k = 0;
          for (std::size_t i = _start[0]; i < _end[0]; i++) {
            for (std::size_t j = _start[1]; j < _end[1]; j++) {
              _data(i, j) = values[k];
              k++;
            }
          }
        }
      } else if (r == 3) {
        if (values.size() != (_end[0] - _start[0]) * (_end[1] - _start[1]) *
                                 (_end[2] - _start[2])) {
          std::cerr << "Error Assigning a vector to a tensor slice, "
                       "incompatible sizes.\n";
        } else {
          std::size_t l = 0;
          for (std::size_t i = _start[0]; i < _end[0]; i++) {
            for (std::size_t j = _start[1]; j < _end[1]; j++) {
              for (std::size_t k = _start[2]; k < _end[2]; k++) {
                _data(i, j, k) = values[l];
                l++;
              }
            }
          }
        }
      } else if (r == 4) {
        if (values.size() != (_end[0] - _start[0]) * (_end[1] - _start[1]) *
                                 (_end[2] - _start[2]) *
                                 (_end[3] - _start[3])) {
          std::cerr << "Error Assigning a vector to a tensor slice, "
                       "incompatible sizes.\n";
        } else {
          std::size_t m = 0;
          for (std::size_t i = _start[0]; i < _end[0]; i++) {
            for (std::size_t j = _start[1]; j < _end[1]; j++) {
              for (std::size_t k = _start[2]; k < _end[2]; k++) {
                for (std::size_t l = _start[3]; l < _end[3]; l++) {
                  _data(i, j, k, l) = values[m];
                  m++;
                }
              }
            }
          }
        }
      } else if (r == 5) {
        if (values.size() != (_end[0] - _start[0]) * (_end[1] - _start[1]) *
                                 (_end[2] - _start[2]) * (_end[3] - _start[3]) *
                                 (_end[4] - _start[4])) {
          std::cerr << "Error Assigning a vector to a tensor slice, "
                       "incompatible sizes.\n";
        } else {
          std::size_t n = 0;
          for (std::size_t i = _start[0]; i < _end[0]; i++) {
            for (std::size_t j = _start[1]; j < _end[1]; j++) {
              for (std::size_t k = _start[2]; k < _end[2]; k++) {
                for (std::size_t l = _start[3]; l < _end[3]; l++) {
                  for (std::size_t m = _start[4]; m < _end[4]; m++) {
                    _data(i, j, k, l, m) = values[n];
                    n++;
                  }
                }
              }
            }
          }
        }
      }
    } else if (values_rank == 2) {
      // Assign a matrix
      std::size_t r = rank();
      if (r > 5) {
        std::cerr << "Matrix assignemnt is supported only for slices of 2d to "
                     "5d tensors.";
      }

      if (r == 2) {
        if (values.size() != (_end[0] - _start[0]) * (_end[1] - _start[1])) {
          std::cerr << "Error Assigning a matrix to a matrix slice, "
                       "incompatible sizes.\n";
        } else {
          // FIXME Maybe Different data access for col major and row major
          // matrices
          for (std::size_t i = _start[0]; i < _end[0]; i++) {
            for (std::size_t j = _start[1]; j < _end[1]; j++) {
              for (std::size_t k = 0; k < values.dim(0); k++) {
                for (std::size_t l = 0; l < values.dim(1); l++) {
                  _data(i, j) = values(k, l);
                }
              }
            }
          }
        }
      } else if (r == 3) {
        if (values.size() != (_end[0] - _start[0]) * (_end[1] - _start[1]) *
                                 (_end[2] - _start[2])) {
          std::cerr << "Error Assigning a matrix to a tensor slice, "
                       "incompatible sizes.\n";
        } else {
          for (std::size_t i = _start[0]; i < _end[0]; i++) {
            for (std::size_t j = _start[1]; j < _end[1]; j++) {
              for (std::size_t k = _start[2]; k < _end[2]; k++) {
                for (std::size_t l = 0; l < values.dim(0); l++) {
                  for (std::size_t m = 0; m < values.dim(1); m++) {
                    _data(i, j, k) = values(l, m);
                  }
                }
              }
            }
          }
        }
      } else if (r == 4) {
        if (values.size() != (_end[0] - _start[0]) * (_end[1] - _start[1]) *
                                 (_end[2] - _start[2]) *
                                 (_end[3] - _start[3])) {
          std::cerr << "Error Assigning a matrix to a tensor slice, "
                       "incompatible sizes.\n";
        } else {
          for (std::size_t i = _start[0]; i < _end[0]; i++) {
            for (std::size_t j = _start[1]; j < _end[1]; j++) {
              for (std::size_t k = _start[2]; k < _end[2]; k++) {
                for (std::size_t l = _start[3]; l < _end[3]; l++) {
                  for (std::size_t m = 0; m < values.dim(0); m++) {
                    for (std::size_t n = 0; n < values.dim(1); n++) {
                      _data(i, j, k, l) = values(m, n);
                    }
                  }
                }
              }
            }
          }
        }
      } else if (r == 5) {
        if (values.size() != (_end[0] - _start[0]) * (_end[1] - _start[1]) *
                                 (_end[2] - _start[2]) * (_end[3] - _start[3]) *
                                 (_end[4] - _start[4])) {
          std::cerr << "Error Assigning a matrix to a tensor slice, "
                       "incompatible sizes.\n";
        } else {
          for (std::size_t i = _start[0]; i < _end[0]; i++) {
            for (std::size_t j = _start[1]; j < _end[1]; j++) {
              for (std::size_t k = _start[2]; k < _end[2]; k++) {
                for (std::size_t l = _start[3]; l < _end[3]; l++) {
                  for (std::size_t m = _start[4]; m < _end[4]; m++) {
                    for (std::size_t n = 0; n < values.shape().dim(0); n++) {
                      for (std::size_t p = 0; p < values.shape().dim(1); p++) {
                        _data(i, j, k, l, m) = values(n, p);
                      }
                    }
                  }
                }
              }
            }
          }
        }
      }
    }

    // TODO Assign a 3d, 4d and 5d tensor

    return *this;
  }
};

////////////////////////////////////////////////////////////////////////////////
// Special matrices

/// Uninitialized diagonal matrix
/*
template<typename T>
[[nodiscard]] tensor<T> diagonal(std::initializer_list<std::size_t>&& dims,
   bool requires_grad = false, ::ten::storage_order order =
::ten::storage_order::col_major) { std::vector<std::size_t>
shape(std::move(dims)); tensor<T> m(shape, ::ten::storage_format::diagonal,
requires_grad, order); return m;
}*/

// TODO Diagonal matrix initialized by a value
// TODO Diagonal matrix initialized by a vector

////////////////////////////////////////////////////////////////////////////////

/// transposed matrix
template <typename T>
[[nodiscard]] ten::tensor<T> transposed(const ten::tensor<T> &t) {
  ::ten::storage_format format = static_cast<storage_format>(
      t.format() | ::ten::storage_format::transposed);
  auto node = t.node();
  auto shape = t.shape();
  auto order = t.storage_order();
  return T(node, shape, format, false, order);
}

/// Symmetric tensor
template <typename T> ten::tensor<T> symmetric(const ten::tensor<T> &t) {
  ::ten::storage_format format = static_cast<storage_format>(
      t.format() | ::ten::storage_format::symmetric);
  auto node = t.node();
  auto shape = t.shape();
  auto order = t.storage_order();
  return T(node, shape, format, false, order);
}

// TODO Extend hermitian to tensor

/// Hermitian matrix
template <typename T> ten::tensor<T> hermitian(const ten::tensor<T> &t) {
  ::ten::storage_format format = static_cast<storage_format>(
      t.format() | ::ten::storage_format::hermitian);
  auto node = t.node();
  auto shape = t.shape();
  auto order = t.storage_order();
  return T(node, shape, format, false, order);
}

/// Conjugated tensor
template <typename T> ten::tensor<T> conj(const ten::tensor<T> &t) {
  ::ten::storage_format format =
      static_cast<storage_format>(t.format() | ::ten::storage_format::conj);
  auto node = t.node();
  auto shape = t.shape();
  auto order = t.storage_order();
  return T(node, shape, format, false, order);
}

/// Lower triangular matrix
template <typename T> ten::tensor<T> lower_tr(const ten::tensor<T> &t) {
  ::ten::storage_format format =
      static_cast<storage_format>(t.format() | ::ten::storage_format::lower_tr);
  auto node = t.node();
  auto shape = t.shape();
  auto order = t.storage_order();
  return T(node, shape, format, false, order);
}

/// Upper triangular matrix
template <typename T> ten::tensor<T> upper_tr(const ten::tensor<T> &t) {
  ::ten::storage_format format =
      static_cast<storage_format>(t.format() | ::ten::storage_format::upper_tr);
  auto node = t.node();
  auto shape = t.shape();
  auto order = t.storage_order();
  return T(node, shape, format, false, order);
}

////////////////////////////////////////////////////////////////////////////////
// TODO Columns
/*
template <class T>
class column final
    : public ten::expr, public ::ten::tensor_base {

public:

  /// \typedef base_type
  /// Type of the tensor operations.
  using base_type = tensor<T>;

  /// \typedef node_type
  /// Node type
  using node_type = tensor_node<T>;

private:
  // FOrmat is dense
  // _shape = {}
  size_t _index = 0;
   std::vector<std::size_t> _shape = {};

  std::shared_ptr<node_type> _node = nullptr;

public:
  /// Constructors for dynamic ranked_column
  ranked_column(const size_t index, const Shape &shape,
                const std::shared_ptr<node_type> &node) noexcept
    requires(Shape::is_dynamic())
      : _node(node), _shape(shape), _index(index) {}

  /// Constructor for static ranked_column
  ranked_column(const size_t index,
                const std::shared_ptr<node_type> &node) noexcept
    requires(Shape::is_static())
      : _index(index), _node(node) {}

  /// Copy constructor
  ranked_column(const ranked_column &t) { _node = t._node; }
  /// Move constructor
  ranked_column(ranked_column &&t) { _node = std::move(t._node); }

  /// Assignment operator
  ranked_column &operator=(const ranked_column &t) {
    _node = t._node;
    return *this;
  }
  /// Assignment operator
  ranked_column &operator=(ranked_column &&t) {
    _node = std::move(t._node);
    return *this;
  }

  /// Returns the shape
  [[nodiscard]] inline const Shape &shape() const { return _shape.value(); }

  /// Returns the strides
  [[nodiscard]] inline const typename base_type::stride_type &strides() const {
     return _node.get()->strides();
  }

  /// Returns the dynamic size
  [[nodiscard]] inline size_type size() const {
    // The size is the number of rows
    if constexpr (Shape::is_dynamic()) {
      return _shape.value().dim(0);
    }
    if constexpr (Shape::is_static()) {
      return Shape::template static_dim<0>();
    }
  }

  /// Returns the index'th dynamic dimension
  [[nodiscard]] inline size_type dim(size_type index) const {
    return _shape.value().dim(index);
  }

  // Returns the shared ptr to the node
  [[nodiscard]] std::shared_ptr<node_type> node() const { return _node; }

  /// Get the data
  [[nodiscard]] const T *data() const {
    size_t rows = this->shape().dim(0);
    return _node.get()->data() + _index * rows;
  }

  /// Get the data
  [[nodiscard]] T *data() {
    size_t rows = this->shape().dim(0);
    return _node.get()->data() + _index * rows;
  }

  /// Returns the shared ptr to the storage
  [[nodiscard]] std::shared_ptr<Storage> storage() const {
    return _node.get()->storage();
  }

  /// Overloading the [] operator
  [[nodiscard]] inline const typename base_type::value_type &
  operator[](size_type index) const noexcept {
    if constexpr (Shape::is_dynamic()) {
      size_t rows = this->shape().dim(0);
      return (*_node.get())[index + _index * rows];
    }
    if constexpr (Shape::is_static()) {
      size_t rows = Shape::template static_dim<0>();
      return (*_node.get())[index + _index * rows];
    }
  }

  /// Overloading the [] operator
  [[nodiscard]] inline typename base_type::value_type &
  operator[](size_type index) noexcept {
    if constexpr (Shape::is_dynamic) {
      size_t rows = this->shape().dim(0);
      return (*_node.get())[index + _index * rows];
    }
    if constexpr (Shape::is_static()) {
      size_t rows = Shape::template static_dim<0>();
      return (*_node.get())[index + _index * rows];
    }
  }

  /// Asignment from a static expression
  template <class ExprType>
    requires((::ten::is_unary_expr<std::remove_cvref_t<ExprType>>::value ||
              ::ten::is_binary_expr<std::remove_cvref_t<ExprType>>::value) &&
             std::remove_cvref_t<ExprType>::output_type::is_static())
  ranked_column &operator=(ExprType &&expr) noexcept {
    using expr_type = std::remove_cvref_t<ExprType>;
    using evaluated_type = typename expr_type::output_type;

    static_assert(::ten::is_tensor<evaluated_type>::value,
                  "Error: Evaluated type must be a tensor.");

    auto value = expr.eval();

    // FIXME maybe static_assert(Shape::template  ==
    // evaluated_type::static_size(),
    //               "Expected equal shape size.");
    size_t rows = _shape.value().dim(0);
    for (size_t idx = 0; idx < rows; idx++) {
      (*_node.get())[idx + _index * rows] = value[idx];
    }
    return *this;
  }

  /// Asignment from a dynamic expression
  template <class ExprType>
    requires((::ten::is_unary_expr<std::remove_cvref_t<ExprType>>::value ||
              ::ten::is_binary_expr<std::remove_cvref_t<ExprType>>::value) &&
             std::remove_cvref_t<ExprType>::output_type::is_dynamic())
  ranked_column &operator=(ExprType &&expr) noexcept {
    using expr_type = std::remove_cvref_t<ExprType>;
    using evaluated_type = typename expr_type::output_type;

    static_assert(::ten::is_tensor<evaluated_type>::value,
                  "Evaluated type must be a tensor.");
    auto value = expr.eval();
    // Copy the data
    size_t rows = _shape.value().dim(0);
    for (size_t idx = 0; idx < rows; idx++) {
      (*_node.get())[idx + _index * rows] = value[idx];
    }
    return *this;
  }

  // Asignement from a static vector
  template <StaticVector V> ranked_column &operator=(V &&value) noexcept {
    size_t rows = _shape.value().dim(0);
    for (size_t idx = 0; idx < rows; idx++) {
      (*_node.get())[idx + _index * rows] = value[idx];
    }
    return *this;
  }

  // Asgnement from a dynamic vector
  template <DynamicVector V> ranked_column &operator=(V &&value) noexcept {
    size_t rows = _shape.value().dim(0);
    for (size_t idx = 0; idx < rows; idx++) {
      (*_node.get())[idx + _index * rows] = value[idx];
    }
    return *this;
  }

  // Assign from a value of type T
  ranked_column &operator=(T value) noexcept {
    size_t rows = _shape.value().dim(0);
    for (size_t idx = 0; idx < rows; idx++) {
      (*_node.get())[idx + _index * rows] = value;
    }
    return *this;
  }

  // Convert a column to a vector
  auto vector() {
    size_t rows = _shape.value().dim(0);
    ten::vector<T, Order, Storage, Allocator> v({rows});
    for (size_t idx = 0; idx < rows; idx++) {
      v[idx] = (*_node.get())[idx + _index * rows];
    }
    return v;
  }

  // Convert a column to a static vector
  auto svector() {
    constexpr size_t rows = Shape::template static_dim<0>();
    ten::svector<T, rows, Order> v;
    for (size_t idx = 0; idx < rows; idx++) {
      v[idx] = (*_node.get())[idx + _index * rows];
    }
    return v;
  }
};*/

////////////////////////////////////////////////////////////////////////////////
// TODO Rows
/*
template <class T>
class row final
    : public ten::expr,public ::ten::tensor_base {

public:
  /// \typedef base_type
  /// Type of the tensor operations.
  using base_type = tensor<T>;

  /// \typedef node_type
  /// Node type
  using node_type = tensor_node<T>;

private:
  size_t _index = 0;
   std::vector<std::size_t> _shape = {};
  std::shared_ptr<node_type> _node = nullptr;

public:
  /// Constructors for dynmaic ranked_row
  ranked_row(const size_t index, const Shape &shape,
             const std::shared_ptr<node_type> &node) noexcept
    requires(Shape::is_dynamic())
      : _index(index), _shape(shape), _node(node) {}

  /// Constructor for static ranked_row
  ranked_row(const size_t index,
             const std::shared_ptr<node_type> &node) noexcept
    requires(Shape::is_static())
      : _index(index), _shape(std::nullopt), _node(node) {}

  /// Copy constructor
  // ranked_row(const ranked_row &t) { _node = t._node; }
  /// Move constructor
  // ranked_row(ranked_row &&t) { _node = std::move(t._node); }

  /// Assignment operator
  ranked_row &operator=(const ranked_row &t) {
     _node = t._node;
     return *this;
  }
  /// Assignment operator
  ranked_row &operator=(ranked_row &&t) {
     _node = std::move(t._node);
     return *this;
  }

  /// Returns the shape
  /// TODO Requires for only dynamic shape row
  [[nodiscard]] inline const Shape &shape() const { return _shape.value(); }

  /// Returns the strides
  [[nodiscard]] inline const typename base_type::stride_type &strides() const {
     return _node.get()->strides();
  }

  /// Returns the dynamic size
  [[nodiscard]] inline size_type size() const {
    // The size if the number of cols
    if constexpr (Shape::is_dynamic()) {
      return _shape.value().dim(1);
    }
    if constexpr (Shape::is_static()) {
      return Shape::template static_dim<1>();
    }
  }

  /// Returns the index'th dynamic dimension
  [[nodiscard]] inline size_type dim(size_type index) const {
    return _shape.value().dim(index);
  }

  // Returns the shared ptr to the node
  [[nodiscard]] std::shared_ptr<node_type> node() const { return _node; }

  /// Get the data
  [[nodiscard]] const T *data() const { return _node.get()->data() + _index; }

  /// Get the data
  [[nodiscard]] T *data() { return _node.get()->data() + _index; }

  /// Returns the shared ptr to the storage
  [[nodiscard]] std::shared_ptr<Storage> storage() const {
    return _node.get()->storage();
  }

  /// Overloading the [] operator
  [[nodiscard]] inline const typename base_type::value_type &
  operator[](size_type index) const noexcept {
    if constexpr (Shape::is_dynamic()) {
      size_t rows = this->shape().dim(0);
      return (*_node.get())[_index + index * rows];
    }
    if constexpr (Shape::is_static()) {
      constexpr size_t rows = Shape::template static_dim<0>();
      return (*_node.get())[_index + index * rows];
    }
  }

  /// Overloading the [] operator
  [[nodiscard]] inline typename base_type::value_type &
  operator[](size_type index) noexcept {
    if constexpr (Shape::is_dynamic()) {
      size_t rows = this->shape().dim(0);
      return (*_node.get())[_index + index * rows];
    }
    if constexpr (Shape::is_static()) {
      size_t rows = Shape::template static_dim<0>();
      return (*_node.get())[_index + index * rows];
    }
  }

  /// Asignment from a static expression
  template <class ExprType>
    requires((::ten::is_unary_expr<std::remove_cvref_t<ExprType>>::value ||
              ::ten::is_binary_expr<std::remove_cvref_t<ExprType>>::value) &&
             std::remove_cvref_t<ExprType>::output_type::is_static())
  ranked_row &operator=(ExprType &&expr) noexcept {
    using expr_type = std::remove_cvref_t<ExprType>;
    using evaluated_type = typename expr_type::output_type;

    static_assert(::ten::is_tensor<evaluated_type>::value,
                  "Error: Evaluated type must be a tensor.");

    auto value = expr.eval();

    // FIXME maybe static_assert(Shape::template  ==
    // evaluated_type::static_size(),
    //               "Expected equal shape size.");
    size_t rows = _shape.value().dim(0);
    size_t cols = _shape.value().dim(1);
    for (size_t idx = 0; idx < cols; idx++) {
      (*_node.get())[_index + idx * rows] = value[idx];
    }
    return *this;
  }

  /// Asignment from a dynamic expression
  template <class ExprType>
    requires((::ten::is_unary_expr<std::remove_cvref_t<ExprType>>::value ||
              ::ten::is_binary_expr<std::remove_cvref_t<ExprType>>::value) &&
             std::remove_cvref_t<ExprType>::output_type::is_dynamic())
  ranked_row &operator=(ExprType &&expr) noexcept {
    using expr_type = std::remove_cvref_t<ExprType>;
    using evaluated_type = typename expr_type::output_type;

    static_assert(::ten::is_tensor<evaluated_type>::value,
                  "Evaluated type must be a tensor.");
    auto value = expr.eval();
    // Copy the data
    size_t rows = _shape.value().dim(0);
    size_t cols = _shape.value().dim(1);
    for (size_t idx = 0; idx < cols; idx++) {
      (*_node.get())[_index + idx * rows] = value[idx];
    }
    return *this;
  }

  // Asignement from a static vector
  template <StaticVector V> ranked_row &operator=(V &&value) noexcept {
    size_t rows = _shape.value().dim(0);
    size_t cols = _shape.value().dim(1);
    for (size_t idx = 0; idx < cols; idx++) {
      (*_node.get())[_index + idx * rows] = value[idx];
    }
    return *this;
  }

  // Assignement from a dynamic vector
  template <DynamicVector V> ranked_row &operator=(V &&value) noexcept {
    size_t rows = _shape.value().dim(0);
    size_t cols = _shape.value().dim(1);
    for (size_t idx = 0; idx < cols; idx++) {
      (*_node.get())[_index + idx * rows] = value[idx];
    }
    return *this;
  }

  // Assign a value of type T
  ranked_row &operator=(T value) noexcept {
    size_t rows = _shape.value().dim(0);
    size_t cols = _shape.value().dim(1);
    for (size_t idx = 0; idx < cols; idx++) {
      (*_node.get())[_index + idx * rows] = value;
    }
    return *this;
  }

  // Convert a row to a vector
  auto vector() {
    size_t rows = _shape.value().dim(0);
    size_t cols = _shape.value().dim(1);
    ten::vector<T, Order, Storage, Allocator> v({cols});
    for (size_t idx = 0; idx < cols; idx++) {
      v[idx] = (*_node.get())[_index + idx * rows];
    }
    return v;
  }

  // TODO Convert a row to a static vector
  auto svector() {
    constexpr size_t rows = Shape::template static_dim<0>();
    constexpr size_t cols = Shape::template static_dim<1>();
    ten::svector<T, cols, Order> v;
    for (size_t idx = 0; idx < cols; idx++) {
      v[idx] = (*_node.get())[_index + idx * rows];
    }
    return v;
  }
};*/

////////////////////////////////////////////////////////////////////////////////
// Basic functions

// Cast a dense tensor
template <class To, Tensor T> auto cast(const T &x) {
  using tensor_type = ten::tensor<To>;
  auto format = x.format();
  tensor_type r(x.shape(), format, x.requires_grad(), x.storage_order());
  if (format == ten::storage_format::dense) {
    for (size_type i = 0; i < x.size(); i++) {
      r[i] = static_cast<To>(x[i]);
    }
  } else {
    std::cerr << "cast, Tensor must be dense.\n";
  }
  return r;
}

// TODO reshape(x, shape)
/*
template <Expr ExprType>
[[nodiscard]] auto reshape(ExprType &&expr, std::vector<std::size_t> &dims) {
  using expr_type = typename std::remove_cvref_t<ExprType>;
  using output_type = typename details::output_type<expr_type>::type;
  return ::ten::unary_expr<
      expr_type, ::ten::functional::reshape<value_type>::template
func>(expr,dims);
}

template<Expr ExprType>
[[nodiscard]] auto reshape(ExprType&& expr, std::initializer_list<std::size_t>&&
dims) { std::vector<std::size_t> shape(std::move(dims)); return reshape(expr,
shape);
}*/

// TODO flatten(x)
/*template <Expr ExprType> auto flatten(ExprType expr) {
  using expr_type = std::remove_cvref_t<ExprType>;

  // tensor
  if constexpr (is_tensor<expr_type>::value) {
      std::vector<std::size_t> shape = {expr.size()};
      return reshape(std::forward<expr_type>(expr), shape);
    }
  }

  // unar_expr or binary_expr
  if constexpr (is_unary_expr<expr_type>::value ||
                is_binary_expr<expr_type>::value) {
    using output_type = typename expr_type::evaluated_type;
    using shape_type = typename output_type::shape_type;
    return reshape<, expr_type>(std::forward<expr_type>(expr));
  }
}*/

// TODO transpose(x)
/*
template <Expr ExprType> auto transpose(ExprType &&expr) {
  using expr_type = std::remove_cvref_t<ExprType>;
  static_assert(::ten::is_tensor<expr_type>::value ||
                    ::ten::is_column<expr_type>::value ||
                    ::ten::is_row<expr_type>::value,
                "Currently transpose support only tensor, column and row.");

  // tensor
  if constexpr (is_tensor<expr_type>::value) {
    using shape_type = expr_type::shape_type;

    if constexpr (shape_type::is_static()) {
      using transpose_type =
          ::ten::details::static_transpose_result<expr_type, shape_type>::type;
      return ::ten::unary_expr<
          expr_type, transpose_type,
          ::ten::functional::static_transpose<shape_type>::template func>(expr);
    }

    if constexpr (shape_type::is_dynamic()) {
      using transpose_result =
          ::ten::details::transpose_result<expr_type, shape_type>::type;
      return ::ten::unary_expr<
          expr_type, transpose_result,
          ::ten::functional::dynamic_transpose<shape_type>::template func>(
          expr);
    }
  }

  // TODO unar_expr or binary_expr
  // if constexpr (is_unary_expr<expr_type>::value ||
  //             is_binary_expr<expr_type>::value) {
  //  using output_type = typename expr_type::evaluated_type;
  //  using shape_type = typename output_type::shape_type;

  //  return reshape<::ten::shape<shape_type::staticSize()>, expr_type>(
  //       std::forward<expr_type>(expr));
  //}
}*/

////////////////////////////////////////////////////////////////////////////////
// Functions for creating a new tensor

// fill<tensor<...>>(Shape, value)
template <class T>
  requires(::ten::is_tensor<T>::value)
[[nodiscard]] auto fill(const std::vector<std::size_t> &shape,
                        typename T::value_type value,
                        const bool requires_grad = false,
                        const storage_order order = storage_order::col_major) {
  T x(shape, ::ten::storage_format::dense, requires_grad, order);
  for (size_type i = 0; i < x.size(); i++) {
    x[i] = value;
  }
  return x;
}

template <class T>
  requires(::ten::is_tensor<T>::value)
[[nodiscard]] auto fill(std::initializer_list<std::size_t> &&dims,
                        typename T::value_type value,
                        const bool requires_grad = false,
                        const storage_order order = storage_order::col_major) {
  std::vector<size_t> shape(std::move(dims));
  return fill<T>(shape, value, requires_grad, order);
}

// fill<T>(shape, value)
template <typename T>
[[nodiscard]] auto fill(const std::vector<std::size_t> &dims, T value,
                        const bool requires_grad = false,
                        const storage_order order = storage_order::col_major) {
  using tensor_type = tensor<T>;
  return fill<tensor_type>(dims, value, requires_grad, order);
}
template <typename T>
[[nodiscard]] auto fill(std::initializer_list<std::size_t> &&dims, T value,
                        bool requires_grad = false,
                        const storage_order order = storage_order::col_major) {
  std::vector<std::size_t> shape(std::move(dims));
  using tensor_type = tensor<T>;
  return fill<tensor_type>(shape, value, requires_grad, order);
}

// zeros<tensor<...>>(shape)
template <class T>
  requires(::ten::is_tensor_v<T>)
[[nodiscard]] auto zeros(const std::vector<std::size_t> &dims,
                         const bool requires_grad = false,
                         const storage_order order = storage_order::col_major) {
  using value_type = typename T::value_type;
  return fill<T>(dims, value_type(0), requires_grad, order);
}
template <class T>
  requires(::ten::is_tensor_v<T>)
[[nodiscard]] auto zeros(std::initializer_list<std::size_t> &&dims,
                         const bool requires_grad = false,
                         const storage_order order = storage_order::col_major) {
  return zeros<T>(std::move(dims), requires_grad, order);
}
// zeros<T>(shape)
template <typename T>
[[nodiscard]] auto zeros(const std::vector<std::size_t> &dims,
                         const bool requires_grad = false,
                         const storage_order order = storage_order::col_major) {
  return zeros<tensor<T>>(dims, requires_grad, order);
}
template <typename T>
[[nodiscard]] auto zeros(std::initializer_list<std::size_t> &&dims,
                         const bool requires_grad = false,
                         const storage_order order = storage_order::col_major) {
  const std::vector<std::size_t> shape(std::move(dims));
  return zeros<tensor<T>>(shape, requires_grad, order);
}

// ones<tensor<...>>(shape)
template <class T>
  requires(::ten::is_tensor_v<T>)
[[nodiscard]] auto ones(const std::vector<std::size_t> &dims,
                        const bool requires_grad = false,
                        const storage_order order = storage_order::col_major) {
  using value_type = typename T::value_type;
  return fill<T>(dims, value_type(1), requires_grad, order);
}

template <class T>
  requires(::ten::is_tensor_v<T>)
[[nodiscard]] auto ones(std::initializer_list<std::size_t> &&dims,
                        const bool requires_grad = false,
                        const storage_order order = storage_order::col_major) {
  using value_type = typename T::value_type;
  const std::vector<std::size_t> shape(std::move(dims));
  return fill<T>(shape, value_type(1), requires_grad, order);
}

// ones<T>(shape)
template <class T>
[[nodiscard]] auto ones(const std::vector<std::size_t> &dims,
                        const bool requires_grad = false,
                        const storage_order order = storage_order::col_major) {
  return fill<tensor<T>>(dims, T(1), requires_grad, order);
}
template <class T>
[[nodiscard]] auto ones(std::initializer_list<std::size_t> &&dims,
                        const bool requires_grad = false,
                        const storage_order order = storage_order::col_major) {
  const std::vector<std::size_t> shape(std::move(dims));
  return fill<tensor<T>>(shape, T(1), requires_grad, order);
}

// range<tensor<...>>(shape, value)
template <class T>
  requires(::ten::is_tensor_v<T>)
[[nodiscard]] auto
range(const std::vector<std::size_t> &dims,
      typename T::value_type value = typename T::value_type(0),
      const bool requires_grad = false,
      const storage_order order = storage_order::col_major) {
  using value_type = typename T::value_type;
  T x(dims, requires_grad, order);
  x[0] = value;
  for (std::size_t i = 1; i < x.size(); i++) {
    x[i] = x[i - 1] + value_type(1);
  }
  return x;
}
template <class T>
  requires(::ten::is_tensor_v<T>)
[[nodiscard]] auto
range(std::initializer_list<size_type> &&dims,
      typename T::value_type value = typename T::value_type(0),
      const bool requires_grad = false,
      const storage_order order = storage_order::col_major) {
  const std::vector<std::size_t> shape(std::move(dims));
  return range<T>(shape, value, requires_grad, order);
}

// range<T>(shape, value)
template <class T>
[[nodiscard]] auto range(const std::vector<std::size_t> &dims, T value = T(0),
                         const bool requires_grad = false,
                         const storage_order order = storage_order::col_major) {
  return range<tensor<T>>(dims, value, requires_grad, order);
}
template <class T>
[[nodiscard]] auto range(std::initializer_list<std::size_t> &&dims,
                         T value = T(0), const bool requires_grad = false,
                         const storage_order order = storage_order::col_major) {
  const std::vector<std::size_t> shape(std::move(dims));
  return range<tensor<T>>(shape, value, requires_grad, order);
}

// linear<tensor<...>>(shape, start, stop, requires_grad, order)
template <class T>
  requires(::ten::is_tensor_v<T>)
[[nodiscard]] auto
linear(const std::vector<std::size_t> &dims, typename T::value_type start,
       typename T::value_type stop, const bool requires_grad = false,
       const storage_order order = storage_order::col_major) {
  using value_type = typename T::value_type;
  T x(dims, requires_grad, order);
  x[0] = start;
  size_t n = x.size();
  value_type step = (stop - start) / (n - 1);
  for (std::size_t i = 1; i < x.size(); i++) {
    x[i] = x[i - 1] + step;
  }
  return x;
}
template <class T>
  requires(::ten::is_tensor_v<T>)
[[nodiscard]] auto
linear(std::initializer_list<std::size_t> &&dims, typename T::value_type start,
       typename T::value_type stop, const bool requires_grad = false,
       const storage_order order = storage_order::col_major) {
  const std::vector<std::size_t> shape(std::move(dims));
  return linear<T>(shape, start, stop, requires_grad, order);
}

// linear<T>(shape, start, stop, requires_grad, order)
template <typename T>
[[nodiscard]] auto
linear(const std::vector<std::size_t> &dims, T start, T stop,
       const bool requires_grad = false,
       const storage_order order = storage_order::col_major) {
  return linear<tensor<T>>(dims, start, stop, requires_grad, order);
}
template <typename T>
[[nodiscard]] auto
linear(std::initializer_list<std::size_t> &&dims, T start, T stop,
       const bool requires_grad = false,
       const storage_order order = storage_order::col_major) {
  const std::vector<std::size_t> shape(std::move(dims));
  return linear<tensor<T>>(shape, start, stop, requires_grad, order);
}

////////////////////////////////////////////////////////////////////////////////
// TODO Identity

/*
// identity<matrix<...>>(shape)
template <Tensor T>
[[nodiscard]] auto identity(std::vector<std::size_t> &dims) {
  using value_type = typename T::value_type;
  using shape_type = typename T::shape_type;

  size_type m = std::min(shape.dim(0), shape.dim(1));
  size_type n = shape.size();
  T x(std::forward<shape_type>(shape));
  for (size_type i = 0; i < m; i++) {
    x(i, i) = value_type(1);
  }
  return x;
}

template <Matrix T>
[[nodiscard]] auto identity(std::initializer_list<size_type> &&dims) {
}*/

////////////////////////////////////////////////////////////////////////////////
// Conversion from special matrices and tensors

/// Convert diagonal matrix to dense
template <Diagonal T> auto dense(T x) -> decltype(auto) {
  using value_type = T::value_type;
  std::size_t m = x.dim(0);
  std::size_t n = x.dim(1);
  if (m != n) {
    std::cerr << "Diagonal matrix must be square\n";
  }
  auto order = x.storage_order();
  tensor<value_type> y = zeros<value_type>({m, n}, false, order);
  for (size_t i = 0; i < m; i++) {
    y(i, i) = x[i];
  }
  return y;
}

////////////////////////////////////////////////////////////////////////////////
// Expressions

// TODO Blas level 1 functions

/*
/// asum
/// Sum of the absolute values of the elements of a vector
template <Expr X> static auto asum(X &&x) -> decltype(auto) {
  using expr_type = std::remove_cvref_t<X>;
  if constexpr (::ten::is_tensor_v<expr_type> ||
                ::ten::is_column_v<expr_type> || ::ten::is_row_v<expr_type>) {
    return ::ten::kernels::asum(std::forward<X>(x));
  } else {
    auto value = x.eval();
    using tensor_type = decltype(value);
    return ::ten::kernels::asum(std::forward<tensor_type>(value));
  }
}

/// axpy
/// y <- a*x + y
template <typename T, Expr X, class Y>
  requires(::ten::is_vector_v<Y> || ::ten::is_column_v<Y> || ::ten::is_row_v<Y>)
static void axpy(const T a, X &&x, Y &y) {
  using x_expr_type = std::remove_cvref_t<X>;

  if constexpr (::ten::is_vector_v<x_expr_type> ||
                ::ten::is_column_v<x_expr_type> ||
                ::ten::is_row_v<x_expr_type>) {
    ::ten::kernels::axpy(a, std::forward<X>(x), y);
  }
  if constexpr (!::ten::is_vector_v<x_expr_type> &&
                !::ten::is_column_v<x_expr_type> &&
                !::ten::is_row_v<x_expr_type>) {
    auto xtensor = x.eval();
    using tensor_type = decltype(xtensor);
    ::ten::kernels::axpy(a, std::forward<tensor_type>(xtensor), y);
  }
}

/// copy
/// Copy a vector to another vector
template <class X, class Y>
  requires(
      (::ten::is_vector_v<X> || ::ten::is_column_v<X> || ::ten::is_row_v<X>) &&
      (::ten::is_vector_v<Y> || ::ten::is_column_v<Y> || ::ten::is_row_v<Y>))
static void copy(const X &x, Y &y) {
  ::ten::kernels::copy(x, y);
}

/// dot
/// Dot product of two vectors
template <class X, class Y>
  requires(
      (::ten::is_vector_v<X> || ::ten::is_column_v<X> || ::ten::is_row_v<X>) &&
      (::ten::is_vector_v<Y> || ::ten::is_column_v<Y> || ::ten::is_row_v<Y>))
static auto dot(const X &x, const Y &y) -> decltype(auto) {
  return ::ten::kernels::dot(x, y);
}

/// dotc
/// Dot product of a conjugated vector with another vector
template <class X, class Y>
  requires(
      (::ten::is_vector_v<X> || ::ten::is_column_v<X> || ::ten::is_row_v<X>) &&
      (::ten::is_vector_v<Y> || ::ten::is_column_v<Y> || ::ten::is_row_v<Y>))
static auto dotc(const X &x, const Y &y) -> decltype(auto) {
  using x_value_type = typename X::value_type;
  static_assert(::ten::is_complex<x_value_type>::value);
  using y_value_type = typename Y::value_type;
  static_assert(::ten::is_complex<y_value_type>::value);

  return ::ten::kernels::dotc(x, y);
}

/// iamax
/// Position of the element with maximum absolute value
template <class X>
  requires(::ten::is_vector_v<X> || ::ten::is_column_v<X> || ::ten::is_row_v<X>)
static auto iamax(const X &x) -> decltype(auto) {
  return ::ten::kernels::iamax(x);
}

/// nrm2
/// Euclidian norm of a vector
template <class X>
  requires(::ten::is_vector_v<X> || ::ten::is_column_v<X> || ::ten::is_row_v<X>)
static auto nrm2(const X &x) -> decltype(auto) {
  return ::ten::kernels::nrm2(x);
}

/// scal
/// Compute the scaled vector alpha*x and reset the result to x
template <typename T, class X>
  requires(::ten::is_vector_v<X> || ::ten::is_column_v<X> || ::ten::is_row_v<X>)
static void scal(const T alpha, X &x) {
  ::ten::kernels::scal(alpha, x);
}

/// swap
/// Swap the contents of two vectors
template <class X, class Y>
  requires(
      (::ten::is_vector_v<X> || ::ten::is_column_v<X> || ::ten::is_row_v<X>) &&
      (::ten::is_vector_v<Y> || ::ten::is_column_v<Y> || ::ten::is_row_v<Y>))
static void swap(X &x, Y &y) {
  // FIXME This requires BLAS
  // ::ten::kernels::swap(x, y);
}

// Blas level 2 functions

/// gemv
/// Generalized matrix vector multiplication
template <typename T, class A, class B, class C>
  requires(
      ::ten::is_matrix_v<A> &&
      (::ten::is_vector_v<B> || ::ten::is_column_v<B> || ::ten::is_row_v<B>) &&
      (::ten::is_vector_v<C> || ::ten::is_column_v<C> || ::ten::is_row_v<C>))
static void gemv(const T alpha, const A &a, const B &b, const T beta, C &c) {
  ::ten::kernels::gemv(alpha, a, b, beta, c);
}

/// ger
/// Rank one update a <- a + alpha * x outer y
template <typename T, class X, class Y, class A>
  requires(
      ::ten::is_matrix_v<A> &&
      (::ten::is_vector_v<X> || ::ten::is_column_v<X> || ::ten::is_row_v<X>) &&
      (::ten::is_vector_v<Y> || ::ten::is_column_v<Y> || ::ten::is_row_v<Y>))
static void ger(const T alpha, const X &x, const Y &y, A &a) {
  ::ten::kernels::ger(alpha, x, y, a);
}
*/

/*
/// TODO symv
/// Symmetric matrix vector multiplication
/// y <- alpha * A * x + beta * y
template<typename T, class A, class X, class Y>
requires(::ten::is_matrix_v<A> && (::ten::is_vector_v<X> ||
::ten::is_column_v<X> || ::ten::is_row_v<X>)
   && (::ten::is_vector_v<Y> || ::ten::is_column_v<Y> || ::ten::is_row_v<Y>))
static void symv(const T alpha, const A& a, const X& x, const T beta, Y& y) {
   ::ten::kernels::symv(alpha, a, x, beta, y);
}*/

// TODO BLAS Level 3 functions

/*
/// gemm
/// C <- alpha * X * Y + beta * C
template <class T, Expr X, Expr Y, Tensor C>
  requires(std::is_same_v<T, typename C::value_type>)
static void gemm(const T alpha, X &&x, Y &&y, const T beta, C &c) {
  using x_expr_type = std::remove_cvref_t<X>;
  using y_expr_type = std::remove_cvref_t<Y>;

  if constexpr (::ten::is_tensor<x_expr_type>::value &&
                ::ten::is_tensor<y_expr_type>::value) {
    ::ten::kernels::mul_add<T>(alpha, std::forward<X>(x), std::forward<Y>(y),
                               beta, c);
  }

  if constexpr (::ten::is_tensor<x_expr_type>::value &&
                !::ten::is_tensor<y_expr_type>::value) {
    auto ytensor = y.eval();
    using YTensor = decltype(ytensor);
    ::ten::kernels::mul_add<T>(alpha, std::forward<X>(x),
                               std::forward<YTensor>(ytensor), beta, c);
  }

  if constexpr (!::ten::is_tensor<x_expr_type>::value &&
                ::ten::is_tensor<y_expr_type>::value) {
    auto xtensor = x.eval();
    using XTensor = decltype(xtensor);
    ::ten::kernels::mul_add<T>(alpha, std::forward<XTensor>(xtensor),
                               std::forward<Y>(y), beta, c);
  }

  if constexpr (!::ten::is_tensor<x_expr_type>::value &&
                !::ten::is_tensor<y_expr_type>::value) {
    auto xtensor = x.eval();
    using XTensor = decltype(xtensor);
    auto ytensor = y.eval();
    using YTensor = decltype(ytensor);
    ::ten::kernels::mul_add<T>(alpha, std::forward<XTensor>(x),
                               std::forward<YTensor>(y), beta, c);
  }
}*/

////////////////////////////////////////////////////////////////////////////////
/// TODO functions

/*
/// \fn min
/// Returns the maximum of an expression
template <Expr ExprType> auto min(ExprType &&expr) {
  using expr_type = std::remove_cvref_t<ExprType>;
  using value_type = typename expr_type::value_type;
  using output_type = ten::scalar<value_type>;
  return unary_expr<expr_type, output_type, functional::min>(expr);
}

/// \fn max
/// Return the maximum of an tensor or an expression
template <Expr ExprType> auto max(ExprType &&expr) {
  using expr_type = std::remove_cvref_t<ExprType>;
  using value_type = typename expr_type::value_type;
  using output_type = ten::scalar<value_type>;
  return unary_expr<expr_type, output_type, functional::max>(expr);
}

/// \fn mean
/// Return the mean of a tensor or an expression
template <Expr ExprType> auto mean(ExprType &&expr) {
  using expr_type = std::remove_cvref_t<ExprType>;
  using value_type = typename expr_type::value_type;
  using output_type = ten::scalar<value_type>;
  return unary_expr<expr_type, output_type, functional::mean>(expr);
}

/// \fn sum
/// Return the sum of a tensor or an expression
template <Expr ExprType> auto sum(ExprType &&expr) {
  using expr_type = std::remove_cvref_t<ExprType>;
  using value_type = typename expr_type::value_type;
  using output_type = ten::scalar<value_type>;
  return unary_expr<expr_type, output_type, functional::sum>(expr);
}

/// \fn cum_sum
/// Return the cumulative sum of a tensor or an expression
template <Expr ExprType> auto cum_sum(ExprType &&expr) {
  using expr_type = std::remove_cvref_t<ExprType>;
  using output_type = typename details::output_type<expr_type>::type;
  return unary_expr<expr_type, output_type, functional::cum_sum>(expr);
}

/// \fn prod
/// Return the maximum of an tensor or an expression
template <Expr ExprType> auto prod(ExprType &&expr) {
  using expr_type = std::remove_cvref_t<ExprType>;
  using value_type = typename expr_type::value_type;
  using output_type = ten::scalar<value_type>;
  return unary_expr<expr_type, output_type, functional::prod>(expr);
}

/// \fn abs
/// Returns the absolute value of a scalar, a tensor or an expression
template <Expr ExprType> auto abs(ExprType &&expr) {
  using expr_type = std::remove_cvref_t<ExprType>;
  using output_type = typename ::ten::details::output_type<expr_type>::type;
  return unary_expr<expr_type, output_type, functional::abs>(expr);
}*/

/// \fn sqrt
/// Returns the square root of a scalar, a tensor or an expression
template <Expr ExprType> auto sqrt(ExprType &&expr) {
  using expr_type = std::remove_cvref_t<ExprType>;
  using input_type = typename ::ten::details::input_type<expr_type>::type;
  using output_type = typename ::ten::details::output_type<expr_type>::type;
  using func_type = ::ten::functional::sqrt<input_type, output_type>;
  func_type *f = new func_type();
  return unary_expr<expr_type, output_type, func_type>(expr, std::move(f));
}

/// \fn sqr
/// Returns the square of an expression
template <Expr ExprType> auto sqr(ExprType &&expr) {
  using expr_type = std::remove_cvref_t<ExprType>;
  using input_type = typename ::ten::details::input_type<expr_type>::type;
  using output_type = typename ::ten::details::output_type<expr_type>::type;
  using func_type = ::ten::functional::sqr<input_type, output_type>;
  func_type *f = new func_type();
  return unary_expr<expr_type, output_type, func_type>(expr, std::move(f));
}

/*
/// \fn sin
/// Returns the sine of a scalar, a tensor or an expression
template <Expr ExprType> auto sin(ExprType &&expr) {
  using expr_type = std::remove_cvref_t<ExprType>;
  using output_type = typename ::ten::details::output_type<expr_type>::type;
  return unary_expr<expr_type, output_type, functional::sin>(expr);
}

/// \fn sinh
/// Hyperbolic sine
template <Expr ExprType> auto sinh(ExprType &&expr) {
  using expr_type = std::remove_cvref_t<ExprType>;
  using output_type = typename ::ten::details::output_type<expr_type>::type;
  return unary_expr<expr_type, output_type, functional::sinh>(expr);
}

/// \fn asin
/// Arc sine
template <Expr ExprType> auto asin(ExprType &&expr) {
  using expr_type = std::remove_cvref_t<ExprType>;
  using output_type = typename ::ten::details::output_type<expr_type>::type;
  return unary_expr<expr_type, output_type, functional::asin>(expr);
}

/// \fn cos
/// Returns the cosine of a scalar, a tensor or an expression
template <Expr ExprType> auto cos(ExprType &&expr) {
  using expr_type = std::remove_cvref_t<ExprType>;
  using output_type = typename ::ten::details::output_type<expr_type>::type;
  return unary_expr<expr_type, output_type, functional::cos>(expr);
}

/// \fn acos
template <Expr ExprType> auto acos(ExprType &&expr) {
  using expr_type = std::remove_cvref_t<ExprType>;
  using output_type = typename ::ten::details::output_type<expr_type>::type;
  return unary_expr<expr_type, output_type, functional::acos>(expr);
}

/// \fn cosh
template <Expr ExprType> auto cosh(ExprType &&expr) {
  using expr_type = std::remove_cvref_t<ExprType>;
  using output_type = typename ::ten::details::output_type<expr_type>::type;
  return unary_expr<expr_type, output_type, functional::cosh>(expr);
}

/// \fn tan
/// Returns the tangent of a scalar, a tensor or an expression
template <Expr ExprType> auto tan(ExprType &&expr) {
  using expr_type = std::remove_cvref_t<ExprType>;
  using output_type = typename ::ten::details::output_type<expr_type>::type;
  return unary_expr<expr_type, output_type, functional::tan>(expr);
}

/// \fn atan
template <Expr ExprType> auto atan(ExprType &&expr) {
  using expr_type = std::remove_cvref_t<ExprType>;
  using output_type = typename ::ten::details::output_type<expr_type>::type;
  return unary_expr<expr_type, output_type, functional::atan>(expr);
}

/// \fn tanh
template <Expr ExprType> auto tanh(ExprType &&expr) {
  using expr_type = std::remove_cvref_t<ExprType>;
  using output_type = typename ::ten::details::output_type<expr_type>::type;
  return unary_expr<expr_type, output_type, functional::tanh>(expr);
}

/// \fn exp
/// Returns the tangent of a scalar, a tensor or an expression
template <Expr ExprType> auto exp(ExprType &&expr) {
  using expr_type = std::remove_cvref_t<ExprType>;
  using output_type = typename ::ten::details::output_type<expr_type>::type;
  return unary_expr<expr_type, output_type, functional::exp>(expr);
}

/// \fn log
/// Returns the tangent of a scalar, a tensor or an expression
template <Expr ExprType> auto log(ExprType &&expr) {
  using expr_type = std::remove_cvref_t<ExprType>;
  using output_type = typename ::ten::details::output_type<expr_type>::type;
  return unary_expr<expr_type, output_type, functional::log>(expr);
}

/// \fn log10
template <Expr ExprType> auto log10(ExprType &&expr) {
  using expr_type = std::remove_cvref_t<ExprType>;
  using output_type = typename ::ten::details::output_type<expr_type>::type;
  return unary_expr<expr_type, output_type, functional::log10>(expr);
}

/// \fn floor
template <Expr ExprType> auto floor(ExprType &&expr) {
  using expr_type = std::remove_cvref_t<ExprType>;
  using output_type = typename ::ten::details::output_type<expr_type>::type;
  return unary_expr<expr_type, output_type, functional::floor>(expr);
}

/// \fn ceil
template <Expr ExprType> auto ceil(ExprType &&expr) {
  using expr_type = std::remove_cvref_t<ExprType>;
  using output_type = typename ::ten::details::output_type<expr_type>::type;
  return unary_expr<expr_type, output_type, functional::ceil>(expr);
}*/

////////////////////////////////////////////////////////////////////////////////
// TODO Parameteric functions

/*
/// \fn pow
/// Power
template <Expr ExprType, class T = float> auto pow(ExprType &&expr, T n) {
  using expr_type = std::remove_cvref_t<ExprType>;
  using output_type = typename ::ten::details::output_type<expr_type>::type;
  return unary_expr<expr_type, output_type, functional::pow>(expr, n);
}*/

////////////////////////////////////////////////////////////////////////////////
// TODO Neural networks (activation functions and layers)

/*
template <Expr ExprType> auto relu(ExprType &&expr) {
  using expr_type = std::remove_cvref_t<ExprType>;
  using output_type = typename ::ten::details::output_type<expr_type>::type;
  return unary_expr<expr_type, output_type, functional::relu>(expr);
}

template <Expr ExprType> auto leaky_relu(ExprType &&expr) {
  using expr_type = std::remove_cvref_t<ExprType>;
  using output_type = typename ::ten::details::output_type<expr_type>::type;
  return unary_expr<expr_type, output_type, functional::leaky_relu>(expr);
}

template <Expr ExprType> auto sigmoid(ExprType &&expr) {
  using expr_type = std::remove_cvref_t<ExprType>;
  using output_type = typename ::ten::details::output_type<expr_type>::type;
  return unary_expr<expr_type, output_type, functional::sigmoid>(expr);
}

template <Expr LeftExpr, Expr RightExpr>
auto mse(LeftExpr &&left, RightExpr &&right) {
  using left_expr_type = std::remove_cvref_t<LeftExpr>;
  using right_expr_type = std::remove_cvref_t<RightExpr>;
  using left_value_type = left_expr_type::value_type;
  using right_value_type = left_expr_type::value_type;
  static_assert(std::is_same_v<left_value_type, right_value_type>);
  using output_type = ten::scalar<left_value_type>;
  return binary_expr<left_expr_type, right_expr_type, output_type,
                     functional::mse>(left, right);
}*/

////////////////////////////////////////////////////////////////////////////////
// Tensor functions

/// Test if elements of tensor, column or row are close to zero
template <class T>
  requires(::ten::is_tensor_v<T> || ::ten::is_column_v<T> || ::ten::is_row_v<T>)
bool all_close(const T &t, double eps) {
  bool res = true;
  for (size_t i = 0; i < t.size(); i++) {
    if (std::abs(t[i]) >= eps) {
      res = false;
      break;
    }
  }
  return res;
}

// Returns a tensor like x with same shape, format, order and requires_grad
// value
template <class T>
tensor<T> like(const tensor<T> &x, std::optional<bool> requires_grad) {
  if (requires_grad.has_value()) {
    return tensor<T>(x.shape(), x.format(), requires_grad.value(), x.order());
  } else {
    return tensor<T>(x.shape(), x.format(), x.requires_grad(), x.order());
  }
}

// Returns a scalar like x with same requires_grad value
template <class T>
scalar<T> like(const scalar<T> &x, std::optional<bool> requires_grad) {
  if (requires_grad.has_value()) {
    return scalar<T>(requires_grad.value());
  } else {
    return scalar<T>(x.requires_grad());
  }
}

} // namespace ten

#endif
