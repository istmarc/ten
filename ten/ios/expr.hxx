#ifndef TEN_IOS_EXPR_HXX
#define TEN_IOS_EXPR_HXX

#include <iostream>
#include <ten/types.hxx>
#include <ten/utils.hxx>

namespace ten {

/// Forward declaration
template <class Input, class Output, class Func>
static void print(std::ostream &os,
                  const ::ten::unary_expr<Input, Output, Func> &expr);

template <class X, class Y, class Output, class Func>
static void print(std::ostream &os,
                  const ::ten::binary_expr<X, Y, Output, Func> &expr);

/// print unary_expr
template <class Input, class Output, class Func>
static void print(std::ostream &os,
                  const ::ten::unary_expr<Input, Output, Func> &expr) {
  using expr_type = ::ten::unary_expr<Input, Output, Func>;
  os << "unary_expr(Input: ";
  if constexpr (::ten::is_tensor_v<Input>) {
    os << "tensor<";
  }
  if constexpr (::ten::is_column_v<Input>) {
    os << "column<";
  }
  if constexpr (::ten::is_row_v<Input>) {
    os << "row<";
  }
  if constexpr (::ten::is_tensor_v<Input> || ::ten::is_column_v<Input> ||
                ::ten::is_row_v<Input>) {
    os << ::ten::to_string<typename Input::value_type>();
    os << ">";
  }
  if constexpr (::ten::is_unary_expr<Input>::value) {
    os << expr.input();
  }
  os << ", Output: ";
  if constexpr (::ten::is_scalar<Output>::value) {
    os << "scalar<";
  }
  if constexpr (::ten::is_tensor_v<Output>) {
    os << "tensor<";
  }
  if constexpr (::ten::is_column_v<Output>) {
    os << "column<";
  }
  if constexpr (::ten::is_row_v<Output>) {
    os << "row<";
  }
  os << ::ten::to_string<typename Output::value_type>();
  os << ">, Function: ";
  os << expr_type::func_type::name();
  os << ")";
}

/// print unary_expr
template <class Input, class Output, class Func>
std::ostream &operator<<(std::ostream &os,
                         const ::ten::unary_expr<Input, Output, Func> &expr) {
  print(os, expr);
  return os;
}

/// print binary expr
template <class X, class Y, class Output, class Func>
static void print(std::ostream &os,
                  const ::ten::binary_expr<X, Y, Output, Func> &expr) {
  using expr_type = ::ten::binary_expr<X, Y, Output, Func>;
  // Print the left input
  os << "binary_expr(Left: ";
  if constexpr (::ten::is_tensor_v<X>) {
    os << "tensor<";
  }
  if constexpr (::ten::is_column_v<X>) {
    os << "column<";
  }
  if constexpr (::ten::is_row_v<X>) {
    os << "row<";
  }
  if constexpr (::ten::is_tensor_v<X> || ::ten::is_column_v<X> ||
                ::ten::is_row_v<X>) {
    os << ::ten::to_string<typename X::value_type>();
    os << ">";
  }
  if constexpr (::ten::is_unary_expr<X>::value ||
                ::ten::is_binary_expr<X>::value) {
    print(os, expr.left());
  }
  // print the right input
  os << ", Right: ";
  if constexpr (::ten::is_tensor_v<Y>) {
    os << "tensor<";
  }
  if constexpr (::ten::is_column_v<Y>) {
    os << "column<";
  }
  if constexpr (::ten::is_row_v<Y>) {
    os << "row<";
  }
  if constexpr (::ten::is_tensor_v<Y> || ::ten::is_column_v<Y> ||
                ::ten::is_row_v<Y>) {
    os << ::ten::to_string<typename Y::value_type>();
    os << ">";
  }
  if constexpr (::ten::is_unary_expr<Y>::value ||
                ::ten::is_binary_expr<Y>::value) {
    print(os, expr.right());
  }
  // Print the output
  os << ", Output: ";
  if constexpr (::ten::is_scalar<Output>::value) {
    os << "scalar<";
  }
  if constexpr (::ten::is_tensor_v<Output>) {
    os << "tensor<";
  }
  if constexpr (::ten::is_column_v<Output>) {
    os << "column<";
  }
  if constexpr (::ten::is_row_v<Output>) {
    os << "row<";
  }
  os << ::ten::to_string<typename Output::value_type>();
  // Print the function
  os << ">, Function: ";
  os << expr_type::func_type::name();
  os << ")";
}

/// Print binary_expr
template <class X, class Y, class Output, class Func>
std::ostream &operator<<(std::ostream &os,
                         const ::ten::binary_expr<X, Y, Output, Func> &expr) {
  print(os, expr);
  return os;
}

} // namespace ten

#endif
