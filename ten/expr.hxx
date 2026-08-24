#ifndef TEN_EXPR_HXX
#define TEN_EXPR_HXX

#include <memory>
#include <optional>
#include <type_traits>

#include <ten/functions.hxx>
#include <ten/types.hxx>

#include <ten/functional.hxx>

// #include <ten/autograd.hxx>

namespace ten::details {

// Input type
template <class> struct input_type;

template <class T> struct input_type<scalar<T>> {
  using type = scalar<T>;
};

template <class T> struct input_type<tensor<T>> {
  using type = tensor<T>;
};

// Output type
template <class> struct output_type;

template <class T> struct output_type<scalar<T>> {
  using type = scalar<T>;
};

template <class T> struct output_type<tensor<T>> {
  using type = tensor<T>;
};

// Input type of a unary expr
template <class Input, class Output, class Func>
  requires(ten::is_tensor_v<Input> || ten::is_scalar_v<Input>)
struct input_type<ten::unary_expr<Input, Output, Func>> {
  using type = Input;
};
template <class Input, class Output, class Func>
  requires(ten::is_unary_expr_v<Input> || ten::is_binary_expr_v<Input>)
struct input_type<ten::unary_expr<Input, Output, Func>> {
  using type = typename Input::output_type;
};

// Output type of a unary expr
template <class Input, class Output, class Func>
struct output_type<::ten::unary_expr<Input, Output, Func>> {
  using type = Output;
};

// Output type of a binary expr
template <class X, class Y, class Output, class Func>
struct output_type<ten::binary_expr<X, Y, Output, Func>> {
  using type = Output;
};

/*

template <class Left, class Right, class Output, template <typename...> class F,
          class... Args>
struct input_type<::ten::binary_expr<Left, Right, Output, F, Args...>> {
   using type = typename ::ten::binary_expr<Left, Right, Output, F,
                                            Args...>::output_type;
};*/

/*
template <class Left, class Right, class Output, template <typename...> class F,
          class... Args>
struct output_type<::ten::binary_expr<Left, Right, Output, F, Args...>> {
   using type = typename ::ten::binary_expr<Left, Right, Output, F,
                                            Args...>::output_type;
};

// Input shape
template <Tensor T> static inline auto input_shape(T &t) -> decltype(auto) {
   return t.shape();
}

template <UnaryExpr ExprType>
static inline auto input_shape(ExprType &expr) -> decltype(auto) {
   return expr.value().shape();
}

template <BinaryExpr ExprType>
static inline auto input_shape(ExprType &expr) -> decltype(auto) {
   return expr.value().shape();
}

// Input value
template <Scalar T> static inline auto input_value(T &t) { return t; }

template <Tensor T> static inline auto input_value(T &t) { return t; }

template <UnaryExpr ExprType> static inline auto input_value(ExprType &expr) {
   return expr.value();
}

template <BinaryExpr ExprType> static inline auto input_value(ExprType &expr) {
   return expr.value();
}
*/

// Input node
template <Scalar T> static inline auto input_node(std::shared_ptr<T> &t) {
  return t;
}

template <Tensor T> static inline auto input_node(std::shared_ptr<T> &t) {
  return t;
}

template <UnaryExpr ExprType>
static inline auto input_node(std::shared_ptr<ExprType> &expr) {
  return expr->value_node();
}

template <BinaryExpr ExprType>
static inline auto input_node(std::shared_ptr<ExprType> &expr) {
  return expr->value_node();
}

} // namespace ten::details

namespace ten {

// Unary node
template <class Input, class Output, class Func> struct unary_node {
  using input_type = Input;
  using output_type = Output;
  using Func_type = Func;

  bool _evaluated = false;
  // bool _retain_grad = false;
  Func *_func = nullptr;
  std::shared_ptr<Input> _input = nullptr;
  std::shared_ptr<Output> _value = nullptr;

  /// Return the input
  [[nodiscard]] inline Input &input() const { return *_input.get(); }

  /// Return the input node
  [[nodiscard]] inline std::shared_ptr<Input> input_node() { return _input; }

  /// Return the output
  [[nodiscard]] inline Output &value() const { return *_value.get(); }

  /// Return the std::shared_ptr to the output
  [[nodiscard]] inline std::shared_ptr<Output> value_node() { return _value; }

  // Construct a unary node from input and function
  unary_node(input_type &inp, Func *f) noexcept
      : _func(f), _input(std::make_shared<input_type>(inp)) {}

  ~unary_node() { delete _func; }
};

// \class unary_expr
// Apply a function to a tensor, column or row
// template <class Input, class Output, template <typename...> class Func,
// typename... Args>
template <class Input, class Output, class Func> class unary_expr : ten::expr {
public:
  using node_type = ::ten::unary_node<Input, Output, Func>;

  using input_type = node_type::input_type;
  using output_type = node_type::output_type;
  using func_type = Func;

  using value_type = typename input_type::value_type;

  using expr_type = unary_expr<Input, Output, Func>;

private:
  std::shared_ptr<node_type> _node = nullptr;

public:
  unary_expr() {}
  ~unary_expr() {}

  /// Construct a unary expr
  unary_expr(input_type &inp, Func *f) noexcept {
    _node = std::make_shared<node_type>(inp, f);
  }

  /// Requires gradient
  [[nodiscard]] inline bool requires_grad() {
    return _node->_input.get()->requires_grad();
  }

  /// Retain the gradient
  [[nodiscard]] inline bool has_retain_grad() { return _node->_retain_grad; }

  // Retain the gradient for this node
  void retain_grad() { _node->_retain_grad = true; }

  /// Return the input
  [[nodiscard]] inline Input &input() { return *_node->_input.get(); }

  /// Return the input node
  [[nodiscard]] inline std::shared_ptr<Input> input_node() {
    return _node->_input;
  }

  /// Returns whether the expression is evaluated
  [[nodiscard]] inline bool evaluated() { return _node->_evaluated; }

  /// Returns the evaluated expression of type ten::scalar or ten::tensor
  [[nodiscard]] inline Output &value() { return *_node->_value.get(); }

  /// Return the node to the evaluated expression of type ten::scalar or
  /// ten::tensor
  [[nodiscard]] inline std::shared_ptr<Output> &value_node() {
    return _node->_value;
  }

  [[nodiscard]] auto grad() noexcept -> Output {
    return _node->_value.get()->grad();
  }

  /// Evaluate the expression
  [[maybe_unused]] auto eval() noexcept -> Output {
    if (evaluated())
      return _node->value();

    // Evaluate input
    if constexpr (::ten::is_unary_expr<Input>::value ||
                  ::ten::is_binary_expr<Input>::value) {
      if (!_node->_input->evaluated()) {
        _node->_input->eval();
      }
    }

    // Evaluate
    if constexpr (::ten::is_scalar<Input>::value || ::ten::is_tensor_v<Input> ||
                  ::ten::is_diagonal_v<Input> ||
                  ::ten::is_sparse_tensor_v<Input>) {
      _node->_func->call(_node->_input, _node->_value);
    }

    if constexpr (::ten::is_unary_expr_v<Input> ||
                  ::ten::is_binary_expr_v<Input>) {
      auto input_node = _node->_input->value_node();
      _node->_func->call(input_node, _node->_value);
    }

    // This expression has been evaluated
    _node->_evaluated = true;

    return _node->value();
  }
};

template <class X, class Y, class Output, class Func> struct binary_node {
  /// Left input type
  using left_type = X;

  /// Right input type
  using right_type = Y;

  /// Output type
  using output_type = Output;

  /// Function type
  using func_type = Func;

  /// Flag for evaluated expression
  bool _evaluated = false;
  // Retain the gradient
  bool _retain_grad = false;
  /// Function
  Func *_func = nullptr;
  /// Left input
  std::shared_ptr<X> _left = nullptr;
  /// Right input
  std::shared_ptr<Y> _right = nullptr;
  /// Output value
  std::shared_ptr<Output> _value = nullptr;

  /// Returns the left input
  [[nodiscard]] X &left() const { return *_left.get(); }

  /// Returns a std::shared_ptr to the left input
  [[nodiscard]] std::shared_ptr<X> left_node() const { return _left; }

  // Returns the right input
  [[nodiscard]] Y &right() const { return *_right.get(); }

  /// Returns a std::shared_ptr to the right input
  [[nodiscard]] std::shared_ptr<Y> right_node() const { return _right; }

  /// return the output
  [[nodiscard]] Output &value() const { return *_value.get(); }

  /// return a std::shared_ptr to the output
  [[nodiscard]] std::shared_ptr<Output> value_node() const { return _value; }

  /// Construct a binary node
  binary_node(X &l, Y &r, Func *f) noexcept
      : _left(std::make_shared<X>(l)), _right(std::make_shared<Y>(r)),
        _func(f) {}

  ~binary_node() { delete _func; }
};

// \class binary_expr
// Binary expresion
// Left and Right can be scalar, tensor, row, column, unary_expr or binary_expr
template <class X, class Y, class Output, class Func>
class binary_expr : ten::expr {
public:
  /// Left input type
  using left_type = X;

  /// Right input type
  using right_type = Y;

  /// Output type
  using output_type = Output;

  using func_type = Func;

  using node_type = binary_node<X, Y, Output, Func>;

  using value_type = output_type::value_type;

  using expr_type = binary_expr<X, Y, Output, Func>;

private:
  std::shared_ptr<node_type> _node = nullptr;

public:
  binary_expr() {}

  ~binary_expr() {}

  /// Construct a binary expr
  binary_expr(X &l, Y &r, Func *f) noexcept {
    _node = std::make_shared<node_type>(l, r, f);
  }

  /// Requires gradient
  [[nodiscard]] inline bool requires_left_grad() {
    return _node->_left.requires_grad();
  }
  [[nodiscard]] inline bool requires_right_grad() {
    return _node->_right.requires_grad();
  }

  // Retain gradient
  void retain_grad() { _node->_retain_grad = true; }

  /// Returns whether the node has retain_grad
  [[nodiscard]] inline bool has_retain_grad() const {
    return _node->_retain_grad;
  }

  /// Returns the left input
  [[nodiscard]] X &left() const { return *(_node->_left.get()); }

  /// Returns a std::shared_ptr to the left input
  [[nodiscard]] std::shared_ptr<X> &left_node() const { return _node->_left; }

  // Returns the right input
  [[nodiscard]] Y &right() const { return *(_node->_right.get()); }

  /// Returns a std::shared_ptr to the right input
  [[nodiscard]] std::shared_ptr<Y> right_node() const { return _node->_right; }

  // Left gradient
  /*
  [[nodiscard]] auto left_grad() const {
     if constexpr (::ten::is_tensor_v<Left> || ::ten::is_scalar_v<Left>) {
        return _node->_left->grad();
     } else {
        return _node->_left->value().grad();
     }
  }

  // Right gradient
  [[nodiscard]] auto right_grad() const {
     if constexpr (::ten::is_tensor_v<Right> || ::ten::is_scalar_v<Right>) {
        return _node->_right->grad();
     } else {
        return _node->_right->value().grad();
     }
  }*/

  /// Returns whether the expression is evaluated
  [[nodiscard]] inline bool evaluated() const { return _node->_evaluated; }

  /// Returns the the evaluated expression of type ten::scalar or ten::tensor
  [[nodiscard]] Output &value() { return *(_node->_value.get()); }

  /// Returns the the std::shared_ptr to the evaluated expression of type
  /// ten::scalar or ten::tensor
  [[nodiscard]] std::shared_ptr<Output> value_node() { return _node->_value; }

  /// Evaluate a binary expression
  /// If the input expression has not been evaluated, it will evaluate it
  /// recursively before evaluating this expression
  [[maybe_unused]] auto eval() noexcept -> output_type {
    if (evaluated())
      return _node->value();

    // Evaluate the left expr
    using left_type = std::remove_cvref_t<X>;
    if constexpr (::ten::is_unary_expr_v<left_type> ||
                  ::ten::is_binary_expr_v<left_type>) {
      if (!_node->_left->evaluated()) {
        _node->_left->eval();
      }
    }
    // Evaluate the right expr
    using right_type = std::remove_cvref_t<Y>;
    if constexpr (::ten::is_unary_expr_v<right_type> ||
                  ::ten::is_binary_expr_v<right_type>) {
      if (!_node->_right->evaluated()) {
        _node->_right->eval();
      }
    }

    // Call the function
    auto left_node = ::ten::details::input_node(_node->_left);
    auto right_node = ::ten::details::input_node(_node->_right);
    _node->_func->call(left_node, right_node, _node->_value);

    // This expression has been evaluated
    _node->_evaluated = true;

    return _node->value();
  }
};

} // namespace ten

#endif
