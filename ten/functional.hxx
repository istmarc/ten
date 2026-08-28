#ifndef TEN_FUNCTIONAL_HXX
#define TEN_FUNCTIONAL_HXX

#include "kernels/binary_ops.hxx"
#include <cmath>
#include <initializer_list>
#include <memory>
#include <type_traits>

#include <ten/kernels/host>
#include <ten/types.hxx>
#include <ten/utils.hxx>

namespace ten {
enum class binary_operation;
}

namespace ten::details {
template <class, class> struct common_type;

// Tensor, Tensor
template <Tensor X, Tensor Y> struct common_type<X, Y> {
  using value_type =
      std::common_type_t<typename X::value_type, typename Y::value_type>;
  using type = tensor<value_type>;
};

// Scalar, Tensor
template <Scalar X, Tensor Y> struct common_type<X, Y> {
  using value_type =
      std::common_type_t<typename X::value_type, typename Y::value_type>;
  using type = tensor<value_type>;
};

// Tensor, Scalar
template <Tensor X, Scalar Y> struct common_type<X, Y> {
  using value_type =
      std::common_type_t<typename X::value_type, typename Y::value_type>;
  using type = tensor<value_type>;
};

template <class X, class Y>
using common_type_t = typename common_type<X, Y>::type;

/// Multiplication result
template <class, class> struct mul_result;

template <class X, class Y>
using mul_result_t = typename mul_result<X, Y>::type;

// scalar * scalar
template <Scalar X, Scalar Y> struct mul_result<X, Y> {
  using value_type =
      std::common_type_t<typename X::value_type, typename Y::value_type>;
  using type = ::ten::scalar<value_type>;
};

// tensor * tensor
template <Tensor X, Tensor Y> struct mul_result<X, Y> {
  using value_type =
      std::common_type_t<typename X::value_type, typename Y::value_type>;
  using type = tensor<value_type>;
};

// scalar * tensor
template <Scalar X, Tensor Y> struct mul_result<X, Y> {
  using type = Y;
};

// tensor * scalar
template <Tensor X, Scalar Y> struct mul_result<X, Y> {
  using type = X;
};

/// scalar * unary_expr
template <Scalar X, UnaryExpr Y> struct mul_result<X, Y> {
  using type = std::remove_cvref_t<Y>::output_type;
};

/// scalar * binary_expr
template <Scalar X, BinaryExpr Y> struct mul_result<X, Y> {
  using type = std::remove_cvref_t<Y>::output_type;
};

/// unary_expr * scalar
template <UnaryExpr X, Scalar Y> struct mul_result<X, Y> {
  using type = std::remove_cvref_t<X>::output_type;
};

/// binary_expr * scalar
template <BinaryExpr X, Scalar Y> struct mul_result<X, Y> {
  using type = std::remove_cvref_t<X>::output_type;
};

/// unary_expr * tensor
template <UnaryExpr X, Tensor Y> struct mul_result<X, Y> {
  using evaluated_type = std::remove_cvref_t<X>::output_type;
  using type = mul_result_t<evaluated_type, Y>;
};

/// binary_expr * tensor
template <BinaryExpr X, Tensor Y> struct mul_result<X, Y> {
  using evaluated_type = std::remove_cvref_t<X>::output_type;
  using type = mul_result_t<evaluated_type, Y>;
};

/// tensor * unary_expr
template <Tensor X, UnaryExpr Y> struct mul_result<X, Y> {
  using evaluated_type = std::remove_cvref_t<Y>::output_type;
  using type = mul_result_t<evaluated_type, X>;
};

/// tensor * binary_expr
template <Tensor X, BinaryExpr Y> struct mul_result<X, Y> {
  using evaluated_type = std::remove_cvref_t<Y>::output_type;
  using type = mul_result_t<evaluated_type, X>;
};

/*
// TODO reshape result
template <class X> struct reshape_result {
  using type = tensor<typename X::value_type>;
};

// dynamic transpose result
template <class X> struct transpose_result {
  using type = tensor<typename X::value_type>;
};*/

} // namespace ten::details

namespace ten::functional {
////////////////////////////////////////////////////////////////////////////////
// Functions types
struct func {
  virtual ~func() {}
};

// template <bool params = false, bool with_shape = false> struct func : public
// abstract_func {};

/*template <class Func> struct has_params {
   static constexpr bool value = std::is_base_of_v<func<true, true>, Func> ||
                                 std::is_base_of_v<func<true, false>, Func>;
};

template <class Func> struct has_shape {
   static constexpr bool value = std::is_base_of_v<func<true, true>, Func> ||
                                 std::is_base_of_v<func<false, true>, Func>;
};*/

////////////////////////////////////////////////////////////////////////////////
// Unary functions

/// Square root
template <class X, class Y>
  requires(
      ((::ten::is_tensor_v<X> || ::ten::is_column_v<X> || ::ten::is_row_v<X>) &&
       ::ten::is_tensor_v<Y>) ||
      (::ten::is_scalar_v<X> && ::ten::is_scalar_v<Y>))
struct sqrt : func {
  static constexpr std::string name() { return std::string("sqrt"); }

  using value_type = X::value_type;
  using output_value_type = Y::value_type;
  using output_type = Y;

  void call(std::shared_ptr<X> &x, std::shared_ptr<Y> &y) {
    if constexpr (::ten::is_scalar_v<X> && ::ten::is_scalar_v<Y>) {
      if (!y) {
        y = std::make_shared<Y>();
      }
      y.value() = std::sqrt(x.value());
    }
    if constexpr (::ten::is_tensor_v<X> || ::ten::is_column_v<X> ||
                  ::ten::is_row_v<X>) {
      if (!y) {
        y = std::make_shared<Y>(x->shape(), x->format(), x->requires_grad(),
                                x->storage_order());
      }
      for (size_t i = 0; i < x->size(); i++) {
        (*y.get())[i] =
            std::sqrt(static_cast<output_value_type>((*x.get())[i]));
      }
    }
  }

  /*
  template <class Gradient>
     requires(::ten::is_tensor_v<X>)
  void gradient(X &x, Gradient &grad) {
     if (x.shape() != grad.shape()) {
        std::cerr << "ten::functional::sqrt gradient incompatible shapes\n";
     } else {
        for (size_t i = 0; i < x.size(); i++) {
           grad[i] = 1 / (2 * std::sqrt(x[i]));
        }
     }
  }*/
};

/// Square
template <class X, class Y>
  requires(((::ten::is_tensor<X>::value || ::ten::is_column<X>::value ||
             ::ten::is_row<X>::value) &&
            ::ten::is_tensor<Y>::value) ||
           (::ten::is_scalar_v<X> && ::ten::is_scalar_v<Y>))
struct sqr : func {
  static constexpr std::string name() { return std::string("sqr"); }

  using value_type = X::value_type;
  using output_value_type = Y::value_type;
  using output_type = Y;

  void call(std::shared_ptr<X> &x, std::shared_ptr<Y> &y) {
    if constexpr (::ten::is_scalar<X>::value) {
      if (!y) {
        y = std::make_shared<Y>();
      }
      y.value() = x.value() * x.value();
    }
    if constexpr (::ten::is_tensor_v<Y>) {
      if (!y) {
        y = std::make_shared<Y>(x->shape(), x->format(), x->requires_grad(),
                                x->storage_order());
      }
      auto xarr = *x.get();
      auto yarr = *y.get();
      ::ten::kernels::binary_ops<::ten::binary_operation::mul>(xarr, xarr,
                                                               yarr);
    }
  }

  /*
  template <class Gradient>
     requires(::ten::is_scalar_v<X> && ::ten::is_scalar_v<Gradient>)
  void gradient(X &x, Gradient &grad) {
     grad->value() = 2 * x->value();
  }

  template <class Gradient>
     requires(::ten::is_tensor_v<X> && ::ten::is_tensor_v<Gradient>)
  void gradient(X &x, Gradient &grad) {
     if (x.shape() != grad.shape()) {
        std::cerr << "ten::functional::sqr gradient incompatible shapes.\n";
     } else {
        for (size_t i = 0; i < x.size(); i++) {
           grad[i] = 2 * x[i];
        }
     }
  }*/
};

/// absolute value
template <class X, class Y>
  requires(
      ((::ten::is_tensor_v<X> || ::ten::is_column_v<X> || ::ten::is_row_v<X>) &&
       ::ten::is_tensor_v<Y>) ||
      (::ten::is_scalar_v<X> && ::ten::is_scalar_v<Y>))
struct abs : func {
  static constexpr std::string name() { return std::string("abs"); }

  using value_type = X::value_type;
  using output_value_type = Y::value_type;
  using output_type = Y;

  void call(std::shared_ptr<X> &x, std::shared_ptr<Y> &y) {
    if constexpr (::ten::is_scalar_v<Y>) {
      if (!y) {
        y = std::make_shared<Y>();
      }
      y.value() = std::abs(x.value());
    }
    if constexpr (::ten::is_tensor_v<Y>) {
      if (!y) {
        y = std::make_shared<Y>(x->shape(), x->format(), x->requires_grad(),
                                x->storage_order());
      }
      for (size_t i = 0; i < x->size(); i++) {
        (*y.get())[i] = std::abs(static_cast<value_type>((*x.get())[i]));
      }
    }
  }

  /*
     template <class Gradient>
        requires(::ten::is_scalar_v<X> && ::ten::is_scalar_v<Gradient>)
     void gradient(X &x, Gradient &grad) {
        if (x.value() >= 0) {
           grad.value() = 1;
        } else {
           grad.value() = -1;
        }
     }

     template <class Gradient>
        requires(::ten::is_tensor_v<X> && ::ten::is_tensor_v<Gradient>)
     void gradient(X &x, Gradient &grad) {
        if (x.shape() != grad.shape()) {
           std::cerr << "ten::functional::abs gradient incompatible shape\n";
        } else {
           for (size_t i = 0; i < x.size(); i++) {
              if (x[i] >= 0) {
                 grad[i] = 1;
              } else {
                 grad[i] = -1;
              }
           }
        }
     }*/
};

/// Power
/// FIXME power of complex tensor
template <class X, class Y>
  requires(((ten::is_tensor<X>::value || ten::is_column<X>::value ||
             ten::is_row<X>::value) &&
            ::ten::is_tensor<Y>::value) ||
           (::ten::is_scalar_v<X> && ::ten::is_scalar_v<Y>))
struct pow : func {
  static constexpr std::string name() { return std::string("pow"); }
  using value_type = X::value_type;

private:
  value_type _n;

public:
  using output_value_type = Y::value_type;
  using output_type = Y;

  explicit pow(value_type n) : _n(n) {}

  void call(std::shared_ptr<X> &x, std::shared_ptr<Y> &y) const {
    using value_type = typename Y::value_type;
    if constexpr (::ten::is_scalar_v<Y>) {
      if (!y) {
        y = std::make_shared<Y>();
      }
      y.value() = std::pow(x.value(), _n);
    }
    if constexpr (::ten::is_tensor_v<Y>) {
      if (!y) {
        y = std::make_shared<Y>(x->shape(), x->format(), x->requires_grad(),
                                x->storage_order());
      }
      for (std::size_t i = 0; i < x->size(); i++) {
        (*y.get())[i] = std::pow((*x.get())[i], _n);
      }
    }
  }

  /*
  template <class Gradient>
     requires(::ten::is_scalar_v<X> && ::ten::is_scalar_v<Gradient>)
  void gradient(X &x, Gradient &grad) {
     grad.value() =
         static_cast<output_value_type>(_n) * std::pow(x.value(), _n - 1);
  }

  template <class Gradient>
     requires(::ten::is_tensor_v<X> && ::ten::is_tensor_v<Gradient>)
  void gradient(X &x, Gradient &grad) {
     if (x.shape() != grad.shape()) {
        std::cerr << "ten::functional::pow gradient incompatible shapes.\n";
     } else {
        for (size_t i = 0; i < x.size(); i++) {
           grad[i] =
               static_cast<output_value_type>(_n) * std::pow(x[i], _n - 1);
        }
     }
  }*/
};

/// Minimum
template <class X, class Y>
  requires((::ten::is_tensor<X>::value || ::ten::is_column<X>::value ||
            ::ten::is_row<X>::value) &&
           ::ten::is_scalar<Y>::value)
struct min : func {
  static constexpr std::string name() { return std::string("min"); }

  using output_type = Y;

  void call(std::shared_ptr<X> &x, std::shared_ptr<Y> &y) {
    if (!y) {
      y = std::make_shared<Y>(x->requires_grad());
    }
    using type = typename X::value_type;
    type res = (*x.get())[0];
    for (size_t i = 1; i < x->size(); i++) {
      res = std::min((*x.get())[i], res);
    }
    y->value() = res;
  }
};

/// Maximum
template <class X, class Y>
  requires((::ten::is_tensor<X>::value || ::ten::is_column<X>::value ||
            ::ten::is_row<X>::value) &&
           ::ten::is_scalar<Y>::value)
struct max : func {
  static constexpr std::string name() { return std::string("max"); }

  using output_type = Y;

  void call(std::shared_ptr<X> &x, std::shared_ptr<Y> &y) {
    if (!y) {
      y = std::make_shared<Y>(x->requires_grad());
    }
    using type = typename Y::value_type;
    type res = (*x.get())[0];
    for (size_t i = 1; i < x->size(); i++) {
      res = std::max(static_cast<type>((*x.get())[i]), res);
    }
    y->value() = res;
  }
};

/// Mean
template <class X, class Y>
  requires((::ten::is_tensor<X>::value || ::ten::is_column<X>::value ||
            ::ten::is_row<X>::value) &&
           ::ten::is_scalar<Y>::value)
struct mean : func {
  static constexpr std::string name() { return std::string("mean"); }

  using output_type = Y;

  void call(std::shared_ptr<X> &x, std::shared_ptr<Y> &y) {
    if (!y) {
      y = std::make_shared<Y>(x->requires_grad());
    }
    using type = typename Y::value_type;
    type res = type(0);
    for (size_t i = 0; i < x->size(); i++) {
      res += static_cast<type>((*x.get())[i]);
    }
    y->value() = res / x->size();
  }

  /*
  template <class Gradient>
     requires(::ten::is_tensor_v<X> && ::ten::is_tensor_v<Gradient>)
  void gradient(X &x, Gradient &grad) {
     using T = X::value_type;
     for (size_t i = 0; i < x.size(); i++) {
        grad[i] = T(1) / T(x.size());
     }
  }*/
};

namespace details {

/// Mean
template <typename T> T mean(const ten::tensor<T> &x) {
  T m = x[0];
  for (std::size_t i = 1; i < x.size(); i++) {
    m += x[i];
  }
  return m / x.size();
}

/// Sum along an axis
template <typename T>
void sum_axis(ten::tensor<T> &x, ten::tensor<T> &s, std::size_t axis) {
  const std::size_t size = s.size();
  const std::size_t rank = x.rank();
  if (rank == 2) {
    if (axis == 0) {
      for (std::size_t k = 0; k < size; k++) {
        for (std::size_t i = 0; i < x.dim(0); i++) {
          s[k] += x(i, k);
        }
      }
    } else if (axis == 1) {
      for (std::size_t k = 0; k < size; k++) {
        for (std::size_t j = 0; j < x.dim(1); j++) {
          s[k] += x(k, j);
        }
      }
    } else {
      std::cerr << "Sum along an axis not supported.\n";
    }
  } else {
    std::cerr << "Sum along an axis not supported for rank " << rank
              << " tensor.\n";
  }
}
} // namespace details

/// Sum along an axis
template <class X, class Y>
  requires(
      (::ten::is_tensor_v<X> || ::ten::is_column_v<X> || ::ten::is_row_v<X>) &&
      ::ten::is_tensor_v<Y>)
struct sum_axis : func {
private:
  std::size_t _axis;

public:
  using value_type = typename X::value_type;
  using output_type = Y;
  static constexpr std::string name() { return std::string("mean"); }

  sum_axis(std::size_t axis) : _axis(axis) {}

  void call(std::shared_ptr<X> &x, std::shared_ptr<Y> &y) {
    std::size_t size = 1;
    for (std::size_t i = 0; i < x->rank(); i++) {
      if (i != _axis) {
        size *= x->dim(i);
      }
    }
    if (!y) {
      const std::vector<std::size_t> shape = {size};
      y = std::make_shared<Y>(shape, ten::storage_format::dense,
                              x->requires_grad(), x->storage_order());
    }
    for (std::size_t i = 0; i < size; i++) {
      (*y.get())[i] = 0;
    }
    details::sum_axis(*x.get(), *y.get(), _axis);
  }
};

/// Mean along an axis
template <class X, class Y>
  requires(
      (::ten::is_tensor_v<X> || ::ten::is_column_v<X> || ::ten::is_row_v<X>) &&
      ::ten::is_tensor_v<Y>)
struct mean_axis : func {
private:
  std::size_t _axis;

public:
  using value_type = typename X::value_type;
  using output_type = Y;
  static constexpr std::string name() { return std::string("mean"); }

  mean_axis(std::size_t axis) : _axis(axis) {}

  void call(std::shared_ptr<X> &x, std::shared_ptr<Y> &y) {
    std::size_t size = 1;
    for (std::size_t i = 0; i < x->rank(); i++) {
      if (i != _axis) {
        size *= x->dim(i);
      }
    }
    if (!y) {
      const std::vector<std::size_t> shape = {size};
      y = std::make_shared<Y>(shape, ten::storage_format::dense,
                              x->requires_grad(), x->storage_order());
    }
    for (std::size_t i = 0; i < size; i++) {
      (*y.get())[i] = 0;
    }
    details::sum_axis(*x.get(), *y.get(), _axis);
    // Mean
    for (std::size_t i = 0; i < size; i++) {
      (*y.get())[i] /= x->dim(_axis);
    }
  }
};

/// Sum
template <class X, class Y>
  requires((::ten::is_tensor<X>::value || ::ten::is_column<X>::value ||
            ::ten::is_row<X>::value) &&
           ::ten::is_scalar<Y>::value)
struct sum : func {
  static constexpr std::string name() { return std::string("sum"); }

  using output_type = Y;

  void call(std::shared_ptr<X> &x, std::shared_ptr<Y> &y) {
    if (!y) {
      y = std::make_shared<Y>(x->requires_grad());
    }
    using type = typename Y::value_type;
    type res = type(0);
    for (size_t i = 0; i < x->size(); i++) {
      res += static_cast<type>((*x.get())[i]);
    }
    y->value() = res;
  }

  /*
  template <class Gradient>
     requires(::ten::is_tensor_v<X> && ::ten::is_tensor_v<Gradient>)
  void gradient(X &x, Gradient &grad) {
     if (x.shape() != grad.shape()) {
        std::cerr << "ten::fucntional::sum gradient incompatible shapes.\n";
     } else {
        for (size_t i = 0; i < x.size(); i++) {
           grad[i] = 1;
        }
     }
  }*/
};

/// Cumulative sum
template <class X, class Y>
  requires((::ten::is_tensor<X>::value || ::ten::is_column<X>::value ||
            ::ten::is_row<X>::value) &&
           ::ten::is_tensor<Y>::value)
struct cum_sum : func {
  static constexpr std::string name() { return std::string("cum_sum"); }

  using output_type = Y;

  void call(std::shared_ptr<X> &x, std::shared_ptr<Y> &y) {
    if (!y) {
      y = std::make_shared<Y>(x->shape(), x->format(), x->requires_grad(),
                              x->storage_order());
    }
    using type = typename Y::value_type;
    auto yarr = *(y.get());
    auto xarr = *(x.get());
    yarr[0] = static_cast<type>(xarr[0]);
    for (size_t i = 1; i < x->size(); i++) {
      yarr[i] = static_cast<type>(xarr[i]) + yarr[i - 1];
    }
  }
};

/// Prod
template <class X, class Y>
  requires((::ten::is_tensor<X>::value || ::ten::is_column<X>::value ||
            ::ten::is_row<X>::value) &&
           ::ten::is_scalar<Y>::value)
struct prod : func {
  static constexpr std::string name() { return std::string("prod"); }

  using output_type = Y;

  void call(std::shared_ptr<X> &x, std::shared_ptr<Y> &y) {
    if (!y) {
      y = std::make_shared<Y>(x->requires_grad());
    }
    using type = typename Y::value_type;
    type res = type(1);
    for (size_t i = 0; i < x->size(); i++) {
      res *= static_cast<type>((*x.get())[i]);
    }
    y->value() = res;
  }
};

/// Std
template <class X, class Y>
  requires((::ten::is_tensor<X>::value || ::ten::is_column<X>::value ||
            ::ten::is_row<X>::value) &&
           ::ten::is_scalar<Y>::value)
struct std : func {
private:
  bool _biased;

public:
  static constexpr ::std::string name() { return ::std::string("std"); }

  using output_type = Y;

  std(bool biased) : _biased(biased) {}

  void call(::std::shared_ptr<X> &x, ::std::shared_ptr<Y> &y) {
    if (!y) {
      y = ::std::make_shared<Y>(x->requires_grad());
    }
    using T = typename Y::value_type;
    T m = details::mean(*x.get());
    y->value() = T(0);
    for (size_t i = 0; i < x->size(); i++) {
      T diff = (*x.get())[i] - m;
      y->value() += diff * diff;
    }
    if (_biased) {
      y->value() = ::std::sqrt(y->value() / x->size());
    } else {
      y->value() = ::std::sqrt(y->value() / (x->size() - 1));
    }
  }
};

/// Variance
template <class X, class Y>
  requires((::ten::is_tensor<X>::value || ::ten::is_column<X>::value ||
            ::ten::is_row<X>::value) &&
           ::ten::is_scalar<Y>::value)
struct var : func {
private:
  bool _biased;

public:
  static constexpr ::std::string name() { return ::std::string("var"); }

  using output_type = Y;

  var(bool biased) : _biased(biased) {}

  void call(::std::shared_ptr<X> &x, ::std::shared_ptr<Y> &y) {
    if (!y) {
      y = ::std::make_shared<Y>(x->requires_grad());
    }
    using T = typename Y::value_type;
    T m = details::mean(*x.get());
    y->value() = T(0);
    for (size_t i = 0; i < x->size(); i++) {
      T diff = (*x.get())[i] - m;
      y->value() += diff * diff;
    }
    if (_biased) {
      y->value() = y->value() / x->size();
    } else {
      y->value() = y->value() / (x->size() - 1);
    }
  }
};

/// Sine
template <class X, class Y>
  requires(((::ten::is_tensor<X>::value || ::ten::is_column<X>::value ||
             ::ten::is_row<X>::value) &&
            ::ten::is_tensor<Y>::value) ||
           (::ten::is_scalar_v<X> && ::ten::is_scalar_v<Y>))
struct sin : func {
  static constexpr ::std::string name() { return ::std::string("sin"); }

  using output_type = Y;
  using value_type = X::value_type;
  using output_value_type = Y::value_type;

  void call(::std::shared_ptr<X> &x, ::std::shared_ptr<Y> &y) {
    using type = typename Y::value_type;
    if constexpr (::ten::is_scalar_v<Y>) {
      if (!y) {
        y = ::std::make_shared<Y>();
      }
      y->value() = ::std::sin(x->value());
    }
    if constexpr (::ten::is_tensor_v<Y>) {
      if (!y) {
        y = ::std::make_shared<Y>(x->shape(), x->format(), x->requires_grad(),
                                  x->storage_order());
      }
      for (size_t i = 0; i < x->size(); i++) {
        (*y.get())[i] = ::std::sin(static_cast<type>((*x.get())[i]));
      }
    }
  }

  /*
  template <class Gradient>
     requires(::ten::is_scalar_v<X> && ::ten::is_scalar_v<Gradient>)
  void gradient(X &x, Gradient &grad) {
     grad.value() = ::std::cos(x.value());
  }

  template <class Gradient>
     requires(::ten::is_tensor_v<X> && ::ten::is_tensor_v<Gradient>)
  void gradient(X &x, Gradient &grad) {
     if (x.shape() != grad.shape()) {
        ::std::cerr << "ten::functional::sin gradient different shapes.\n";
     } else {
        for (size_t i = 0; i < x.size(); i++) {
           grad[i] = ::std::cos(x[i]);
        }
     }
  }*/
};

/// asin
template <class X, class Y>
  requires((::ten::is_tensor<X>::value || ::ten::is_column<X>::value ||
            ::ten::is_row<X>::value) &&
           ::ten::is_tensor<Y>::value)
struct asin : func {
  static constexpr ::std::string name() { return ::std::string("asin"); }

  using output_type = Y;

  void call(::std::shared_ptr<X> &x, ::std::shared_ptr<Y> &y) {
    if (!y) {
      y = ::std::make_shared<Y>(x->shape(), x->format(), x->requires_grad(),
                                x->storage_order());
    }
    using type = typename Y::value_type;
    for (size_t i = 0; i < x->size(); i++) {
      (*y.get())[i] = ::std::asin(static_cast<type>((*x.get())[i]));
    }
  }
};

/// Sinh
template <class X, class Y>
  requires((::ten::is_tensor<X>::value || ::ten::is_column<X>::value ||
            ::ten::is_row<X>::value) &&
           ::ten::is_tensor<Y>::value)
struct sinh : func {
  static constexpr ::std::string name() { return ::std::string("asin"); }

  using output_type = Y;

  void call(::std::shared_ptr<X> &x, ::std::shared_ptr<Y> &y) {
    if (!y) {
      y = ::std::make_shared<Y>(x->shape(), x->format(), x->requires_grad(),
                                x->storage_order());
    }
    using type = typename Y::value_type;
    for (size_t i = 0; i < x->size(); i++) {
      (*y.get())[i] = ::std::sinh(static_cast<type>((*x.get())[i]));
    }
  }
};

/// Cosine
template <class X, class Y>
  requires(((::ten::is_tensor<X>::value || ::ten::is_column<X>::value ||
             ::ten::is_row<X>::value) &&
            ::ten::is_tensor<Y>::value) ||
           (::ten::is_scalar_v<X> && ::ten::is_scalar_v<Y>))
struct cos : func {
  static constexpr ::std::string name() { return ::std::string("cos"); }

  using value_type = X::value_type;
  using output_value_type = Y::value_type;
  using output_type = Y;

  void call(::std::shared_ptr<X> &x, ::std::shared_ptr<Y> &y) {
    if constexpr (::ten::is_scalar_v<Y>) {
      if (!y) {
        y = ::std::make_shared<Y>();
      }
      y->value() = ::std::cos(x->value());
    }
    if constexpr (::ten::is_tensor_v<Y>) {
      if (!y) {
        y = ::std::make_shared<Y>(x->shape(), x->format(), x->requires_grad(),
                                  x->storage_order());
      }
      for (size_t i = 0; i < x->size(); i++) {
        (*y.get())[i] =
            ::std::cos(static_cast<output_value_type>((*x.get())[i]));
      }
    }
  }

  /*
  template <class Gradient>
     requires(::ten::is_scalar_v<X> && ::ten::is_scalar_v<Gradient>)
  void gradient(X &x, Gradient &grad) {
     grad.value() = -::std::sin(x.value());
  }

  template <class Gradient>
     requires(::ten::is_scalar_v<X>)
  void gradient(::std::shared_ptr<X> &x, ::std::shared_ptr<Gradient> &y) {
     for (size_t i = 0; i < y.size(); i++) {
        y[i] = -::std::sin(x.value());
     }
  }

  template <class Gradient>
     requires(::ten::is_tensor_v<X> && ::ten::is_tensor_v<Gradient>)
  void gradient(X &x, Gradient &grad) {
     if (x.shape() != grad.shape()) {
        ::std::cerr << "ten::functional::cos gradient different sizes\n";
     } else {
        for (size_t i = 0; i < x.size(); i++) {
           grad[i] = -::std::sin(x[i]);
        }
     }
  }*/
};

/// acos
template <class X, class Y>
  requires((::ten::is_tensor<X>::value || ::ten::is_column<X>::value ||
            ::ten::is_row<X>::value) &&
           ::ten::is_tensor<Y>::value)
struct acos : func {
  static constexpr ::std::string name() { return ::std::string("acos"); }

  using output_type = Y;

  void call(::std::shared_ptr<X> &x, ::std::shared_ptr<Y> &y) {
    if (!y) {
      y = ::std::make_shared<Y>(x->shape(), x->format(), x->requires_grad(),
                                x->storage_order());
    }
    using type = typename Y::value_type;
    for (size_t i = 0; i < x->size(); i++) {
      (*y.get())[i] = ::std::acos(static_cast<type>((*x.get())[i]));
    }
  }
};

/// cosh
template <class X, class Y>
  requires((::ten::is_tensor<X>::value || ::ten::is_column<X>::value ||
            ::ten::is_row<X>::value) &&
           ::ten::is_tensor<Y>::value)
struct cosh : func {
  static constexpr ::std::string name() { return ::std::string("cosh"); }

  using output_type = Y;

  void call(::std::shared_ptr<X> &x, ::std::shared_ptr<Y> &y) {
    if (!y) {
      y = ::std::make_shared<Y>(x->shape(), x->format(), x->requires_grad(),
                                x->storage_order());
    }
    using type = typename Y::value_type;
    for (size_t i = 0; i < x->size(); i++) {
      (*y.get())[i] = ::std::cosh(static_cast<type>((*x.get())[i]));
    }
  }
};

/// Tangent
template <class X, class Y>
  requires(((::ten::is_tensor<X>::value || ::ten::is_column<X>::value ||
             ::ten::is_row<X>::value) &&
            ::ten::is_tensor<Y>::value) ||
           (::ten::is_scalar_v<X> && ::ten::is_scalar_v<Y>))
struct tan : func {
  static constexpr ::std::string name() { return ::std::string("tan"); }

  using value_type = X::value_type;
  using output_value_type = Y::value_type;
  using output_type = Y;

  void call(::std::shared_ptr<X> &x, ::std::shared_ptr<Y> &y) {
    if constexpr (::ten::is_scalar_v<Y>) {
      if (!y) {
        y = ::std::make_shared<Y>();
      }
      y->value() = ::std::tan(x->value());
    }
    if constexpr (::ten::is_tensor_v<Y>) {
      if (!y) {
        y = ::std::make_shared<Y>(x->shape(), x->format(), x->requires_grad(),
                                  x->storage_order());
      }
      for (size_t i = 0; i < x->size(); i++) {
        (*y.get())[i] =
            ::std::tan(static_cast<output_value_type>((*x.get())[i]));
      }
    }
  }

  /*
  template <class Gradient>
     requires(::ten::is_scalar_v<X> && ::ten::is_scalar_v<Gradient>)
  void gradient(X &x, Gradient &grad) {
     value_type z = ::std::tan(x.value());
     grad.value() = 1 + z * z;
  }

  template <class Gradient>
     requires(::ten::is_tensor_v<X> && ::ten::is_tensor_v<Gradient>)
  void gradient(X &x, Gradient &grad) {
     if (x.shape() != grad.shape()) {
        ::std::cerr << "ten::functional::tan gradient different shapes.\n";
     } else {
        for (size_t i = 0; i < x.size(); i++) {
           value_type t = ::std::tan(x[i]);
           grad[i] = 1 + t * t;
        }
     }
  }*/
};

template <class X, class Y>
  requires(((::ten::is_tensor<X>::value || ::ten::is_column<X>::value ||
             ::ten::is_row<X>::value) &&
            ::ten::is_tensor<Y>::value) ||
           (::ten::is_scalar_v<X> && ::ten::is_scalar_v<Y>))
struct atan : func {
  static constexpr ::std::string name() { return ::std::string("atan"); }

  using value_type = X::value_type;
  using output_value_type = Y::value_type;
  using output_type = Y;

  void call(::std::shared_ptr<X> &x, ::std::shared_ptr<Y> &y) {
    using type = typename Y::value_type;
    if constexpr (::ten::is_scalar_v<X>) {
      if (!y) {
        y = ::std::make_shared<Y>();
      }
      y->value() = ::std::atan(static_cast<type>(x->value()));
    }
    if constexpr (::ten::is_tensor_v<Y>) {
      if (!y) {
        y = ::std::make_shared<Y>(x->shape(), x->format(), x->requires_grad(),
                                  x->storage_order());
      }
      for (size_t i = 0; i < x->size(); i++) {
        (*y.get())[i] = ::std::atan(static_cast<type>((*x.get())[i]));
      }
    }
  }

  /*
  template <class Gradient>
     requires(::ten::is_scalar_v<X> && ::ten::is_scalar_v<Gradient>)
  void gradient(X &x, Gradient &grad) {
     grad.value() = 1 / (1 + x.value() * x.value());
  }

  template <class Gradient>
     requires(::ten::is_tensor_v<X> && ::ten::is_tensor_v<Gradient>)
  void gradient(X &x, Gradient &grad) {
     if (x.shape() != grad.shape()) {
        ::std::cerr << "ten::functional::atan gradient different shapes.\n";
     } else {
        for (size_t i = 0; i < x->size(); i++) {
           grad[i] = 1 / (1 + x[i] * x[i]);
        }
     }
  }*/
};

template <class X, class Y>
  requires(((::ten::is_tensor<X>::value || ::ten::is_column<X>::value ||
             ::ten::is_row<X>::value) &&
            ::ten::is_tensor<Y>::value) ||
           (::ten::is_scalar_v<X> && ::ten::is_scalar_v<Y>))
struct tanh : func {
  static constexpr ::std::string name() { return ::std::string("tanh"); }

  using value_type = X::value_type;
  using output_value_type = Y::value_type;
  using output_type = Y;

  void call(::std::shared_ptr<X> &x, ::std::shared_ptr<Y> &y) {
    using type = typename Y::value_type;
    if constexpr (::ten::is_scalar_v<X>) {
      if (!y) {
        y = ::std::make_shared<Y>();
      }
      y.value() = ::std::tanh(x.value());
    }
    if constexpr (::ten::is_tensor_v<Y>) {
      if (!y) {
        y = ::std::make_shared<Y>(x->shape(), x->format(), x->requires_grad(),
                                  x->storage_order());
      }
      for (size_t i = 0; i < x->size(); i++) {
        (*y.get())[i] = ::std::tanh(static_cast<type>((*x.get())[i]));
      }
    }
  }
};

/// Exponential
template <class X, class Y>
  requires(((::ten::is_tensor<X>::value || ::ten::is_column<X>::value ||
             ::ten::is_row<X>::value) &&
            ::ten::is_tensor<Y>::value) ||
           (::ten::is_scalar_v<X> && ::ten::is_scalar_v<Y>))
struct exp : func {
  static constexpr ::std::string name() { return ::std::string("exp"); }

  using value_type = X::value_type;
  using output_type = Y;

  void call(::std::shared_ptr<X> &x, ::std::shared_ptr<Y> &y) {
    using type = typename Y::value_type;
    if constexpr (::ten::is_scalar_v<X>) {
      if (!y) {
        y = ::std::make_shared<Y>();
      }
      y->value() = ::std::exp(x->value());
    }
    if constexpr (::ten::is_tensor_v<Y>) {
      if (!y) {
        y = ::std::make_shared<Y>(x->shape(), x->format(), x->requires_grad(),
                                  x->storage_order());
      }
      for (size_t i = 0; i < x->size(); i++) {
        (*y.get())[i] = ::std::exp(static_cast<type>((*x.get())[i]));
      }
    }
  }

  /*
  template <class Gradient>
     requires(::ten::is_scalar_v<X> && ::ten::is_scalar_v<Gradient>)
  void gradient(X &x, Gradient &grad) {
     grad.value() = ::std::exp(x.value());
  }

  template <class Gradient>
     requires(::ten::is_tensor_v<X> && ::ten::is_tensor_v<Gradient>)
  void gradient(X &x, Gradient &grad) {
     if (x.shape() != grad.shape()) {
        ::std::cerr << "ten::functional::exp gradient different shapes.\n";
     } else {
        for (size_t i = 0; i < x.size(); i++) {
           grad[i] = ::std::exp(x[i]);
        }
     }
  }*/
};

/// Natural logarithm
template <class X, class Y>
  requires(((::ten::is_tensor<X>::value || ::ten::is_column<X>::value ||
             ::ten::is_row<X>::value) &&
            ::ten::is_tensor<Y>::value) ||
           (::ten::is_scalar_v<X> && ::ten::is_scalar_v<Y>))
struct log : func {
  static constexpr ::std::string name() { return ::std::string("log"); }

  using value_type = X::value_type;
  using output_type = Y;

  void call(::std::shared_ptr<X> &x, ::std::shared_ptr<Y> &y) {
    using type = typename Y::value_type;
    if constexpr (::ten::is_scalar_v<X>) {
      if (!y) {
        y = ::std::make_shared<Y>();
      }
      y->value() = ::std::log(x->value());
    }
    if constexpr (::ten::is_tensor_v<Y>) {
      if (!y) {
        y = ::std::make_shared<Y>(x->shape(), x->format(), x->requires_grad(),
                                  x->storage_order());
      }
      for (size_t i = 0; i < x->size(); i++) {
        (*y.get())[i] = ::std::log(static_cast<type>((*x.get())[i]));
      }
    }
  }

  /*
  template <class Gradient>
     requires(::ten::is_scalar_v<X> && ::ten::is_scalar_v<Gradient>)
  void gradient(X &x, Gradient &grad) {
     grad.value() = value_type(1) / x.value();
  }

  template <class Gradient>
     requires(::ten::is_tensor_v<X> && ::ten::is_tensor_v<Gradient>)
  void gradient(X &x, Gradient &grad) {
     if (x.shape() != grad.shape()) {
        ::std::cerr << "ten::functional::log gradient different sizes.\n";
     } else {
        for (size_t i = 0; i < x->size(); i++) {
           grad[i] = value_type(1) / x[i];
        }
     }
  }*/
};

/// Logarithm
template <class X, class Y>
  requires(((::ten::is_tensor<X>::value || ::ten::is_column<X>::value ||
             ::ten::is_row<X>::value) &&
            ::ten::is_tensor<Y>::value) ||
           (::ten::is_scalar_v<X> && ::ten::is_scalar_v<Y>))
struct log10 : func {
  static constexpr ::std::string name() { return ::std::string("log10"); }
  using value_type = X::value_type;
  using output_type = Y;

  void call(::std::shared_ptr<X> &x, ::std::shared_ptr<Y> &y) {
    using type = typename Y::value_type;
    if constexpr (::ten::is_scalar_v<X>) {
      if (!y) {
        y = ::std::make_shared<Y>();
      }
      y->value() = ::std::log10(x->value());
    }
    if constexpr (::ten::is_tensor_v<Y>) {
      if (!y) {
        y = ::std::make_shared<Y>(x->shape(), x->format(), x->requires_grad(),
                                  x->storage_order());
      }
      for (size_t i = 0; i < x->size(); i++) {
        (*y.get())[i] = ::std::log10(static_cast<type>((*x.get())[i]));
      }
    }
  }

  /*
  template <class Gradient>
     requires(::ten::is_scalar_v<X> && ::ten::is_scalar_v<Gradient>)
  void gradient(X &x, Gradient &grad) {
     grad.value() = value_type(1) / (x.value() * ::std::log(value_type(10)));
  }

  template <class Gradient>
     requires(::ten::is_tensor_v<X> && ::ten::is_tensor_v<Gradient>)
  void gradient(X &x, Gradient &grad) {
     if (x.shape() != grad.shape()) {
        ::std::cerr << "ten::functional::log10 gradient different shapes.\n";
     } else {
        for (size_t i = 0; i < x->size(); i++) {
           grad[i] = value_type(1) / (x[i] * ::std::log(value_type(10)));
        }
     }
  }*/
};

/// Floor
template <class X, class Y>
  requires((::ten::is_tensor<X>::value || ::ten::is_column<X>::value ||
            ::ten::is_row<X>::value) &&
           ::ten::is_tensor<Y>::value)
struct floor : func {
  static constexpr ::std::string name() { return ::std::string("floor"); }

  using output_type = Y;

  void call(::std::shared_ptr<X> &x, ::std::shared_ptr<Y> &y) {
    if (!y) {
      y = ::std::make_shared<Y>(x->shape(), x->format(), x->requires_grad(),
                                x->storage_order());
    }
    using type = typename Y::value_type;
    for (size_t i = 0; i < x->size(); i++) {
      (*y.get())[i] = ::std::floor(static_cast<type>((*x.get())[i]));
    }
  }
};

/// Ceil
template <class X, class Y>
  requires(
      (::ten::is_tensor_v<X> || ::ten::is_column_v<X> || ::ten::is_row_v<X>) &&
      ::ten::is_tensor_v<Y>)
struct ceil : func {
  static constexpr ::std::string name() { return ::std::string("ceil"); }

  using output_type = Y;

  void call(::std::shared_ptr<X> &x, ::std::shared_ptr<Y> &y) {
    if (!y) {
      y = ::std::make_shared<Y>(x->shape(), x->format(), x->requires_grad(),
                                x->storage_order());
    }
    using type = typename Y::value_type;
    for (size_t i = 0; i < x->size(); i++) {
      (*y.get())[i] = ::std::ceil(static_cast<type>((*x.get())[i]));
    }
  }
};

////////////////////////////////////////////////////////////////////////////////
// Binary functions (Add, Sub, Mul and Div)

/// Binary function
template <ten::binary_operation Kind> struct binary_func {

  template <class X, class Y, class Z> struct func : ::ten::functional::func {
    static constexpr ::std::string name() {
      if constexpr (Kind == ::ten::binary_operation::add) {
        return ::std::string("add");
      }
      if constexpr (Kind == ::ten::binary_operation::sub) {
        return ::std::string("sub");
      }
      if constexpr (Kind == ::ten::binary_operation::mul) {
        return ::std::string("mul");
      }
      if constexpr (Kind == ::ten::binary_operation::div) {
        return ::std::string("div");
      }
    }

    using output_type = Z;

    void call(::std::shared_ptr<X> &x, ::std::shared_ptr<Y> &y,
              ::std::shared_ptr<Z> &z) {
      if (!ten::details::same_shape(x->shape(), y->shape())) {
        ::std::cerr << "ten::functional::binary_func<" << name()
                    << ">, input have different shapes\n";
      } else if (!x->is_dense() || !y->is_dense()) {
        ::std::cerr << "ten::fuctional::binary_fun<" << name()
                    << ">, tensor storage format must be dense.\n";
      } else if (x->storage_order() != y->storage_order()) {
        ::std::cerr << "ten::functional::binary_fun<" << name()
                    << ">, different storage order.\n";
      } else {
        if (!z) {
          z = ::std::make_shared<Z>(x->shape(), ten::storage_format::dense,
                                    x->requires_grad() || y->requires_grad(),
                                    x->storage_order());
        }
        auto xarr = *x.get();
        auto yarr = *y.get();
        auto zarr = *z.get();
        ::ten::kernels::binary_ops<Kind>(xarr, yarr, zarr);
      }
    }

    /*
    // Compute dz(x,y)/dx in grad
    template <class Gradient>
       requires(::ten::is_tensor_v<Gradient>)
    //void gradient_left(const X & x, const Y &y, Gradient &grad) {
       if (y.shape() != grad.shape()) {
          ::std::cerr << "ten::functional::binary_func<" << name()
                    << "> gradient ";
          ::std::cerr << "different shapes.\n";
       } else {
          if constexpr ((Kind == ::ten::binary_operation::add) ||
                        (Kind == ::ten::binary_operation::sub)) {
             for (size_t i = 0; i < grad.size(); i++) {
                grad[i] = 1;
             }
          }
          if (Kind == ::ten::binary_operation::mul) {
             for (size_t i = 0; i < grad.size(); i++) {
                grad[i] = y[i];
             }
          }
          if (Kind == ::ten::binary_operation::div) {
             for (size_t i = 0; i < grad.size(); i++) {
                grad[i] = 1 / y[i];
             }
          }
       }
    }

    // Compute dz(x,y)/dy in grad
    template <class Gradient>
       requires(::ten::is_tensor_v<Gradient>)
    void gradient_right(const X &x, const Y &y, Gradient &grad) {
       if ((x.shape() != grad.shape()) || (y.shape() != grad.shape())) {
          ::std::cerr << "ten::functional::binary_func<" << name()
                    << "> gradient ";
          ::std::cerr << "different shapes.\n";
       } else {
          if constexpr (Kind == ::ten::binary_operation::add) {
             for (size_t i = 0; i < grad.size(); i++) {
                grad[i] = 1;
             }
          }
          if constexpr (Kind == ::ten::binary_operation::sub) {
             for (size_t i = 0; i < grad.size(); i++) {
                grad[i] = -1;
             }
          }
          if (Kind == ::ten::binary_operation::mul) {
             for (size_t i = 0; i < grad.size(); i++) {
                grad[i] = x[i];
             }
          }
          if (Kind == ::ten::binary_operation::div) {
             for (size_t i = 0; i < grad.size(); i++) {
                grad[i] = -x[i] / (y[i] * y[i]);
             }
          }
       }
    }
    */
  };
};

/// Binary function with scalar
template <::ten::binary_operation Kind> struct scalar_left_binary_func {

  template <Scalar X, class Y, class Z> struct func : ::ten::functional::func {
    static constexpr ::std::string name() {
      if constexpr (Kind == ::ten::binary_operation::add) {
        return ::std::string("add");
      }
      if constexpr (Kind == ::ten::binary_operation::sub) {
        return ::std::string("sub");
      }
      if constexpr (Kind == ::ten::binary_operation::mul) {
        return ::std::string("mul");
      }
      if constexpr (Kind == ::ten::binary_operation::div) {
        return ::std::string("div");
      }
    }

    using output_type = Z;

    void call(::std::shared_ptr<X> &x, ::std::shared_ptr<Y> &y,
              ::std::shared_ptr<Z> &z) {
      if (!z) {
        z = ::std::make_shared<Z>(y->shape(), y->format(), y->requires_grad(),
                                  y->storage_order());
      }
      Y xarr(y->shape(), ten::storage_format::dense, false, y->storage_order());
      for (size_t i = 0; i < y->size(); i++) {
        xarr[i] = x->value();
      }
      ::ten::kernels::binary_ops<Kind>(xarr, *y.get(), *z.get());
    }
  };
};

/// Binary function with scalar right type
template <::ten::binary_operation Kind> struct scalar_right_binary_func {

  template <class X, Scalar Y, class Z> struct func : ::ten::functional::func {
    static constexpr ::std::string name() {
      if constexpr (Kind == ::ten::binary_operation::add) {
        return ::std::string("add");
      }
      if constexpr (Kind == ::ten::binary_operation::sub) {
        return ::std::string("sub");
      }
      if constexpr (Kind == ::ten::binary_operation::mul) {
        return ::std::string("mul");
      }
      if constexpr (Kind == ::ten::binary_operation::div) {
        return ::std::string("div");
      }
    }

    using output_type = Z;

    void call(::std::shared_ptr<X> &x, ::std::shared_ptr<Y> &y,
              ::std::shared_ptr<Z> &z) {
      if (!z) {
        z = ::std::make_shared<Z>(x->shape(), x->format(), x->requires_grad(),
                                  x->storage_order());
      }
      X yarr(x->shape(), ten::storage_format::dense, false, x->storage_order());
      for (size_t i = 0; i < x->size(); i++) {
        yarr[i] = y->value();
      }
      ::ten::kernels::binary_ops<Kind>(*x.get(), yarr, *z.get());
    }
  };
};

/// Multiply
template <class A, class B, class C> struct mul : func {};

/// tensor * tensor
///
/// - Elementwise vector * vector
/// - Matrix * vector
/// - Matrix * matrix
/// - Elementwise tensor * tensor
template <Tensor X, Tensor Y, Tensor Z>
struct mul<X, Y, Z> : ::ten::functional::func {
  static constexpr ::std::string name() { return ::std::string("mul"); }

  using output_type = Z;

  void call(::std::shared_ptr<X> &x, ::std::shared_ptr<Y> &y,
            ::std::shared_ptr<Z> &z) {
    const ::std::size_t xrank = x->rank();
    const ::std::size_t yrank = y->rank();
    if (xrank == 1 && yrank == 1) {
      // Elementwise vector - vector multiplication
      if (x->size() != y->size()) {
        ::std::cerr << "ten::functional::mul, different vector shapes\n";
      } else if (!x->is_dense() && !y->is_dense()) {
        ::std::cerr << "ten::functional::mul, storage format must be dense.\n";
      } else if (x->storage_order() != y->storage_order()) {
        ::std::cerr << "ten::functional::mul, different storage order.\n";
      } else {
        if (!z) {
          z = ::std::make_shared<Z>(x->shape(), ten::storage_format::dense,
                                    x->requires_grad() || y->requires_grad(),
                                    x->storage_order());
        }
        ::ten::kernels::binary_ops<::ten::binary_operation::mul>(
            *x.get(), *y.get(), *z.get());
      }
    } else if (xrank == 2 && yrank == 2) {
      // Matrix - matrix multiplication
      if (x->dim(1) != y->dim(0)) {
        ::std::cerr
            << "ten::functional::mul - incompatibles input matrices shapes.\n";
      } else if (!x->is_dense() || !y->is_dense()) {
        ::std::cerr << "ten::functional::mul, storage format must be dense.\n";
      } else if (x->storage_order() != y->storage_order()) {
        ::std::cerr << "ten::functional::mul, different storage order.\n";
      } else {
        if (!z) {
          ::std::initializer_list<size_type> &&dims = {x->dim(0), y->dim(1)};
          z = ::std::make_shared<Z>(
              ::std::move(dims), ten::storage_format::dense,
              x->requires_grad() || y->requires_grad(), x->storage_order());
        }
        kernels::mul(*x.get(), *y.get(), *z.get());
      }
    } else if (xrank == 2 && yrank == 1) {
      // Matrix - vector multiplication
      if (x->dim(1) != y->size()) {
        ::std::cerr << "ten::functional::mul matrix vector incompatible input "
                       "shapes\n";
      } else if (!x->is_dense() || !y->is_dense()) {
        ::std::cerr << "ten::functional::mul, storage format must be dense.\n";
      } else if (x->storage_order() != y->storage_order()) {
        ::std::cerr << "ten::functional::mul, different storage order.\n";
      } else {
        if (!z) {
          ::std::initializer_list<size_type> &&dims = {y->size()};
          z = ::std::make_shared<Z>(
              ::std::move(dims), ten::storage_format::dense,
              x->requires_grad() || y->requires_grad(), x->storage_order());
        }
        kernels::mul(*x.get(), *y.get(), *z.get());
      }
    } else if (xrank == yrank) {
      if (!ten::details::same_shape(x->shape(), y->shape())) {
        ::std::cerr << "ten::functional::mul elementwise tensor multiplication "
                       "expects same tensor shape.\n";
      } else if (!x->is_dense() || !y->is_dense()) {
        ::std::cerr << "ten::functional::mul, storage format must be dense.\n";

      } else if (x->storage_order() != y->storage_order()) {
        ::std::cerr << "ten::functional::mul, different storage order.\n";
      } else {
        if (!z) {
          z = ::std::make_shared<Z>(x->shape(), ten::storage_format::dense,
                                    x->requires_grad() || x->requires_grad(),
                                    x->storage_order());
        }
        kernels::binary_ops<ten::binary_operation::mul>(*x.get(), *y.get(),
                                                        *z.get());
      }
    } else {
      ::std::cerr << "ten::functional::mul, multiplication not supported.\n";
    }
  }

  /*
   template <class Gradient>
      requires(::ten::is_tensor_v<Gradient>)
   void gradient_left(const X &x, const Y &y, Gradient &grad) {
      if ((x.shape() != grad.shape()) || (y.shape() != grad.shape())) {
         ::std::cerr << "ten::functional::mul gradient - different shapes.\n";
      } else {
         for (size_t i = 0; i < grad.size(); i++) {
            grad[i] = y[i];
         }
      }
   }

   template <class Gradient>
      requires(::ten::is_tensor_v<Gradient>)
   void gradient_right(const X &x, const Y &y, Gradient &grad) {
      if ((x.shape() != grad.shape()) || (y.shape() != grad.shape())) {
         ::std::cerr << "ten::functional::mul gradient - different shapes.\n";
      } else {
         for (size_t i = 0; i < grad.size(); i++) {
            grad[i] = x[i];
         }
      }
   }
  */

  /*
 // Static shapes gradient
 template <class Gradient>
    requires(::ten::is_matrix_v<X> && ::ten::is_matrix_v<Y> &&
             ::ten::is_matrix_v<Gradient>)
 void gradient_left(const X &x, const Y &y, Gradient &grad) {
    if (x.shape().dim(1) != y.shape().dim(0)) {
       ::std::cerr << "ten::functional::mul gradient - incompatible shapes.\n";
    } else if ((x.shape().dim(0) != grad.shape().dim(0)) ||
               (x.shape().dim(1) != grad.shape().dim(1))) {
       ::std::cerr
           << "ten::functional::mul gradient - incompatible output shape.\n";
    } else {
       if constexpr (Gradient::shape_type::is_dynamic()) {
          auto rows = grad.shape().dim(0);
          auto cols = grad.shape().dim(1);
          for (size_t i = 0; i < cols; i++) {
             grad(0, i) = 0;
             for (size_t j = 0; j < y.shape().dim(1); j++) {
                grad(0, i) += y(i, j);
             }
          }
          for (size_t j = 0; j < cols; j++) {
             for (size_t i = 1; i < rows; i++) {
                grad(i, j) = grad(0, j);
             }
          }
       }
    }
 }

 template <class Gradient>
    requires(::ten::is_matrix_v<X> && ::ten::is_matrix_v<Y> &&
             ::ten::is_matrix_v<Gradient>)
 void gradient_right(const X &x, const Y &y, Gradient &grad) {
    if (x.shape().dim(1) != y.shape().dim(0)) {
       ::std::cerr << "ten::functional::mul gradient incompatible shapes.\n";
    } else if ((y.shape().dim(0) != grad.shape().dim(0)) ||
               (y.shape().dim(1) != grad.shape().dim(1))) {
       ::std::cerr
           << "ten::functional::mul gradient incompatible output shape.\n";
    } else {
       if constexpr (Gradient::shape_type::is_dynamic()) {
          auto rows = grad.shape().dim(0);
          auto cols = grad.shape().dim(1);
          size_t j = 0;
          for (size_t i = 0; i < rows; i++) {
             grad(i, 0) = 0;
             for (size_t k = 0; k < x.shape().dim(0); k++) {
                grad(i, 0) += x(k, j);
             }
             j++;
          }
          for (size_t j = 1; j < cols; j++) {
             for (size_t i = 0; i < rows; i++) {
                grad(i, j) = grad(i, 0);
             }
          }
       }
    }
 }
*/

  /*
   template <class Gradient>
      requires(::ten::is_matrix_v<Gradient>)
   void gradient_left(const X & x, const Y &y, Gradient &grad) {
      if constexpr (Gradient::shape_type::is_dynamic()) {
         auto rows = grad.shape().dim(0);
         auto cols = grad.shape().dim(1);
         for (size_t i = 0; i < cols; i++) {
            grad(0, i) = y[i];
         }
         for (size_t j = 0; j < cols; j++) {
            for (size_t i = 1; i < rows; i++) {
               grad(i, j) = grad(0, j);
            }
         }
      }
   }

   template <class Gradient>
      requires(::ten::is_vector_v<Gradient>)
   void gradient_right(const X &x, const Y & y, Gradient &grad) {
      if constexpr (Gradient::shape_type::is_dynamic()) {
         size_t j = 0;
         for (size_t i = 0; i < grad.size(); i++) {
            grad[i] = 0;
            for (size_t k = 0; k < x.shape().dim(0); k++) {
               grad[i] += x(k, j);
            }
            j++;
         }
      }
   }*/
};

// Multiply by a scalar
/*template<Scalar X, Tensor Y, Tensor Z>
struct mul<X, Y, Z>  = scalar_left_binary_func<X, Y, Z>::func;

template<Tensor X, Scalar Y, Tensor Z>
struct mul<X, Y, Z>  = scalar_right_binary_func<X, Y, Z>::func;
*/

// scalar * tensor
template <Scalar X, Tensor Y, Tensor Z> struct mul<X, Y, Z> : func {

  static constexpr ::std::string name() { return ::std::string("mul"); }

  using output_type = Z;

  void call(::std::shared_ptr<X> &x, ::std::shared_ptr<Y> &y,
            ::std::shared_ptr<Z> &z) {
    if (!y->is_dense()) {
      ::std::cerr << "mul, storage format must be dense.\n";
    } else {
      if (!z) {
        z = ::std::make_shared<Z>(y->shape(), ten::storage_format::dense,
                                  y->requires_grad(), y->storage_order());
      }
      Y xarr(y->shape(), ten::storage_format::dense, false, y->storage_order());
      for (size_t i = 0; i < xarr.size(); i++) {
        xarr[i] = x->value();
      }
      ten::kernels::mul(xarr, *y.get(), *z.get());
    }
  }
};

// tensor * scalar
template <Tensor X, Scalar Y, Tensor Z> struct mul<X, Y, Z> : func {

  static constexpr ::std::string name() { return ::std::string("mul"); }

  using output_type = Z;

  void call(::std::shared_ptr<X> &x, ::std::shared_ptr<Y> &y,
            ::std::shared_ptr<Z> &z) {
    if (!x->is_dense()) {
      ::std::cerr << "mul, storage format must be dense.\n";
    } else {
      if (!z) {
        z = ::std::make_shared<Z>(x->shape(), ten::storage_format::dense,
                                  x->requires_grad(), x->storage_order());
      }
      X yarr(x->shape(), ten::storage_format::dense, false, x->storage_order());
      for (size_t i = 0; i < yarr.size(); i++) {
        yarr[i] = y->value();
      }
      ten::kernels::mul(yarr, *y.get(), *z.get());
    }
  }
};

////////////////////////////////////////////////////////////////////////////////
// Reshape and flatten

/// Reshape
template <class X, class Y>
  requires((::ten::is_tensor<X>::value || ::ten::is_column<X>::value ||
            ::ten::is_row<X>::value) &&
           ::ten::is_tensor<Y>::value)
struct reshape : ::ten::functional::func {
  static constexpr ::std::string name() { return ::std::string("reshape"); }

public:
  using output_type = Y;

private:
  ::std::vector<::std::size_t> _shape;

public:
  reshape(const ::std::vector<::std::size_t> &s) : _shape(s) {}
  reshape(::std::vector<::std::size_t> &&s) : _shape(::std::move(s)) {}

  void call(::std::shared_ptr<X> &x, ::std::shared_ptr<Y> &y) {
    y = ::std::make_shared<Y>(x->node(), _shape, ten::storage_format::dense,
                              x->requires_grad(), x->storage_order());
  }
};

/// Flatten
template <class X, class Y>
  requires((::ten::is_tensor<X>::value || ::ten::is_column<X>::value ||
            ::ten::is_row<X>::value) &&
           ::ten::is_tensor<Y>::value)
struct flatten : ::ten::functional::func {
  static constexpr ::std::string name() { return ::std::string("flatten"); }

public:
  using output_type = Y;

public:
  void call(::std::shared_ptr<X> &x, ::std::shared_ptr<Y> &y) {
    ::std::vector<::std::size_t> shape({x->size()});
    y = ::std::make_shared<Y>(x->node(), shape, ten::storage_format::dense,
                              x->requires_grad(), x->storage_order());
  }
};

////////////////////////////////////////////////////////////////////////////////
// Transpose a matrix

/*
// Dynamic transpose
template <class Shape> struct dynamic_transpose {

   template <class X, class Y> struct func : ::ten::functional::func<true> {
      static_assert(Shape::is_dynamic(), "Shape must be dynamic");
      // FIXME Currently defined only for matrices
      static_assert(Shape::rank() == 2, "Shape rank must be 2.");

      static constexpr ::std::string name() { return ::std::string("transpose");
}

      using output_type = Y;

      void operator()(::std::shared_ptr<X> &x, ::std::shared_ptr<Y> &y) {
         if (!y) {
            typename Y::shape_type yshape(
                {x->shape().dim(1), x->shape().dim(0)});
            y = ::std::make_shared<Y>(yshape);
         }
         size_type m = x->shape().dim(0);
         size_type n = x->shape().dim(1);
         for (size_type i = 0; i < n; i++) {
            for (size_type j = 0; j < m; j++) {
               (*y.get())(i, j) = (*x.get())(j, i);
            }
         }
      }
   };
};*/

////////////////////////////////////////////////////////////////////////////////
// Neural networks activation functions

/*
/// Relu
template <class X, class Y>
   requires((::ten::is_tensor_v<X> && ::ten::is_tensor_v<Y>) ||
            (::ten::is_stensor_v<X> && ::ten::is_stensor_v<Y>))
struct relu : func<> {
   static constexpr ::std::string name() { return ::std::string("relu"); }
   using value_type = X::value_type;
   using output_type = Y;

   void operator()(::std::shared_ptr<X> &x, ::std::shared_ptr<Y> &y) {
      if constexpr (Y::is_dynamic()) {
         if (!y) {
            y = ::std::make_shared<Y>(x->shape());
         }
         if (x->size() != y->size()) {
            ::std::cerr << "ten::functional::relu different sizes" <<
::std::endl; } else { auto xarr = *x.get(); auto yarr = *y.get(); for (size_t i
= 0; i < y->size(); i++) { if (xarr[i] >= 0) { yarr[i] = xarr[i]; } else {
                  yarr[i] = 0;
               }
            }
         }
      } else {
         if (!y) {
            y = ::std::make_shared<Y>();
         }
         if constexpr (X::static_size() != Y::static_size()) {
            ::std::cerr << "ten::functional::relu different sizes" <<
::std::endl; } else { auto xarr = *x.get(); auto yarr = *y.get(); for (size_t i
= 0; i < Y::static_size(); i++) { if (xarr[i] >= 0) { yarr[i] = xarr[i]; } else
{ yarr[i] = 0;
               }
            }
         }
      }
   }

   // Compute dy/dx in grad
   template <class Gradient>
      requires(::ten::is_tensor_v<X> && ::ten::is_tensor_v<Gradient>)
   void gradient(X &x, Gradient &grad) {
      if (x.shape() != grad.shape()) {
         ::std::cerr << "ten::functional::relu gradient different shapes.\n";
      } else {
         auto xarr = *x.get();
         auto gradarr = *grad.get();
         for (size_t i = 0; i < grad->size(); i++) {
            if (xarr[i] >= 0) {
               gradarr[i] = 1;
            } else {
               gradarr[i] = 0;
            }
         }
      }
   }

   template <class Gradient>
      requires(::ten::is_stensor_v<X> && ::ten::is_stensor_v<Gradient>)
   void gradient(X &x, Gradient &grad) {
      if constexpr (!::std::is_same_v<typename X::shape_type,
                                    typename Gradient::shape_type>) {
         ::std::cerr << "ten::functional::relu gradient different shapes.\n";
      } else {
         auto xarr = *x.get();
         auto gradarr = *grad.get();
         for (size_t i = 0; i < Gradient::static_size(); i++) {
            if (xarr[i] >= 0) {
               gradarr[i] = 1;
            } else {
               gradarr[i] = 0;
            }
         }
      }
   }
};

/// Leaky relu
template <class X, class Y>
   requires((::ten::is_tensor_v<X> && ::ten::is_tensor_v<Y>) ||
            (::ten::is_stensor_v<X> && ::ten::is_stensor_v<Y>))
struct leaky_relu : func<> {
   static constexpr ::std::string name() { return ::std::string("leaky_relu"); }
   using value_type = X::value_type;
   using output_type = Y;

   void operator()(::std::shared_ptr<X> &x, ::std::shared_ptr<Y> &y) {
      if constexpr (Y::is_dynamic()) {
         if (!y) {
            y = ::std::make_shared<Y>(x->shape());
         }
         if (x->size() != y->size()) {
            ::std::cerr << "ten::functional::relu different sizes" <<
::std::endl; } else { auto xarr = *x.get(); auto yarr = *y.get(); for (size_t i
= 0; i < y->size(); i++) { if (xarr[i] <= 0) { yarr[i] = value_type(0.01) *
xarr[i]; } else { yarr[i] = xarr[i];
               }
            }
         }
      } else {
         if (!y) {
            y = ::std::make_shared<Y>();
         }
         if constexpr (X::static_size() != Y::static_size()) {
            ::std::cerr << "ten::functional::relu different sizes" <<
::std::endl; } else { auto xarr = *x.get(); auto yarr = *y.get(); for (size_t i
= 0; i < Y::static_size(); i++) { if (xarr[i] <= 0) { yarr[i] = value_type(0.01)
* xarr[i]; } else { yarr[i] = xarr[i];
               }
            }
         }
      }
   }

   // Compute dy/dx in grad
   template <class Gradient>
      requires(::ten::is_tensor_v<X> && ::ten::is_tensor_v<Gradient>)
   void gradient(X &x, Gradient &grad) {
      if (x.shape() != grad.shape()) {
         ::std::cerr
             << "ten::functional::leaky_relu gradient different shapes.\n";
      } else {
         auto xarr = *x.get();
         auto gradarr = *grad.get();
         for (size_t i = 0; i < grad->size(); i++) {
            if (xarr[i] < 0) {
               gradarr[i] = 0.01;
            } else {
               gradarr[i] = 1;
            }
         }
      }
   }

   template <class Gradient>
      requires(::ten::is_stensor_v<X> && ::ten::is_stensor_v<Gradient>)
   void gradient(X &x, Gradient &grad) {
      if constexpr (!::std::is_same_v<typename X::shape_type,
                                    typename Gradient::shape_type>) {
         ::std::cerr
             << "ten::functional::leaky_relu gradient different shapes.\n";
      } else {
         auto xarr = *x.get();
         auto gradarr = *grad.get();
         for (size_t i = 0; i < Gradient::static_size(); i++) {
            if (xarr[i] < 0) {
               gradarr[i] = 0.01;
            } else {
               gradarr[i] = 1;
            }
         }
      }
   }
};

/// Sigmoid
template <class X, class Y>
   requires((::ten::is_tensor_v<X> && ::ten::is_tensor_v<Y>) ||
            (::ten::is_stensor_v<X> && ::ten::is_stensor_v<Y>))
struct sigmoid : func<> {
   static constexpr ::std::string name() { return ::std::string("sigmoid"); }
   using value_type = X::value_type;
   using output_type = Y;

   void operator()(::std::shared_ptr<X> &x, ::std::shared_ptr<Y> &y) {
      if constexpr (X::is_dynamic()) {
         if (!y) {
            y = ::std::make_shared<Y>(x->shape());
         }
         if (x->size() != y->size()) {
            ::std::cerr << "ten::functional::relu different input sizes\n";
         } else {
            for (size_t i = 0; i < y->size(); i++) {
               (*y.get())[i] =
                   value_type(1) / (value_type(1) + ::std::exp(-(*x.get())[i]));
            }
         }
      } else {
         if (!y) {
            y = ::std::make_shared<Y>();
         }
         if (X::static_size() != Y::static_size()) {
            ::std::cerr << "ten::functional::relu different input sizes\n";
         } else {
            for (size_t i = 0; i < Y::static_size(); i++) {
               (*y.get())[i] =
                   value_type(1) / (value_type(1) + ::std::exp(-(*x.get())[i]));
            }
         }
      }
   }

   // Compute dy/dx in grad
   template <class Gradient>
      requires(::ten::is_tensor_v<X> && ::ten::is_tensor_v<Gradient>)
   void gradient(X &x, Gradient &grad) {
      if (x.shape() != grad.shape()) {
         ::std::cerr << "ten::functional::relu gradient different shapes.\n";
      } else {
         for (size_t i = 0; i < x.size(); i++) {
            value_type e = ::std::exp(-x[i]);
            grad[i] = e / ((1 + e) * (1 + e));
         }
      }
   }

   template <class Gradient>
      requires(::ten::is_stensor_v<X> && ::ten::is_stensor_v<Gradient>)
   void gradient(X &x, Gradient &grad) {
      if constexpr (!::std::is_same_v<typename X::shape_type,
                                    typename Gradient::shape_type>) {
         ::std::cerr << "ten::functional::relu gradient different shapes.\n";
      } else {
         for (size_t i = 0; i < X::static_size(); i++) {
            value_type e = ::std::exp(-x[i]);
            grad[i] = e / ((1 + e) * (1 + e));
         }
      }
   }
};

/// MSE Loss
template <class X, class Y, class Z>
   requires((::ten::is_tensor_v<X> && ::ten::is_tensor_v<Y> &&
             ::ten::is_scalar_v<Z>) ||
            (::ten::is_stensor_v<X> && ::ten::is_stensor_v<Y> &&
             ::ten::is_scalar_v<Z>))
struct mse : func<> {
   static constexpr ::std::string name() { return ::std::string("mse"); }
   using value_type = X::value_type;
   using output_type = Y;

   void operator()(::std::shared_ptr<X> &x, ::std::shared_ptr<Y> &y,
                   ::std::shared_ptr<Z> &z) {
      if constexpr (X::is_dynamic() && Y::is_dynamic()) {
         if (x->size() != y->size()) {
            ::std::cerr << "ten::functional::mse different input sizes.\n";
         } else {
            if (!z) {
               z = ::std::make_shared<Z>();
            }
            value_type res = 0;
            for (size_t i = 0; i < x->size(); i++) {
               value_type diff = (*x.get())[i] - (*y.get())[i];
               res += diff * diff;
            }
            z->value() = res / x->size();
         }
      } else {
         if constexpr (X::static_size() != Y::static_size()) {
            ::std::cerr << "ten::functional::mse different input sizes.\n";
         } else {
            if (!z) {
               z = ::std::make_shared<Z>();
            }
            value_type res = 0;
            for (size_t i = 0; i < X::static_size(); i++) {
               value_type diff = (*x.get())[i] - (*y.get())[i];
               res += diff * diff;
            }
            z->value() = res / X::static_size();
         }
      }
   }

   // Compute dz/dx in grad
   template <class Gradient>
      requires(::ten::is_vector_v<Gradient> || ::ten::is_matrix_v<Gradient>)
   void gradient_left(X &x, Y &y, Gradient &grad) {
      if (x.size() != y.size()) {
         ::std::cerr << "ten::functional::mse gradient different sizes.\n";
      } else {
         if (x.size() != grad.size()) {
            ::std::cerr
                << "ten::functional::mse gradient size should be equal to "
                << x.size() << "\n";
         }
         using T = Gradient::value_type;
         for (size_t i = 0; i < grad.size(); i++) {
            grad[i] = 2 * (x[i] - y[i]) / T(grad.size());
         }
      }
   }

   template <class Gradient>
      requires(::ten::is_svector_v<Gradient> || ::ten::is_smatrix_v<Gradient>)
   void gradient_left(X &x, Y &y, Gradient &grad) {
      if constexpr (X::static_size() != Y::static_size()) {
         ::std::cerr << "ten::functional::mse gradient different sizes.\n";
      } else {
         if constexpr (X::static_size() != Gradient::static_size()) {
            ::std::cerr
                << "ten::functional::mse gradient size should be equal to "
                << X::static_size() << "\n";
         } else {
            using T = Gradient::value_type;
            for (size_t i = 0; i < Gradient::static_size(); i++) {
               grad[i] = 2 * (x[i] - y[i]) / T(Gradient::static_size());
            }
         }
      }
   }

   // Compute dz/dy in grad
   template <class Gradient>
      requires(::ten::is_vector_v<Gradient> || ::ten::is_matrix_v<Gradient>)
   void gradient_right(X &x, Y &y, Gradient &grad) {
      if (x.size() != y.size()) {
         ::std::cerr << "ten::functional::mse gradient different sizes.\n";
      } else {
         if (x.size() != grad.size()) {
            ::std::cerr
                << "ten::functional::mse gradient size should be equal to "
                << x.size() << "\n";
         } else {
            using T = Gradient::value_type;
            for (size_t i = 0; i < grad.size(); i++) {
               grad[i] = -2 * (x[i] - y[i]) / T(grad.size());
            }
         }
      }
   }

   template <class Gradient>
      requires(::ten::is_svector_v<Gradient> || ::ten::is_smatrix_v<Gradient>)
   void gradient_right(X &x, Y &y, Gradient &grad) {
      if (X::static_size() != Y::static_size()) {
         ::std::cerr << "ten::functional::mse gradient different sizes.\n";
      } else {
         if constexpr (X::static_size() != Gradient::static_size()) {
            ::std::cerr
                << "ten::functional::mse gradient size should be equal to "
                << X::static_size() << "\n";
         } else {
            using T = Gradient::value_type;
            for (size_t i = 0; i < Gradient::static_size(); i++) {
               grad[i] = -2 * (x[i] - y[i]) / T(Gradient::static_size());
            }
         }
      }
   }
};*/

} // namespace ten::functional

#endif
