#ifndef TEN_COMBINATORICS_ENUMERATIVE
#define TEN_COMBINATORICS_ENUMERATIVE

#include <ten/tensor>

namespace ten{

/// Compute the factorial of n
template<typename T = std::size_t>
T factorial(T n) {
  T f = 1;
  for (std::size_t i = 1; i <= n; i++) {
    f *= i;
  }
  return f;
}

/// Compute the permutations of k objects from n
/// that is n(n-1) ... (n-k+1) = n!/(n-k)!
template<typename T = std::size_t>
T perm(T n, T k) {
  if (k == 0) {
    return 1;
  }
  T p = n;
  for (std::size_t i = 1; i + 1 <= k; i++) {
    p *= n - i;
  }
  return p;
}

/// Compute the combinations of k objects from n
/// that is n!/k!(n-k)! = n(n-1)...(n-k+1)/k!
template<typename T = std::size_t>
T comb(T n, T k) {
  if ((k == 0) || (k == n)) {
    return 1;
  }
  return T(std::floor(perm(n,k) / factorial(k)));
}

/// Pascal triangle for 0 to n included
template<typename T = std::size_t>
tensor<T> pascal_triangle(std::size_t n) {
  tensor<T> x = zeros<T>({n+1, n+1});
  x(0,0) = 1;
  for (std::size_t i = 1; i <= n; i++) {
    x(i,0) = 1;
    for (std::size_t j = 1; j <= i; j++) {
      x(i,j) = x(i-1, j-1) + x(i-1,j);
    }
  }
  return x;
}

/// nth line of the pascal triangle
template<typename T = std::size_t>
tensor<T> nth_pascal_triangle(std::size_t n) {
  tensor<T> xprev({n+1});
  xprev[0] = 1;
  tensor<T> x({n+1});
  x[0] = 1;
  for (std::size_t i = 1; i <= n; i++) {
    // Compute x
    for (std::size_t j = 1; j <= i; j++) {
      x[j] = xprev[j-1] + xprev[j];
    }
    // Save x into xprev
    for (std::size_t j = 1; j <= i; j++) {
      xprev[j] = x[j];
    }
  }
  return x;
}

}

#endif
