# ten

ten is a header only C++23 dynamic tensor library designed for high performance numerical computations, prioritizing speed above all else. It assume that the user will ensure the correctness of their program. Exception handling and bounds checking are disabled to minimize overhead. This makes it ideal for applications where computational efficiency is the goal such as deep learning and scientific computing. Support for automatic differentiation and basic neural networks is planned.

## Tensor classes

The library is build around a core tensor class `tensor<T>` that provide efficient storage (column major and row major) and manipulation of multidimensional arrays through operator overloading. Operations are implemented using techniques such as SIMD instructions and cache friendly memory access. Error handling is minimized, instead of throwing exceptions, the library relies on the user to validate inputs and manage potential issues. Other classes such as `sparse_tensor<T>`, `diagonal<T>`, `view<T>`, `col<T>`, and `row<T>` are planned.

## Expressions API

An expression API class for representing unary and binary operations between tensors is planned to be implemented. Its inspired by compiler optimization techniques and passes. It will make it possible to do expression matching at compile time and fuse some opeations, for examples a basic call to gemm can be written as `c = a * b + c`, it will be lowered to `gemm(1.0, a, b, 1.0, c);` instead of writing `c.noalis() = a * b + c` as in most numerical libraries. Also unary operation will be lowered to inplace operations whenever that's possible. For example `x = ten::sqrt(x)` will be lowered to an inplace operation `inplace_sqrt(x)`.

## Features

- [x] Multi dimensional dynamic arrays
- [x] Special matrices (diagonal, triangular, transposed, ...)
- [x] BLAS and LAPACK backend for high performance numerical linear algebra
- [x] Lazy evaluation of expressions
- [x] Factory functions (fill, ones, zeros, range, linear)
- [x] Probability distributions and random numbers generation
- [x] Generate automatic python bindings ([tenpy](https://github.com/istmarc/tenpy))
- [x] Tensor views and slicing
- [x] Sorting

## Roadmap
- [] Chain expressions
- [] Automatic differentiation
- [] Sparse tensors
- [] Save and load binary data
- [] Dataframe
- [] Match and fuse operations
- [] Inplace operations
- [] CI/CD with tests
- [] Python documentation
- [] C++ doygen documentation
- [] C++ API documentation

## Long term
- [] Linear algebra and least squares
- [] Combinatorics
- [] Quasi random sequences
- [] Tensor broadcasting
- [] Statistics and hypothesis testing
- [] Optimization
- [] Machine learning
- [] Basic feed forward neural networks
- [] Neural netowrks (Convolution, RNNs, LSTM, ...)
- [] Ordinary differential equations (ODEs)
- [] Partial differential equations (PDEs)
- [] Time series
- [] Quantitative finance

## Requirements

- Clang compiler with C++23 support

- CMake

- [Optional] BLAS library (OpenBlas or BLIS)

- [Optional] LAPACKE for linear algebra

# Examples

- Tensors

```c++
// Uninitialized tensors
ten::tensor<float> x({2, 3, 4});
// Assign and access indices
x(0, 1, 2) = 3.0f;
std::cout << x(0, 1, 2) << std::endl;
// Access linear indices
x[20] = 1.0f;
```

- Default float tensors

```c++
// Uninitialized float32 tensor
constexpr std::size_t n = 3;
ten::tensor x({n, n});
// Initialize
auto zeros = ten::zeros({n, n})
auto ones = ten::ones({n, n});
auto fill = ten::fill({n, n}, .5);
auto lin = ten::linear({n, n}, 0., 1.);
// Random
auto norm = ten::rand_norm({n, n});
auto unif = ten::rand_unif({n, n});
```

- Slicing, assignment, rows and columns

```c++
ten::tensor<float> x({3, 3});
// Assign a single value
x = 1.0f;
// Row and columns can be accessed using col and row
x.row(0) = 2.0f;
x.col(0) = 3.0f;
std::cout << x << std::endl;
// Slicing using sequences
using ten::seq;
using ten::last;
auto slice = x(seq(1, last), seq(1, 2));
slice = .3f;
std::cout << x << std::endl;
// Slicing using multidimensional sequences ten::mdseq
using ten::mdseq;
auto index = mdseq(seq(0, last), seq(0));
auto second_slice = x[index];
second_slice = .5f;
```

- Gemm with expressions matching

```c++
#include <ten/tensor>
#include <ten/io>

int main() {
   auto a = ten::range<float>({3, 3});
   auto b = ten::range<float>({3, 3});
   auto c = ten::range<float>({3, 3});

   c = a * b + c;
   std::cout << c << std::endl;
}
```

- QR factorization

```c++
#include <ten/tensor>
#include <ten/linalg>

int main() {
   auto a = ten::range<float>({4, 4});
   auto [q, r] = ten::linalg::qr(a);

   std::cout << q << std::endl;
   std::cout << r << std::endl;
}
```

- Automatic differentiation

```c++
#include <ten/tensor>
#include <ten/io>

int main() {
   ten::tensor<float> x({5}, {1.0f, 2.0f, 3.0f, 4.0f, 5.0f}, true);
   ten::tensor<float> y({5}, {1.0f, 2.0f, 3.0f, 4.0f, 5.0f}, true);
   auto z = x / y;
   auto t = ten::sum(z);
   t.eval();
   t.backward();
   std::cout << "Output = " << t.value() << std::endl;
   std::cout << "The gradients\n";
   std::cout << x.grad() << std::endl;
   std::cout << y.grad() << std::endl;
}
```

- Save and load binary data

```c++
#include <ten/tensor>
#include <ten/io>

int main() {
   auto x = ten::range<float>({3, 4});
   ten::io::save(x, "matrix.ten");
   auto y = ten::io::load<ten::tensor<float>>("matrix.ten").value();
   std::cout << "shape = " << y.shape() << std::endl;
   std::cout << "stride = " << y.strides() << std::endl;
   std::cout << "data = \n" << y << std::endl;
}
```

- Data frame

```c++
#include <ten/dataframe>

int main() {
    auto df = ten::read_csv("path/to/file.csv");
    std::cout << df << std::endl;
    // Select by column name
    std::cout << df[{"x"}] << std::endl;
    // Select by column index
    std::cout << df[{0, 1, 4}] << std::endl;
    // Select by row index and column name
    std::cout << df.select(ten::seq(0, 2), std::vector<std::string>{"x", "y"}) << std::endl;
}
```

# Building ten

## Build the examples

```shell
mkdir build-examples
cd build-examples
cmake .. -DCMAKE_CXX_COMPILER=clang++ -DTEN_EXAMPLES=ON
cmake --build . --
```

## Build the tests

```shell
mkdir build-tests
cd build-tests
cmake .. -DCMAKE_CXX_COMPILER=clang++ -DTEN_TESTS=ON
cmake --build . --
```

## Build the docs

```shell
mkdir build-docs
cd build-docs
cmake .. -DCMAKE_CXX_COMPILER=clang++ -DTEN_DOCS=ON
cmake --build . --
```
## Build and install the python bindings

**Required libraries**:
- BLAS (Openblas or BLIS)
- LAPACK

```shell
mkdir build-bindings
cd build-bindings
cmake .. -DCMAKE_CXX_COMPILER=clang++ -DTEN_PYTHON=ON
cmake --build . --
sudo make install
```

