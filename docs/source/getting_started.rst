Getting started
===============

Tensor
------

A ``ten::tensor<T>`` is a multidimensional array. The number of dimensions is theoretically unlimited.


By default a tensor is colum major that is its data is in a contiguous column. Support for row major tensors is limited.

.. code-block:: cpp

   #include <ten/tensor>

   int main(){
      ten::tensor<float> x({2, 3, 4});
      // Access indices with () operator
      x(0, 0, 1) = 1.f;
      // Access linear indices with [] operator
      x[0] = 1.f;
      x[23] = 1.f;
   }

Printing is supported for vectors and matrices.

.. code-block:: cpp

   #include <ten/tensor>
   #include <ten/io>

   int main() {
      constexpr std::size_t n = 3;
      ten::tensor<float> x({n, n});
      for (std::size_t i = 0; i < n; i++) {
         for (std::size_t j = 0; j < n; j++) {
            x(i,j) = i + j;
         }
      }
      // Print
      std::cout << x << std::endl;
   }

- The number of dimensions of a tensor is returned by the function ``rank()``.
- The ``nth`` dimension is returned by ``dim(n)``
- The size of the tensor is returned by ``size()``.

.. code-block:: cpp

   ten::tensor<float> x({2, 3, 4});
   std::cout << x.rank() << std::endl; // 3
   std::cout << x.dim(0) << std::endl; // 2
   std::cout << x.dim(1) << std::endl; // 3
   std::cout << x.dim(2) << std::endl; // 4
   std::cout << x.size() << std::endl; // 2*3*4 = 24


By default most functions don't copy the tensor, it can be copied by calling ``copy()``.

.. code-block:: cpp

   #include <ten/tensor>
   #include <ten/io>

   int main() {
      constexpr std::size_t m = 2;
      constexpr std::size_t n = 3;
      ten::tensor<float> x({m, n});
      for (std::size_t i = 0; i < m*n; i++) {
         x[i] = i;
      }
      auto y = x.copy();
      std::cout << y << std::endl;
   }

Constructors
------------

The following constructors are defined for vectors, matrices, and tensors:

Vector
------

.. code-block:: cpp

   // Uninitialized vector
   ten::vector<T> x({size});
   // Vector initialized with data
   ten::vector<T> y({size}, data);

Matrix
------

.. code-block:: cpp

   std::vector<std::size_t> shape = {2, 3};
   ten::matrix<float> x(shape);
   ten::matrix<float> y(shape, data);

Tensor
------

.. code-block:: cpp

   std::vector<std::size_t> shape = {2, 3, 4};
   ten::tensor<float> x(shape);
   ten::tensor<float> y(shape, data);

Special matrices
----------------

- Diagonal

.. code-block:: cpp

   #include <ten/tensor>
   #include <ten/io>

   int main() {
      constexpr std::size_t n = 3;
      ten::diagonal<float> x({n, n});
      for (std::size_t i = 0; i < n; i++) {
         x[i] = i;
      }
      std::cout << x << std::endl;
      std::cout << std::boolalpha << y.is_diagonal() << std::endl;
   }



- Transposed

.. code-block:: cpp

   constexpr std::size_t n = 3;
   ten::tensor<float> x({n, n});
   auto y = ten::transposed(x);
   std::cout << std::boolalpha << y.is_transposed() << std::endl;

- Symmetric

.. code-block:: cpp

   constexpr std::size_t n = 3;
   ten::tensor<float> x({n, n});
   auto y = ten::symmetric(x);
   std::cout << std::boolalpha << y.is_symmetric() << std::endl;

- Hermitian

.. code-block:: cpp

   constexpr std::size_t n = 3;
   ten::tensor<std::complex<float>> x({n, n});
   auto y = ten::hermitian(x);
   std::cout << std::boolalpha << y.is_hermitian() << std::endl;

- Lower triangular

.. code-block:: cpp

   constexpr std::size_t n = 3;
   ten::tensor<float> x({n, n});
   auto y = ten::lower_tr(x);
   std::cout << std::boolalpha << y.is_lower_tr() << std::endl;

- Upper triangular

.. code-block:: cpp

   constexpr std::size_t n = 3;
   ten::tensor<float> x({n, n});
   auto y = ten::upper_tr(x);
   std::cout << std::boolalpha << y.is_upper_tr() << std::endl;

