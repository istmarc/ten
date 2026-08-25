Operators overloading
=====================

The operators are defined in the `ten/tensor.hxx` and in `ten/functional.hxx`.

Elementwise operations
----------------------

Operators ``+``, ``-``, ``*``, and ``/`` are defined for tensors, matrices and vectors.

.. code-block:: cpp

   ten::tensor<float> x(shape);
   ten::tensor<float> y(shape);
   auto z = x op y;

Scalar tensor operations
------------------------

Operators ``+``, ``-``, ``*``, and ``/`` are overloaded for scalar - tensor, matrix and vector operations.

.. code-block:: cpp

   ten::tensor<float> x(shape);
   auto a = 1.0f op x;
   auto b = x op 1.0f;

.. code-block:: cpp

   ten::tensor<float> x({2, 3, 4});
   auto a = 1.0f - x;
   auto b = x - 1.0f;

- Multiplication

.. code-block:: cpp

   ten::tensor<float> x({2, 3, 4});
   auto a = 2.0f * x;
   auto b = x * 2.0f;

- Division

.. code-block:: cpp

   ten::tensor<float> x({2, 3, 4});
   auto a = 1.0f / x;
   auto b = x / 2.0f;

Matrix multiplication
---------------------

.. code-block:: cpp

   constexpr std::size_t m = 2;
   constexpr std::size_t n = 3;
   constexpr std::size_t p = 4;
   ten::tensor<float> x({m, n});
   ten::tensor<float> y({n, p});
   ten::tensor<float> z = x * y;

Matrix vector multiplication
----------------------------

.. code-block:: cpp

   constexpr std::size_t m = 2;
   constexpr std::size_t n = 3;
   ten::tensor<float> x({m, n});
   ten::tensor<float> y({n});
   ten::tensor<float> z = x * y;

