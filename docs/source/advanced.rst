Advanced
===============

Tensor classes
---------------

- Expression

The library use expressions which is a useful technique used by some numerical linear algbera softwares in C++. tensors are also defined as expression. All the operations on the tensors are expressions as well. This with combination of lazy evaluation allow expression matching and fuse operations.

- Storage order

By default a tensor is colum major but row major tensors can be created using the ``ten::tensor`` class.

.. code-block:: cpp

   ten::tensor<float> x({2, 3, 4}, false, ten::storage_order::col_major);

Or using the ``ten::row_major`` function.

.. code-block:: cpp

   auto x = row_major({2, 3, 4});

- Storage and Allocator

Storage of tensors in memory can have huge impact in performances. That's why a tensor has its elements stored in a contiguous vector. Different storage classes are defined and can be used to store all the elements of a tensor. For example a dense storage class is used for dense tensors. By default all elements are allocated using ``std::allocator<T>``.

- Indexing and slicing

Slicing operations are supported by using ``ten::seq`` and ``ten::mdseq``. They are limited to tensors of up to 5 dimensions.

.. code-block:: cpp

   using ten::seq;
   using ten::last;
   ten::tensor<float> x({2, 3, 4});
   // Slicing using seq
   auto slice = x(seq(0, 1), seq(0, 1), seq(0, last))
   // Assign to slices
   slice = 1.0f;

