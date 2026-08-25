Initialization
===============

zeros
-----

- Create a tensor of zeros

.. code-block:: cpp

   auto x = ten::zeros<float>({2, 3, 4});

ones
----

- Create a tensor of ones

.. code-block:: cpp

   auto x = ten::ones<float>({2, 3, 4});

fill
-----

- Create a tensor of filled with a single value

.. code-block:: cpp

   auto x = ten::fill<float>({2, 3, 4}, 1.0f);

range
-----

- Create a range tensor

.. code-block:: cpp

   auto x = ten::range<float>({2, 3, 4});

linear
------

- Create a linear tensor

.. code-block:: cpp

   auto x = ten::linear<float>({2, 3, 4}, 0., 10.);

TODO
----

- Logarithmic

- Geometric

