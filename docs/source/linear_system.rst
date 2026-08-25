Linear system
=============


A linear system can be solved using ``ten::linalg::solve`` function.

.. code-block:: cpp

   #include <ten/tensor>
   #include <ten/io>
   #include <ten/random>
   #include <ten/linalg>

   int main() {
     ten::set_seed(123);
     constexpr std::size_t n = 10;
     auto A = ten::rand_norm<float>({n, n});
     auto b = ten::rand_norm<float>({n});

     auto xqr = ten::linalg::solve(A, b);
     std::cout << "Solve using QR decomposition\n";
     std::cout << xqr << std::endl;

     auto xlu = ten::linalg::solve(A, b, linalg::ls_method::lu);
     std::cout << "Solve using LU decomposition\n";
     std::cout << xlu << std::endl;

     auto xsvd = ten::linalg::solve(A, b, linalg::ls_method::svd);
     std::cout << "Solve using SVD factorization\n";
     std::cout << xsvd << std::endl;
   }

