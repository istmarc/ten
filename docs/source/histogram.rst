Histogram
==========

A histgram can be fit using ``ten::histogram`` class available in ``ten/ml``.

.. code-block:: cpp

   #include <ten/tensor>
   #include <ten/random>
   #include <ten/ml>
   #include <ten/io>

   int main() {

      ten::set_seed(1234);
      constexpr size_t n = 100;
      ten::tensor<float> x = rand_norm<float>({n});

      std::cout << "Fitting histogram.\n";
      ten::ml::histogram_options options{.standartize = true, .nbins = 10};
      ten::ml::histogram hist(options);
      hist.fit(x);
      auto [h, bins] = hist.hist();

      io::save_mtx(h, "hist.mtx");
      io::save_mtx(bins, "bins.mtx");
   }

