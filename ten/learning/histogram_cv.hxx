#ifndef TEN_LEARNING_HISTOGRAM_HXX
#define TEN_LEARNING_HISTOGRAM_HXX

#include <limits>
#include <tuple>
#include <type_traits>

#include <ten/tensor>
#include <ten/sort>

namespace ten {
namespace ml {

struct histogram_options {
  bool cv = false;
  bool standartize = false;
  bool cumulative = false;
  size_t nbins = 0;
};

namespace details{
/// Fit a histogram with nbins bins
/// Returns the number of bins and the histogram
template<typename T>
std::tuple<ten::tensor<T>, ten::tensor<T>> fit_histogram(ten::tensor<T>& data, std::size_t nbins) {
    const std::size_t n = data.size();
      tensor<T> bins({nbins + 1});
      auto hist = ten::zeros<T>({nbins});
      // Set bins
      T min = ten::min(data).eval().value();
      T max = ten::max(data).eval().value();
      T width = (max - min) / T(nbins);
      bins[0] = min;
      bins[nbins] = max;
      for (size_t i = 1; i < nbins; i++) {
        bins[i] = bins[i - 1] + width;
      }
      for (size_t i = 0; i < n; i++) {
        for (size_t j = 0; j < nbins; j++) {
          if (j == nbins - 1) {
            hist[j] += 1.;
          } else if (bins[j] <= data[i] && data[i] < bins[j + 1]) {
            hist[j] += 1.;
            break;
          }
        }
      }
  return std::make_tuple(hist, bins);
}
}

template <typename T = float> class histogram {
  static_assert(std::is_floating_point_v<T>, "T must be a floating point");

private:
  /// Cross validation score
  T _score = T(0);
  tensor<T> _scores = tensor<T>::make_default();
  /// Bins
  tensor<T> _bins = tensor<T>::make_default();
  /// Histogram
  tensor<T> _hist = tensor<T>::make_default();
  /// Histogram options
  histogram_options _options;

public:
  explicit histogram(const histogram_options &options) : _options(options) {}
  explicit histogram(histogram_options &&options)
      : _options(std::move(options)) {}

  const ten::tensor<T> scores() const {return _scores;}

  // fit a histogram
  void fit(tensor<T> &data) {
    if (_options.nbins > 0 && _options.cv) {
      std::cerr << "Please choose either cross validation or number of bins to "
                   "fit histogram.\n";
      return;
    }
    if (_options.cv) {
      // Fit histogram using cross validation
      ten::tensor<T> bins = ten::tensor<T>::make_default();
      ten::tensor<T> h = ten::tensor<T>::make_default();
      std::cout << "Fitting first histogram.\n";
      auto [besth, bestbins] = details::fit_histogram(data, 1);
      std::cout << "Sizes (bins, hist) = " << bestbins.size() << " and " << besth.size() << std::endl;
      const std::size_t n = data.size();
      std::cout << "Sort the data.\n";
      // Sort the data
      auto x = ten::sort(data);
      std::cout << "Compute the cv score.\n";
      // Cross validation score
      // besth is of size 1
      // var(x-h) = E((x - E(x-h)**2))
      T score = T(0);
      for (std::size_t i = 0; i < n; i++) {
        T v = x[i] - besth[0];
        score += v*v;
      }
      score /= n;
      std::cout << "First cv score = " << score << std::endl;
      _scores = ten::tensor<T>({n});
      _scores[0] = score;
      std::size_t index = 0;
      for (std::size_t i = 1; i < n; i++) {
        std::cout << "Fitting histogram with " << i + 1 << "bins.\n";
        // Fit the histogram
        auto res = details::fit_histogram(data, i+1);
        h = std::get<0>(res);
        bins = std::get<1>(res);
        std::cout << "Computing the score.\n";
        // Compute the cross validation score
        T s = T(0);
        for (std::size_t j = 0; j < n; j++) {
          for (std::size_t k = 0; k < bins.size(); k++) {
            if (bins[k] <= x[j] && x[j] <= bins[k+1]) {
              T v = x[j] - h[k];
              s += v*v;
              break;
            }
          }
        }
        s /= n;
        std::cout << "Score = " << s << std::endl;
        _scores[i] = s;
        std::cout << "Update\n";
        if (s < score) {
          // Update bestbins and besth
          besth = h.copy();
          bestbins = bins.copy();
          // Update score and index
          score = s;
          index = i;
        }
      }
      // Set the cross validation score and the historgam
      _hist = besth;
      _bins = bestbins;
      _score = score;
    } else {
      size_t n = data.size();
      size_t nbins = 0;
      if (_options.nbins == 0 && !_options.cv) {
        // Not cross validated set nbins to ceil(sqrt(n))
        nbins = size_t(std::ceil(std::sqrt(n)));
      } else if (_options.nbins > 0) {
        // Specified nbins
        nbins = _options.nbins;
      }
      auto res = details::fit_histogram(data, nbins);
      _hist = std::get<0>(res);
      _bins = std::get<1>(res);
    }
    std::cout << "Standartize the histogram\n";
    // Standartize the histogram
    if (_options.standartize) {
      for (std::size_t i = 0; i < _hist.size(); i++) {
        _hist[i] /= data.size();
      }
    }
    std::cout << "End of fit.\n";
  }

  std::tuple<tensor<T>, tensor<T>> hist() {
    return std::make_tuple(_hist, _bins);
  }
};

} // namespace ml
} // namespace ten

#endif
