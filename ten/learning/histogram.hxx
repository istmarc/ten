#ifndef TEN_LEARNING_HISTOGRAM_HXX
#define TEN_LEARNING_HISTOGRAM_HXX

#include <tuple>
#include <type_traits>

#include <ten/sort>
#include <ten/tensor>

namespace ten {
namespace ml {

struct histogram_options {
  bool cv = false;
  bool standartize = false;
  bool cumulative = false;
  size_t nbins = 0;
};

namespace details {
/// Fit a histogram with nbins bins
/// Returns the number of bins and the histogram
template <typename T>
std::tuple<ten::tensor<T>, ten::tensor<T>> fit_histogram(ten::tensor<T> &data,
                                                         std::size_t nbins) {
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
} // namespace details

template <typename T = float> class histogram {
  static_assert(std::is_floating_point_v<T>, "T must be a floating point");

private:
  /// Cross validation score
  T _score = T(0);
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

  // fit a histogram
  void fit(tensor<T> &data) {
    if (_options.nbins > 0 && _options.cv) {
      std::cerr << "Please choose either cross validation or number of bins to "
                   "fit histogram.\n";
      return;
    }
    if (_options.cv) {
      std::cerr << "Cross validation histogram not yet implemented.\n";
      return;
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
    // Standartize the histogram
    if (_options.standartize) {
      for (std::size_t i = 0; i < _hist.size(); i++) {
        _hist[i] /= data.size();
      }
    }
  }

  std::tuple<tensor<T>, tensor<T>> hist() {
    return std::make_tuple(_hist, _bins);
  }
};

} // namespace ml
} // namespace ten

#endif
