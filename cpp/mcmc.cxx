#include "ten/ios/mtx.hxx"
#include <ten/tensor>
#include <ten/io>
#include <ten/random>
#include <ten/mcmc>

template<typename T>
auto mcmc(T xt, std::size_t n, T mu, T sigma) {
  std::function<T(T)> f = [mu, sigma](T x){
    T v = x-mu;
    return std::exp((-1. / (2 * sigma * sigma)) * (v*v));
  };

  ten::uniform<T> dist(-1., 1.);

  std::function<T(T)> g = [&dist](T y){
    return y + dist.sample();
  };

  return ten::mcmc(0.f, n, f, g);
}

template<typename T>
auto mcmc2(T xt, std::size_t n, T mu, T sigma) {
  std::function<T(T)> f = [mu, sigma](T x){
    T v = x-mu;
    return std::exp((-1. / (2 * sigma * sigma)) * (v*v));
  };

  return ten::mcmc(0.f, n, f);
}

int main() {
  constexpr std::size_t n = 1000;
  auto s = mcmc<float>(0., n, .5, 1.);
  std::cout << s << std::endl;
  ten::io::save_mtx(s, "mcmc_sample.mtx");

  auto s2 = mcmc2<float>(0., n, .5, 1.);
  std::cout << s2 << std::endl;
  ten::io::save_mtx(s2, "mcmc_sample2.mtx");
}
