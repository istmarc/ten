#include "ten/ios/mtx.hxx"
#include <ten/tensor>
#include <ten/random>
#include <ten/ml>
#include <ten/io>

int main(){
   using namespace ten;


   set_seed(1234);
   normal norm;
   constexpr size_t n = 100;
    ten::tensor<float> x = rand_norm<float>({n});
   io::save_mtx(x, "vect_norm.mtx");


  {
    std::cout << "Normal default histogram.\n";
     ml::histogram_options options{.standartize = true, .nbins = 10};
     ml::histogram hist(options);
     hist.fit(x);
     auto [h, bins] = hist.hist();
     std::cout << "hist:\n";
    std::cout << h << std::endl;
     std::cout << "bins:\n";
    std::cout << bins << std::endl;

    io::save_mtx(h, "hist.mtx");
    io::save_mtx(bins, "bins.mtx");
  }

  /*
  {
    std::cout << "Cross validated histogram.\n";
     ml::histogram hist({.cv = true, .standartize = true});
     hist.fit(x);
     auto [h, bins] = hist.hist();
     std::cout << "hist:\n";
    std::cout << h << std::endl;
     std::cout << "bins:\n";
    std::cout << bins << std::endl;

    io::save_mtx(h, "cvhist.mtx");
    io::save_mtx(bins, "cvbins.mtx");
    auto scores = hist.scores();
    std::cout << scores << std::endl;
    io::save_mtx(scores, "scores.mtx");
  }*/

}
