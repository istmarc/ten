#include <iostream>

#include <ten/tensor>
#include <ten/io>
#include <ten/math>

constexpr float lambda = .5;

std::function<float(float)> Finv = [](float y) {
   return -std::log(y)/lambda;
};

int main() {
   auto x = ten::inv_sample<float>(1000, Finv);
   std::cout << x << std::endl;
}
