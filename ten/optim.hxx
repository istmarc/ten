#ifndef TEN_OPTIM_HXX
#define TEN_OPTIM_HXX

#include <any>
#include <unordered_map>

#include <ten/io>
#include <ten/tensor>

namespace ten::optim {

template <class T = float> struct optimizer_params {
   std::unordered_map<std::string, std::any> _params;

   optimizer_params(std::unordered_map<std::string, std::any> parameters)
       : _params(parameters) {}

   template <class Type>
   void add_param(const std::string &name, std::shared_ptr<Type> x) {
      _params[name] = x;
   }

   void zero_grad() {
      for (auto [name, p] : _params) {
         // shared_ptr<ten::tensor<T>>
         if (typeid(std::shared_ptr<ten::tensor<T>>) == p.type()) {
            auto ptr = std::any_cast<std::shared_ptr<ten::tensor<T>>>(p);
            ptr->zero_grad();
         }
      }
   }
};

struct optimizer {};

template <class T = float> struct sgd : optimizer {
   optimizer_params<T> _params;
   T _lr;

   template <Tensor TensorType>
   void add_param(const std::string &name, std::shared_ptr<TensorType> x) {
      _params.add_param(name, x);
   }

   sgd(std::unordered_map<std::string, std::any> parameters,
       T learning_rate = 1e-3)
       : _params(parameters), _lr(learning_rate) {}

   void zero_grad() { _params.zero_grad(); }

   template <Tensor TensorType>
   void step_tensor(std::shared_ptr<TensorType> &ptr) {
      auto t = *ptr.get();
      for (size_t i = 0; i < t.size(); i++) {
         t[i] = t[i] - _lr * t.grad_value(i);
      }
   }

   // Step set theta = theta - lr * grad
   void step() {
      for (auto [name, p] : _params._params) {
         // shared_ptr<ten::tensor<T>>
         if (typeid(std::shared_ptr<ten::tensor<T>>) == p.type()) {
            auto ptr = std::any_cast<std::shared_ptr<ten::tensor<T>>>(p);
            step_tensor(ptr);
         }
      }
   }
};

} // namespace ten::optim

#endif
