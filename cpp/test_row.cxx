#include "ten/types.hxx"
#include <ten/tensor>

int main() {
   using RowTensor = ten::tensor<float, 3, ten::storage_order::row_major>;
   RowTensor x({2, 3, 4});

   using SRowTensor = ten::ranked_tensor<float, ten::shape<2, 3, 4>, ten::storage_order::row_major>;
   SRowTensor y;
}
