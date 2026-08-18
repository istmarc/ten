#ifndef TENSEUR_IOS_TENSOR
#define TENSEUR_IOS_TENSOR

#include <iostream>
#include <ten/types.hxx>
#include <ten/utils.hxx>

namespace ten {

/// Overload for scalar
template <typename T>
std::ostream &operator<<(std::ostream &os, const ::ten::scalar<T> &s) {
  os << s.value();
  return os;
}

/// Overload << operator for vector
template <class T>
std::ostream &operator<<(std::ostream &os, const tensor<T> &t) {
  const size_t rank = t.rank();
  if (rank == 1) {
    os << "tensor<" << ::ten::to_string<T>() << ">"; // << t.shape() << ">";
    size_type size = t.size();
    if (size <= 10) {
      for (size_type i = 0; i < t.size(); i++) {
        os << "\n" << t[i];
      }
    } else {
      for (size_type i = 0; i < 5; i++) {
        os << "\n" << t[i];
      }
      os << "\n⋮";
      for (size_type i = t.size() - 5; i < t.size(); i++) {
        os << "\n" << t[i];
      }
    }
  } else if (rank == 2) {
    os << "tensor<" << ::ten::to_string<T>() << ">";
    // "," << t.shape() << ">";
    size_type m = t.dim(0);
    size_type n = t.dim(1);
    for (size_type i = 0; i < m; i++) {
      os << "\n";
      os << t(i, 0);
      for (size_type j = 1; j < n; j++) {
        os << "   " << t(i, j);
      }
    }
  } else {
    std::cerr << "Cannot print tensor of rank " << rank << ".\n";
  }
  return os;
}
/// TODO Overload << operator for diagonal matrix
/*template <class T, class Shape, storage_order order, class Storage,
          class Allocator>
   requires(::ten::is_diagonal<
            ranked_tensor<T, Shape, order, Storage, Allocator>>::value)
std::ostream &
operator<<(std::ostream &os,
           const ranked_tensor<T, Shape, order, Storage, Allocator> &t) {
   os << "diagonal<" << ::ten::to_string<T>() << "," << t.shape() << ">";
   size_type size = t.size();
   if (size <= 10) {
      for (size_type i = 0; i < size; i++) {
         os << "\n" << t[i];
      }
   } else {
      for (size_type i = 0; i < 5; i++) {
         os << "\n" << t[i];
      }
      os << "⋮\n";
      for (size_type i = size - 5; i < size; i++) {
         os << "\n" << t[i];
      }
   }

   return os;
}*/
/// TODO Overload << operator for column
/*
template <class T, class Shape, storage_order order, class Storage,
          class Allocator>
   requires(::ten::is_dynamic_column<
            ranked_column<T, Shape, order, Storage, Allocator>>::value)
std::ostream &
operator<<(std::ostream &os,
           const ranked_column<T, Shape, order, Storage, Allocator> &t) {
   os << "column<" << ::ten::to_string<T>() << "," << t.size() << ">";
   size_type size = t.size();
   if (size <= 10) {
      for (size_type i = 0; i < t.size(); i++) {
         os << "\n" << t[i];
      }
   } else {
      for (size_type i = 0; i < 5; i++) {
         os << "\n" << t[i];
      }
      os << "\n⋮";
      for (size_type i = t.size() - 5; i < t.size(); i++) {
         os << "\n" << t[i];
      }
   }
   return os;
}*/

/// TODO Overload << operator for row
/*
template <class T, class Shape, storage_order order, class Storage,
          class Allocator>
   requires(::ten::is_dynamic_row<
            ranked_row<T, Shape, order, Storage, Allocator>>::value)
std::ostream &
operator<<(std::ostream &os,
           const ranked_row<T, Shape, order, Storage, Allocator> &t) {
   os << "row<" << ::ten::to_string<T>() << "," << t.size() << ">\n";
   size_type size = t.size();
   if (size <= 10) {
      os << t[0];
      for (size_type i = 1; i < t.size(); i++) {
         os << " " << t[i];
      }
   } else {
      os << t[0];
      for (size_type i = 0; i < 5; i++) {
         os << " " << t[i];
      }
      os << "\n...";
      for (size_type i = t.size() - 5; i < t.size(); i++) {
         os << " " << t[i];
      }
   }
   return os;
}*/

} // namespace ten
#endif
