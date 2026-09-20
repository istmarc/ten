#ifndef TEN_SSHAPE_HXX
#define TEN_SSHAPE_HXX

#include <array>
#include <cstddef>
#include <initializer_list>
#include <utility>

#include <ten/types.hxx>

namespace ten {

namespace details {
template <std::size_t First, std::size_t... Rest> struct fold_mul {
  static constexpr std::size_t value = First * fold_mul<Rest...>::value;
};
template <std::size_t First> struct fold_mul<First> {
  static constexpr std::size_t value = First;
};

} // namespace details

/// \class sshape
///
/// Handles static shapes of a given rank.
template <std::size_t First, std::size_t... Rest> class sshape {
private:
  /// Rank (Number of dimensions)
  static constexpr std::size_t _rank = 1 + sizeof...(Rest);

  /// Size known at compile time
  static constexpr std::size_t _size{details::fold_mul<First, Rest...>::value};

  /// Stores the dimensions known at compile time
  /// The shape is equal to 0 for dynamic shapes
  static constexpr std::array<std::size_t, _rank> _dims{First, Rest...};

public:
  // TODO For deserialization
  // sshape(std::size_t size, std::array<std::size_t, _rank> &&dims)
  //    : _size(size), _dims(std::move(dims)) {}

  sshape() noexcept {}

  /*
  sshape(const sshape& s) {
     _dims = s._dims;
  }
  sshape(sshape&& s) {
     _dims = std::move(s._dims);
  }*/

  /// Returns the rank (number of dimensions)
  [[nodiscard]] inline static constexpr std::size_t rank() { return _rank; }

  /// Returns the size known at compile time
  [[nodiscard]] inline static constexpr std::size_t size() { return _size; }

  /// Returns the static dimension at index
  template <std::size_t Index>
  [[nodiscard]] inline static constexpr std::size_t size() {
    return _dims[Index];
  }

  [[nodiscard]] inline std::size_t size(std::size_t index) {
    return _dims[index];
  }

  template <class SShapeType>
  // TODO requires SShapeType must be a shape type
  bool operator==(const SShapeType &s) const noexcept {
    if (SShapeType::rank() != rank()) {
      return false;
    }
    if (SShapeType::size() != size()) {
      return false;
    }
    for (size_t i = 0; i < SShapeType::rank(); i++) {
      if (s.size(i) != size(i)) {
        return false;
      }
    }
    return true;
  }

  // overload << operator
  template <std::size_t... Ts>
  friend std::ostream &operator<<(std::ostream &, const sshape<Ts...> &);

  // TODO template <size_type... Ts>
  // friend bool serialize(std::ostream &os, shape<Ts...> &s);

  // TODO template <class Shape>
  //   requires(::ten::is_shape<Shape>::value)
  // friend Shape deserialize(std::istream &is);
};

/*
template <size_type... Ts> bool serialize(std::ostream &os, shape<Ts...> &s) {
   os.write(reinterpret_cast<char *>(&s._size), sizeof(s._size));
   os.write(reinterpret_cast<char *>(s._dims.data()),
            s._rank * sizeof(s._dims[0]));

   return os.good();
}*/

/*
template <class Shape>
   requires(::ten::is_shape<Shape>::value)
Shape deserialize(std::istream &is) {
   constexpr std::size_t rank = Shape::rank();
   std::size_t size;
   std::array<size_type, rank> dims;
   is.read(reinterpret_cast<char *>(&size), sizeof(size));
   is.read(reinterpret_cast<char *>(dims.data()), rank * sizeof(dims[0]));

   return Shape(size, std::move(dims));
}*/

namespace details {
// Compute the Nth static stride
template <std::size_t N, class SShape> struct nth_static_stride {
  static constexpr std::size_t value =
      nth_static_stride<N - 1, SShape>::value * SShape::template size<N - 1>();
};
template <class SShape> struct nth_static_stride<0, SShape> {
  static constexpr std::size_t value = 1;
};

// Strides stored in an array
template <class SShape, class Index> struct static_stride_array;
template <class SShape, std::size_t... Index>
struct static_stride_array<SShape, std::index_sequence<Index...>> {
  static constexpr std::array<std::size_t, sizeof...(Index)> value{
      nth_static_stride<Index, SShape>::value...};
};

// Compute the static strides
template <class SShape, storage_order Order>
consteval auto compute_static_strides() {
  if constexpr (Order == storage_order::col_major) {
    return static_stride_array<SShape,
                               std::make_index_sequence<SShape::rank()>>::value;
  }
  if constexpr (Order == storage_order::row_major) {
    std::array<std::size_t, SShape::rank()> res{};
    return res;
  }
}
} // namespace details

/// \class stride
/// Strides from shape
template <class SShape, storage_order Order> class sstride {
public:
  using shape_type = SShape;

private:
  // Static strides
  static constexpr std::array<std::size_t, SShape::rank()> _strides{
      details::compute_static_strides<SShape, Order>()};

public:
  // stride(std::array<size_type, shape_type::rank()> &&dims)
  //     : _strides(std::move(dims)) {}

  sstride() noexcept {}

  [[nodiscard]] static constexpr std::size_t size() { return SShape::size(); }

  template <std::size_t Index> static constexpr std::size_t size() {
    return _strides[Index];
  }

  [[nodiscard]] std::size_t size(std::size_t index) const {
    return _strides[index];
  }

  /// Returns the rank (number of dimensions)
  [[nodiscard]] inline static constexpr std::size_t rank() {
    return SShape::rank();
  }

  // Overload the << operator
  template <class SShapeType, storage_order StorageOrder>
  friend std::ostream &operator<<(std::ostream &os,
                                  const sstride<SShapeType, StorageOrder> &);

  /*
   template <class ShapeType, storage_order StorageOrder>
   // requires(::ten::is_shape<ShapeType>::value)
   friend bool serialize(std::ostream &os, stride<ShapeType, StorageOrder> &s);

   template <class StrideType>
      requires(ten::is_stride<StrideType>::value)
   friend StrideType deserialize(std::istream &is);
  */
};

/*
template <class ShapeType, storage_order StorageOrder>
// requires(::ten::is_shape<ShapeType>::value)
bool serialize(std::ostream &os, stride<ShapeType, StorageOrder> &s) {
   constexpr size_t rank = stride<ShapeType, StorageOrder>::rank();
   os.write(reinterpret_cast<char *>(s._strides.data()),
            rank * sizeof(s._strides[0]));

   return os.good();
}
template <class StrideType>
   requires(::ten::is_stride<StrideType>::value)
StrideType deserialize(std::istream &is) {
   constexpr size_t rank = StrideType::rank();
   std::array<size_type, rank> dims;
   is.read(reinterpret_cast<char *>(dims.data()), rank * sizeof(dims[0]));

   return StrideType(std::move(dims));
}*/

} // namespace ten
#endif
