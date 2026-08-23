#ifndef TEN_STORAGE_SPARSE_STORAGE_HXX
#define TEN_STORAGE_SPARSE_STORAGE_HXX

#include <unordered_map>

#include <ten/types.hxx>

namespace ten {
/// \class sparse_storage
/// Sparse COO array
template <typename T> class sparse_storage final {
public:
  using value_type = T;

private:
  std::size_t _size = 0;
  std::unordered_map<std::size_t, T> _data;

public:
  // For serialization, deserialization
  // sparse_storage(size_type size, std::unordered_map<size_type, T>&& data)
  //    : _size(size), _data(std::move(data)) {}

  sparse_storage() noexcept {}

  sparse_storage(const std::size_t size) noexcept : _size(size) {}

  ~sparse_storage() {}

  // [[nodiscard]] inline const T *data() const { return _data; }

  // [[nodiscard]] inline T *data() { return _data; }

  [[nodiscard]] inline std::size_t size() const { return _size; }

  /// Get/Set the element at index
  [[nodiscard]] inline const T &operator[](std::size_t index) const noexcept {
    return _data[index];
  }

  [[nodiscard]] inline T &operator[](std::size_t index) noexcept {
    return _data[index];
  }

  /// Set the value
  void set(std::size_t index, T value) { _data[index] = value; }

  // TODO serialize
  /**
  template <typename R, typename Allocator>
  friend bool serialize(std::ostream &os,
                        sparse_storage<R, Allocator> &storage);*/

  // TODO Deserializ
  /*template <class DenseStorage>
     requires(::ten::is_sparse_storage<DenseStorage>::value)
  friend DenseStorage deserialize(std::ostream &os);*/
};

/*
template <typename T, typename Allocator>
bool serialize(std::ostream &os, sparse_storage<T, Allocator> &storage) {
   os.write(reinterpret_cast<char *>(&storage._allocator),
            sizeof(storage._allocator));
   os.write(reinterpret_cast<char *>(&storage._size), sizeof(storage._size));
   os.write(reinterpret_cast<char *>(storage._data), storage._size * sizeof(T));
   return os.good();
}

template <class DenseStorage>
   requires(::ten::is_sparse_storage<DenseStorage>::value)
DenseStorage deserialize(std::istream &is) {
   using T = typename DenseStorage::value_type;
   using allocator_type = typename DenseStorage::allocator_type;
   allocator_type allocator;
   is.read(reinterpret_cast<char *>(&allocator), sizeof(allocator));
   size_t size = 0;
   is.read(reinterpret_cast<char *>(&size), sizeof(size));
   T *data = new T[size];
   is.read(reinterpret_cast<char *>(data), size * sizeof(T));

   return DenseStorage(allocator, size, std::move(data));
}*/

} // namespace ten

#endif
