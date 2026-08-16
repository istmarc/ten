#ifndef TEN_STORAGE_DENSE_STORAGE
#define TEN_STORAGE_DENSE_STORAGE

#include <memory>
#include <new>
#include <type_traits>

#include <ten/types.hxx>

namespace ten {

/// \class dense_storage
/// Dense array
template <typename T> class dense_storage final {
public:
  using value_type = T;
  using allocator_type = std::allocator<T>;

private:
  allocator_type _allocator{};
  size_type _size = 0;
  T *_data = nullptr;

public:
  // For serialization, deserialization
  dense_storage(allocator_type allocator, size_type size, T *data)
      : _allocator(allocator), _size(size), _data(data) {}

  // dense_storage() noexcept {}

  dense_storage(const size_t size) noexcept : _size(size) {
    using allocator_traits = std::allocator_traits<allocator_type>;
    _data = allocator_traits::allocate(_allocator, _size * sizeof(T));
  }

  ~dense_storage() {
    if (_data)
      delete[] _data;
  }

  [[nodiscard]] inline const T *data() const { return _data; }

  [[nodiscard]] inline T *data() { return _data; }

  [[nodiscard]] inline size_type size() const { return _size; }

  /// Get/Set the element at index
  [[nodiscard]] inline const T &operator[](size_type index) const noexcept {
    return _data[index];
  }
  [[nodiscard]] inline T &operator[](size_type index) noexcept {
    return _data[index];
  }

  template <typename Type>
  friend bool serialize(std::ostream &os, dense_storage<Type> &storage);

  template <class DenseStorage>
    requires(::ten::is_dense_storage_v<DenseStorage>)
  friend DenseStorage deserialize(std::ostream &os);
};

template <typename T>
bool serialize(std::ostream &os, dense_storage<T> &storage) {
  os.write(reinterpret_cast<char *>(&storage._allocator),
           sizeof(storage._allocator));
  os.write(reinterpret_cast<char *>(&storage._size), sizeof(storage._size));
  os.write(reinterpret_cast<char *>(storage._data), storage._size * sizeof(T));
  return os.good();
}

template <class DenseStorage>
  requires(::ten::is_dense_storage_v<DenseStorage>)
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
}

} // namespace ten

#endif
