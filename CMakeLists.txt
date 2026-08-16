cmake_minimum_required(VERSION 3.2)

cmake_policy(SET CMP0135 NEW)

project(ten VERSION 0.1.1
            DESCRIPTION "A header only C++23 tensor and mathematical library."
            HOMEPAGE_URL "https://github.com/istmarc/ten")

include(GNUInstallDirs)

# Compiler options
set(CMAKE_CXX_STANDARD 23)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -O2 -march=native")

add_library(ten INTERFACE)
target_include_directories(
  ten
  INTERFACE $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/ten>
			$<INSTALL_INTERFACE:${CMAKE_INSTALL_INCLUDEDIR}>
)
# Compiler features
target_compile_features(ten INTERFACE cxx_std_23)

# Target paths
install(TARGETS ten
        EXPORT ${PROJECT_NAME}_Targets
        ARCHIVE DESTINATION ${CMAKE_INSTALL_LIBDIR}
        LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR}
        RUNTIME DESTINATION ${CMAKE_INSTALL_BINDIR})

# Version
include(CMakePackageConfigHelpers)
write_basic_package_version_file("tenConfigVersion.cmake"
  VERSION ${PROJECT_VERSION}
  COMPATIBILITY SameMajorVersion)


configure_package_config_file(
  "${CMAKE_CURRENT_SOURCE_DIR}/cmake/${PROJECT_NAME}Config.cmake.in"
  "${CMAKE_CURRENT_BINARY_DIR}/${PROJECT_NAME}Config.cmake"
  INSTALL_DESTINATION
  ${CMAKE_INSTALL_DATADIR}/${PROJECT_NAME}/cmake
)

# Targets to install
install(EXPORT ${PROJECT_NAME}_Targets
  FILE ${PROJECT_NAME}Targets.cmake
  NAMESPACE ${PROJECT_NAME}::
  DESTINATION ${CMAKE_INSTALL_DATADIR}/${PROJECT_NAME}/cmake
)

# Files to install
install(FILES "${CMAKE_CURRENT_BINARY_DIR}/${PROJECT_NAME}Config.cmake"
  "${CMAKE_CURRENT_BINARY_DIR}/${PROJECT_NAME}ConfigVersion.cmake"
  DESTINATION ${CMAKE_INSTALL_DATADIR}/${PROJECT_NAME}/cmake)

# Install directory
install(DIRECTORY ${PROJECT_SOURCE_DIR}/ten DESTINATION include)

# Build the tests
option(TEN_TESTS "Build tests." OFF)
if (TEN_TESTS)
  add_subdirectory(tests)
endif()

# Build the examples
option(TEN_EXAMPLES "Build examples." OFF)
if (TEN_EXAMPLES)
  add_subdirectory(examples)
endif()

# Build the docs
option(TEN_DOCS "Build documentation." OFF)
if (TEN_DOCS)
  add_subdirectory(docs)
endif()

# Build the benchmarks
option(TEN_BENCHMARKS "Build documentation." OFF)
if (TEN_BENCHMARKS)
  add_subdirectory(benchmarks)
endif()

# Python bindings
option(TEN_PYTHON "Python bindings." OFF)
if (TEN_PYTHON)
  add_subdirectory(bindings)
endif()

