cmake_minimum_required(VERSION 3.25)

file(STRINGS "${CMAKE_CURRENT_LIST_DIR}/torchpairwise/_version.py" _version_line
     REGEX "^__version__ = ")
string(REGEX MATCH "[0-9]+\\.[0-9]+\\.[0-9]+" _version "${_version_line}")
project(torchpairwise VERSION "${_version}" LANGUAGES CXX)

include(GNUInstallDirs)
include(CMakePackageConfigHelpers)
find_package(Torch REQUIRED)

option(TORCHPAIRWISE_WITH_CUDA "Build specialized CUDA operators" OFF)
option(TORCHPAIRWISE_WITH_OPENMP "Parallelize specialized CPU operators" ON)
option(TORCHPAIRWISE_BUILD_TESTS "Build standalone native correctness tests" OFF)

file(READ "${CMAKE_CURRENT_SOURCE_DIR}/torchpairwise/csrc/sources.json" _sources)
set_property(DIRECTORY APPEND PROPERTY CMAKE_CONFIGURE_DEPENDS
             "${CMAKE_CURRENT_SOURCE_DIR}/torchpairwise/csrc/sources.json")
set(_groups common cpu)
if(TORCHPAIRWISE_WITH_CUDA)
  enable_language(CUDA)
  find_package(CUDAToolkit REQUIRED)
  if(NOT TARGET torch_cuda)
    message(FATAL_ERROR "CUDA operators require a CUDA-enabled LibTorch")
  endif()
  list(APPEND _groups cuda)
endif()
foreach(_group IN LISTS _groups)
  string(JSON _count LENGTH "${_sources}" "${_group}")
  math(EXPR _last "${_count} - 1")
  foreach(_index RANGE 0 ${_last})
    string(JSON _source GET "${_sources}" "${_group}" ${_index})
    list(APPEND _native_sources "torchpairwise/csrc/${_source}")
  endforeach()
endforeach()

# A shared target retains operator-registration translation units. No Python
# extension/embedding library or Python development headers are required.
add_library(torchpairwise SHARED ${_native_sources})
add_library(torchpairwise::torchpairwise ALIAS torchpairwise)
target_compile_features(torchpairwise PUBLIC cxx_std_20)
target_include_directories(torchpairwise PUBLIC
  $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/torchpairwise/csrc>
  $<INSTALL_INTERFACE:${CMAKE_INSTALL_INCLUDEDIR}/torchpairwise>)
target_link_libraries(torchpairwise PUBLIC torch)
set_target_properties(torchpairwise PROPERTIES
  VERSION "${PROJECT_VERSION}" SOVERSION "${PROJECT_VERSION_MAJOR}"
  CXX_EXTENSIONS OFF)

if(TORCHPAIRWISE_WITH_OPENMP)
  find_package(OpenMP REQUIRED COMPONENTS CXX)
  target_link_libraries(torchpairwise PRIVATE OpenMP::OpenMP_CXX)
  target_compile_definitions(torchpairwise PRIVATE AT_PARALLEL_OPENMP=1)
endif()
if(TORCHPAIRWISE_WITH_CUDA)
  target_compile_definitions(torchpairwise PRIVATE WITH_CUDA)
  include(cmake/CudaArchitectureInfo.cmake)
  get_target_property(_architectures torchpairwise CUDA_ARCHITECTURES)
  # LibTorch may supply gencode flags and disable CUDA_ARCHITECTURES. Include
  # both controls; configuration-specific custom flags are conservatively
  # unknown because this query has one value for the whole binary.
  set(_arch_flags "${CMAKE_CUDA_FLAGS}")
  foreach(_config DEBUG RELEASE RELWITHDEBINFO MINSIZEREL)
    if(CMAKE_CUDA_FLAGS_${_config} MATCHES "gencode|generate-code|gpu-code|-code|gpu-architecture|-arch")
      string(APPEND _arch_flags " -arch=unknown")
    endif()
  endforeach()
  torchpairwise_cuda_arch_targets(_compiled_arches "${_arch_flags}" "${_architectures}")
  set_property(SOURCE torchpairwise/csrc/torchpairwise.cpp APPEND PROPERTY
    COMPILE_DEFINITIONS "TORCHPAIRWISE_CUDA_ARCH_FLAGS=\"${_compiled_arches}\"")
  # Match PyTorch's CUDAExtension defaults. CUDA's half overloads otherwise
  # compete with c10::Half arithmetic in the shared kernel sources.
  target_compile_definitions(torchpairwise PRIVATE
    $<$<COMPILE_LANGUAGE:CUDA>:__CUDA_NO_HALF_OPERATORS__>
    $<$<COMPILE_LANGUAGE:CUDA>:__CUDA_NO_HALF_CONVERSIONS__>
    $<$<COMPILE_LANGUAGE:CUDA>:__CUDA_NO_BFLOAT16_CONVERSIONS__>
    $<$<COMPILE_LANGUAGE:CUDA>:__CUDA_NO_HALF2_OPERATORS__>)
  target_link_libraries(torchpairwise PRIVATE CUDA::cudart)
  set_target_properties(torchpairwise PROPERTIES
    CUDA_STANDARD 20 CUDA_STANDARD_REQUIRED ON CUDA_EXTENSIONS OFF
    CUDA_RUNTIME_LIBRARY Shared)
  target_compile_options(torchpairwise PRIVATE
    $<$<COMPILE_LANGUAGE:CUDA>:--expt-relaxed-constexpr>)
endif()

install(TARGETS torchpairwise EXPORT torchpairwiseTargets
  RUNTIME DESTINATION "${CMAKE_INSTALL_BINDIR}"
  LIBRARY DESTINATION "${CMAKE_INSTALL_LIBDIR}"
  ARCHIVE DESTINATION "${CMAKE_INSTALL_LIBDIR}")
install(DIRECTORY torchpairwise/csrc/
  DESTINATION "${CMAKE_INSTALL_INCLUDEDIR}/torchpairwise"
  FILES_MATCHING PATTERN "*.h" PATTERN "*.hpp" PATTERN "*.cuh")
set(_config_dir "${CMAKE_INSTALL_LIBDIR}/cmake/torchpairwise")
install(EXPORT torchpairwiseTargets NAMESPACE torchpairwise::
  DESTINATION "${_config_dir}")
configure_package_config_file(cmake/torchpairwiseConfig.cmake.in
  "${CMAKE_CURRENT_BINARY_DIR}/torchpairwiseConfig.cmake"
  INSTALL_DESTINATION "${_config_dir}")
write_basic_package_version_file(
  "${CMAKE_CURRENT_BINARY_DIR}/torchpairwiseConfigVersion.cmake"
  VERSION "${PROJECT_VERSION}" COMPATIBILITY SameMajorVersion)
install(FILES
  "${CMAKE_CURRENT_BINARY_DIR}/torchpairwiseConfig.cmake"
  "${CMAKE_CURRENT_BINARY_DIR}/torchpairwiseConfigVersion.cmake"
  DESTINATION "${_config_dir}")

if(TORCHPAIRWISE_BUILD_TESTS)
  enable_testing()
  add_subdirectory(tests/cpp)
endif()
