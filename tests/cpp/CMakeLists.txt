add_executable(torchpairwise_native_test native_api.cpp)
target_link_libraries(torchpairwise_native_test PRIVATE torchpairwise::torchpairwise)
add_test(NAME torchpairwise_native_cpu COMMAND torchpairwise_native_test)
if(TORCHPAIRWISE_WITH_CUDA)
  add_test(NAME torchpairwise_native_cuda COMMAND torchpairwise_native_test cuda)
endif()

# Reuse the existing standalone family checks rather than duplicating their
# numerical assertions in a second test suite.
foreach(_name vector_comparisons weighted_comparisons pointset_comparisons
    transport_comparisons structured_comparisons modality_comparisons
    sequence_comparisons streaming_reductions neighbor_masks dispatcher_boundaries)
  add_executable(torchpairwise_test_${_name}
    "${PROJECT_SOURCE_DIR}/examples/libtorch/${_name}.cpp")
  target_link_libraries(torchpairwise_test_${_name}
    PRIVATE torchpairwise::torchpairwise)
  add_test(NAME torchpairwise_${_name} COMMAND torchpairwise_test_${_name})
endforeach()
