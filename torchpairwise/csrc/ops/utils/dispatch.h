#pragma once

#include <ATen/Dispatch.h>

// bool switch
#define TORCHPAIRWISE_DISPATCH_BOOL_NAME(NAME, VAL, ...) \
    if (!(VAL)) {                                  \
        static const bool NAME = false;            \
        __VA_ARGS__();                             \
    } else {                                       \
        static const bool NAME = true;             \
        __VA_ARGS__();                             \
    }

#define TORCHPAIRWISE_DISPATCH_BOOL(ARG1, ...) \
    TORCHPAIRWISE_DISPATCH_BOOL_NAME(ARG1, ARG1, __VA_ARGS__)

// scalar type
#define AT_DISPATCH_CASE_BOOLEAN_TYPE(...) \
  AT_DISPATCH_CASE(at::ScalarType::Bool, __VA_ARGS__)

#define AT_DISPATCH_BOOLEAN_TYPE(TYPE, NAME, ...) \
  AT_DISPATCH_SWITCH(TYPE, NAME, AT_DISPATCH_CASE_BOOLEAN_TYPE(__VA_ARGS__))

// index type
#define TORCHPAIRWISE_DISPATCH_INDEX_TYPE_CPU(N_KERNELS, ...) \
    using index_t = int64_t;                            \
    __VA_ARGS__();                                      \

#define TORCHPAIRWISE_DISPATCH_INDEX_TYPE_CUDA(N_KERNELS, ...) \
    if (((int64_t)N_KERNELS) > (1 << 31)) {              \
        using index_t = int64_t;                         \
        __VA_ARGS__();                                   \
    }                                                    \
    else {                                               \
        using index_t = int;                             \
        __VA_ARGS__();                                   \
    }

#define TORCHPAIRWISE_DISPATCH_INDEX_TYPE_DEVICE(N_KERNELS, DEVICE, ...) \
C10_CONCATENATE(TORCHPAIRWISE_DISPATCH_INDEX_TYPE_, DEVICE)(N_KERNELS, __VA_ARGS__)

#define TORCHPAIRWISE_DISPATCH_INDEX_TYPE(N_KERNELS, ...) \
    if (((int64_t)N_KERNELS) > (1 << 31)) {         \
        using index_t = int64_t;                    \
        __VA_ARGS__();                              \
    }                                               \
    else {                                          \
        using index_t = int;                        \
        __VA_ARGS__();                              \
    }

namespace torchpairwise {
    namespace ops {
        template<at::ScalarType IT>
        struct index_type {
            using type = int;
        };

        template<>
        struct index_type<at::ScalarType::Long> {
            using type = int64_t;
        };

        template<>
        struct index_type<at::ScalarType::Int> {
            using type = int;
        };

        template<at::ScalarType IT>
        using index_type_t = typename index_type<IT>::type;

        inline at::ScalarType get_index_type(int64_t n_kernels) {
            if ((n_kernels) > (1 << 31)) {
                return at::ScalarType::Long;
            } else {
                return at::ScalarType::Int;
            }
        }
    }
}
