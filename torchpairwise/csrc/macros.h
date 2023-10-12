#pragma once

#if defined(_WIN32) && !defined(torchpairwise_BUILD_STATIC_LIBS)
#if defined(torchpairwise_EXPORTS)
#define TORCHPAIRWISE_API __declspec(dllexport)
#else
#define TORCHPAIRWISE_API __declspec(dllimport)
#endif
#else
#define TORCHPAIRWISE_API
#endif

#if (defined __cpp_inline_variables) || __cplusplus >= 201703L
#define TORCHPAIRWISE_INLINE_VARIABLE inline
#else
#ifdef _MSC_VER
#define TORCHPAIRWISE_INLINE_VARIABLE __declspec(selectany)
#define HINT_MSVC_LINKER_INCLUDE_SYMBOL
#else
#define TORCHPAIRWISE_INLINE_VARIABLE __attribute__((weak))
#endif
#endif
