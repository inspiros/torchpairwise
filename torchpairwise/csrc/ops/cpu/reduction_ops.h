#pragma once

#include <ATen/TensorAccessor.h>

#include "cpu_helpers.h"
#include "../common/reduction_ops.h"

namespace torchpairwise {
    namespace ops {
        template<ReductionOp op, typename scalar_t>
        constexpr scalar_t identity_value() {
            if constexpr (op == All)
                return static_cast<scalar_t>(true);
            if constexpr (op == Any)
                return static_cast<scalar_t>(false);
            if constexpr (op == Sum || op == Mean)
                return static_cast<scalar_t>(0);
            if constexpr (op == Prod)
                return static_cast<scalar_t>(1);
        }

        template<ReductionOp op, typename output_t, typename... input_ts>
        __forceinline__ output_t call(input_ts... args) {
            // logical
            if constexpr (op == All)
                return (... && args);
            if constexpr (op == Any)
                return (... || args);
            // arithmetics
            if constexpr (op == Sum)
                return (... + args);
            if constexpr (op == Prod)
                return (... * args);
            if constexpr (op == Mean)
                return (... + args) / static_cast<output_t>(sizeof...(args));
        }

        template<ReductionOp op, typename output_t, typename input_t, typename index_t>
        __forceinline__ output_t call(const at::TensorAccessor<input_t, 1> args) {
            output_t output = identity_value<op, output_t>();
            for (int64_t i = 0; i < args.size(0); i++) {
                // logical
                if constexpr (op == All)
                    output &= args[i];
                if constexpr (op == Any)
                    output |= args[i];
                // arithmetics
                if constexpr (op == Sum || op == Mean)
                    output += args[i];
                if constexpr (op == Prod)
                    output *= args[i];
            }
            if constexpr (op == Mean) {
                output /= static_cast<output_t>(args.size(0));
            }
            return output;
        }

        template<ReductionOp op, typename output_t, typename input_t>
        __forceinline__ void accumulate_call(output_t* val, input_t arg) {
            // logical
            if constexpr (op == All)
                *val &= arg;
            if constexpr (op == Any)
                *val |= arg;
            // arithmetics
            if constexpr (op == Sum || op == Mean)
                *val += arg;
            if constexpr (op == Prod)
                *val *= arg;
        }
    }
}
