// Copyright © 2025 Apple Inc.

#pragma once

#include <string>

#include "mlx/api.h"

namespace mlx::core::metal {

MLX_API bool automatic_debug_labels();
MLX_API void set_encoder_debug_label(
    void* mtl_encoder,
    const std::string& operation_name);
MLX_API void set_buffer_debug_label(void* mtl_buffer, const void* array_ptr);

} // namespace mlx::core::metal
