// Copyright © 2026 Apple Inc.

#pragma once

#include <cstdint>
#include <string>
#include <vector>

#include "mlx/stream.h"

namespace mlx::core {
class array;

namespace debug::detail {

MLX_API bool has_labels();
MLX_API std::string label(const array& arr);
MLX_API bool has_groups();

MLX_API uint64_t stream_label_generation();
MLX_API std::string stream_label(Stream stream);
MLX_API const std::vector<std::string>& groups(Stream stream);
MLX_API uint64_t group_generation(Stream stream);

} // namespace debug::detail
} // namespace mlx::core
