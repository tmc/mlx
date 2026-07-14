// Copyright © 2026 Apple Inc.

#pragma once

#include <string>

#include "mlx/api.h"
#include "mlx/stream.h"

namespace mlx::core {
class array;

namespace debug {

MLX_API void push_label(const std::string& label);
MLX_API void pop_label();
MLX_API const std::string& current_label();

MLX_API void set_label(const array& arr, const std::string& label);
MLX_API void remove_label(const array& arr);

MLX_API void set_stream_label(Stream stream, const std::string& label);
MLX_API void push_group(const std::string& label, Stream stream);
MLX_API void pop_group(Stream stream);

class ScopedLabel {
 public:
  explicit ScopedLabel(const std::string& label) {
    push_label(label);
  }
  ~ScopedLabel() {
    pop_label();
  }

  ScopedLabel(const ScopedLabel&) = delete;
  ScopedLabel& operator=(const ScopedLabel&) = delete;
};

} // namespace debug
} // namespace mlx::core
