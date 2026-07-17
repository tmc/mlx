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
/** Remove the label associated with stream, if any. */
MLX_API void remove_stream_label(Stream stream);
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

class ScopedGroup {
 public:
  // Push label on construction and pop it on destruction for stream.
  // Groups nest in construction order. A moved-from scope is inactive. There
  // is no zero-value scope; construct one with a label and stream.
  ScopedGroup(const std::string& label, Stream stream)
      : stream_(stream), active_(true) {
    push_group(label, stream_);
  }
  ~ScopedGroup() {
    if (active_) {
      pop_group(stream_);
    }
  }

  ScopedGroup(const ScopedGroup&) = delete;
  ScopedGroup& operator=(const ScopedGroup&) = delete;

  ScopedGroup(ScopedGroup&& other) noexcept
      : stream_(other.stream_), active_(other.active_) {
    other.active_ = false;
  }
  ScopedGroup& operator=(ScopedGroup&&) = delete;

 private:
  Stream stream_;
  bool active_;
};

} // namespace debug
} // namespace mlx::core
