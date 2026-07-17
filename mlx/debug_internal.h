// Copyright © 2026 Apple Inc.

#pragma once

#include <cstdint>
#include <memory>
#include <string>
#include <vector>

#include "mlx/stream.h"

namespace mlx::core {
class array;

namespace debug::detail {
extern thread_local bool scope_active;
struct ScopeContext;
MLX_API const std::shared_ptr<const ScopeContext>& scope_context(const array&);
struct ScopeContext {
  Stream stream;
  std::string label;
  std::vector<std::string> groups;
  uint64_t group_generation{0};

  explicit ScopeContext(Stream stream) : stream(stream) {}
};

MLX_API std::shared_ptr<const ScopeContext> capture_scope(Stream stream);
MLX_API const ScopeContext* execution_scope();

class MLX_API ScopedExecutionContext {
 public:
  explicit ScopedExecutionContext(const std::shared_ptr<const ScopeContext>&);
  ~ScopedExecutionContext();

  ScopedExecutionContext(const ScopedExecutionContext&) = delete;
  ScopedExecutionContext& operator=(const ScopedExecutionContext&) = delete;

 private:
  const ScopeContext* previous_;
};

MLX_API bool has_labels();
MLX_API std::string label(const array& arr);
MLX_API bool has_groups();

MLX_API uint64_t stream_label_generation();
MLX_API std::string stream_label(Stream stream);
MLX_API const std::vector<std::string>& groups(Stream stream);
MLX_API uint64_t group_generation(Stream stream);
MLX_API void clear_groups(Stream stream);

} // namespace debug::detail
} // namespace mlx::core
