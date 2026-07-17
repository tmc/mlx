// Copyright © 2025-2026 Apple Inc.

#include "mlx/backend/metal/debug.h"

#include <Metal/Metal.hpp>

#include <cstdlib>

#include "mlx/array.h"
#include "mlx/backend/metal/device.h"
#include "mlx/backend/metal/metal.h"
#include "mlx/debug.h"
#include "mlx/debug_internal.h"
#include "mlx/dtype_utils.h"

namespace mlx::core::metal {
namespace {

bool automatic_labels() {
  static const bool enabled = [] {
    auto value = std::getenv("MLX_DEBUG_AUTOMATIC_LABELS");
    return value != nullptr && std::string(value) != "0";
  }();
  return enabled;
}

} // namespace

bool automatic_debug_labels() {
#ifdef MLX_METAL_DEBUG
  return automatic_labels();
#else
  return false;
#endif
}

void set_encoder_debug_label(
    void* mtl_encoder,
    const std::string& operation_name) {
#ifdef MLX_METAL_DEBUG
  if (mtl_encoder == nullptr) {
    return;
  }
  auto pool = new_scoped_memory_pool();
  auto encoder = static_cast<MTL::ComputeCommandEncoder*>(mtl_encoder);
  const auto* scope = debug::detail::execution_scope();
  const auto& user_label =
      scope == nullptr ? debug::current_label() : scope->label;
  if (user_label.empty() && !automatic_labels()) {
    return;
  }
  auto label =
      user_label.empty() ? operation_name : user_label + ":" + operation_name;
  encoder->setLabel(NS::String::string(label.c_str(), NS::UTF8StringEncoding));
#else
  (void)mtl_encoder;
  (void)operation_name;
#endif
}

void set_buffer_debug_label(void* mtl_buffer, const void* array_ptr) {
  if (mtl_buffer == nullptr || array_ptr == nullptr) {
    return;
  }
  const auto& arr = *static_cast<const array*>(array_ptr);
  auto label = debug::detail::label(arr);
#ifdef MLX_METAL_DEBUG
  if (label.empty() && automatic_labels()) {
    label = "Array[";
    for (size_t i = 0; i < arr.shape().size(); ++i) {
      if (i != 0) {
        label += 'x';
      }
      label += std::to_string(arr.shape()[i]);
    }
    label += "]:";
    label += dtype_to_string(arr.dtype());
  }
#endif
  if (label.empty()) {
    return;
  }
  auto pool = new_scoped_memory_pool();
  auto buffer = static_cast<MTL::Buffer*>(mtl_buffer);
  buffer->setLabel(NS::String::string(label.c_str(), NS::UTF8StringEncoding));
}

} // namespace mlx::core::metal
