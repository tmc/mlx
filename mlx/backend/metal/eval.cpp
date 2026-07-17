// Copyright © 2023-2024 Apple Inc.
#include <memory>

#include "mlx/backend/gpu/eval.h"
#include "mlx/backend/metal/debug.h"
#include "mlx/backend/metal/device.h"
#include "mlx/backend/metal/utils.h"
#include "mlx/debug.h"
#include "mlx/debug_internal.h"
#include "mlx/primitives.h"
#include "mlx/scheduler.h"
#include "mlx/utils.h"

namespace mlx::core::gpu {

void init() {}

void new_stream(Stream s) {
  assert(s.device == Device::gpu);
  debug::remove_stream_label(s);
  debug::detail::clear_groups(s);
  auto& encoders = metal::get_command_encoders();
  auto& d = metal::device(s.device);
  encoders.try_emplace(s.index, d, s, d.residency_set());
}

void new_thread_unsafe_stream(Stream s) {
  assert(s.device == Device::gpu);
  debug::remove_stream_label(s);
  debug::detail::clear_groups(s);
  auto& encoders = metal::get_global_command_encoders();
  auto& d = metal::device(s.device);
  encoders.try_emplace(s.index, d, s, d.residency_set());
}

void eval(array& arr) {
  debug::detail::ScopedExecutionContext scope(debug::detail::scope_context(arr));
  auto pool = metal::new_scoped_memory_pool();
  auto s = arr.primitive().stream();
  auto& encoder = metal::get_command_encoder(s);
  auto* command_buffer = encoder.get_command_buffer();

  auto outputs = arr.outputs();
  {
    // If the array is a tracer hold a reference
    // to its inputs so they don't get donated
    std::vector<array> inputs;
    if (arr.is_tracer()) {
      inputs = arr.inputs();
    }

    const auto* scope_context = debug::detail::execution_scope();
    const auto& user_label = scope_context == nullptr
        ? debug::current_label()
        : scope_context->label;
    if (!user_label.empty()) {
      std::string label = user_label;
#ifdef MLX_METAL_DEBUG
      label += ':';
      label += arr.primitive().name();
#endif
      command_buffer->setLabel(
          NS::String::string(label.c_str(), NS::UTF8StringEncoding));
    }
#ifdef MLX_METAL_DEBUG
    else if (metal::automatic_debug_labels()) {
      auto label = std::string(arr.primitive().name());
      command_buffer->setLabel(
          NS::String::string(label.c_str(), NS::UTF8StringEncoding));
    }
#endif

    arr.primitive().eval_gpu(arr.inputs(), outputs);
  }
  std::unordered_set<std::shared_ptr<array::Data>> buffers;
  for (auto& in : arr.inputs()) {
    buffers.insert(in.data_shared_ptr());
  }
  for (auto& s : arr.siblings()) {
    buffers.insert(s.data_shared_ptr());
  }
  // Remove the output if it was donated to by an input
  if (auto it = buffers.find(arr.data_shared_ptr()); it != buffers.end()) {
    buffers.erase(it);
  }

  if (encoder.needs_commit()) {
    encoder.end_encoding();
    scheduler::notify_new_task(s);
    encoder.commit([s, buffers = std::move(buffers)]() {
      scheduler::notify_task_completion(s);
    });
  } else {
    command_buffer->addCompletedHandler(
        [buffers = std::move(buffers)](MTL::CommandBuffer* cbuf) {});
  }
}

void finalize(Stream s) {
  auto pool = metal::new_scoped_memory_pool();
  auto& encoder = metal::get_command_encoder(s);
  auto* cb = encoder.get_command_buffer();
  encoder.end_encoding();
  encoder.commit();
}

void synchronize(Stream s) {
  metal::get_command_encoder(s).synchronize();
}

void clear_streams() {
  metal::get_command_encoders().clear();
  if (is_main_thread()) {
    metal::get_global_command_encoders().clear();
  }
}

} // namespace mlx::core::gpu
