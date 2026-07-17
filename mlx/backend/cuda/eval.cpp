// Copyright © 2025 Apple Inc.

#include "mlx/backend/gpu/eval.h"
#include "mlx/backend/cuda/allocator.h"
#include "mlx/backend/cuda/cublas_utils.h"
#include "mlx/backend/cuda/cudnn_utils.h"
#include "mlx/backend/cuda/event.h"
#include "mlx/debug.h"
#include "mlx/debug_internal.h"
#include "mlx/primitives.h"
#include "mlx/scheduler.h"
#include "mlx/utils.h"

#include <nvtx3/nvToolsExt.h>
#include <nvtx3/nvtx3.hpp>

namespace {

class DebugRange {
 public:
  DebugRange(const mlx::core::array& arr) {
    auto stream = arr.primitive().stream();
    const auto* scope = mlx::core::debug::detail::execution_scope();
    std::string label = scope == nullptr
        ? mlx::core::debug::current_label()
        : scope->label;
    if ((scope != nullptr && !scope->groups.empty()) ||
        mlx::core::debug::detail::has_groups()) {
      const auto& groups = scope == nullptr
          ? mlx::core::debug::detail::groups(stream)
          : scope->groups;
      for (const auto& group : groups) {
        if (!label.empty()) {
          label += ':';
        }
        label += group;
      }
    }
    if (mlx::core::debug::detail::has_labels()) {
      for (const auto& input : arr.inputs()) {
        auto input_label = mlx::core::debug::detail::label(input);
        if (!input_label.empty()) {
          label += label.empty() ? "input=" : ":input=";
          label += input_label;
        }
      }
      auto output_label = mlx::core::debug::detail::label(arr);
      if (!output_label.empty()) {
        label += label.empty() ? "output=" : ":output=";
        label += output_label;
      }
    }
    if (!label.empty()) {
      label += ':';
      label += arr.primitive().name();
      nvtxRangePushA(label.c_str());
      active_ = true;
    }
  }
  ~DebugRange() {
    if (active_) {
      nvtxRangePop();
    }
  }

 private:
  bool active_{false};
};

} // namespace

namespace mlx::core::gpu {

void init() {
  // Force initialization of CUDA, so CUDA runtime get destroyed last.
  cudaFree(nullptr);
  // Make sure native resources get destroyed after CommandEncoder.
  mlx::core::cu::CudaEvent::init_pool();
  init_cublas_handles_cache();
  init_cudnn_handles_cache();
  init_cudnn_conv_cache();
  init_cudnn_sdpa_cache();
}

void new_stream(Stream s) {
  assert(s.device == Device::gpu);
  debug::remove_stream_label(s);
  debug::detail::clear_groups(s);
  auto& encoders = cu::get_command_encoders();
  auto& d = cu::device(s.device);
  encoders.try_emplace(s.index, d, s);
}

void new_thread_unsafe_stream(Stream s) {
  assert(s.device == Device::gpu);
  debug::remove_stream_label(s);
  debug::detail::clear_groups(s);
  auto& encoders = cu::get_global_command_encoders();
  auto& d = cu::device(s.device);
  encoders.try_emplace(s.index, d, s);
}

void eval(array& arr) {
  nvtx3::scoped_range r("gpu::eval");
  debug::detail::ScopedExecutionContext scope(debug::detail::scope_context(arr));
  DebugRange debug_range(arr);
  // Ensure CUDA context is active on this thread. Required when MLX is called
  // from threads that have not yet established a CUDA context (e.g. thread
  // pools, language runtimes that migrate work across OS threads).
  cu::device(arr.primitive().stream().device).make_current();
  auto outputs = arr.outputs();
  {
    // If the array is a tracer hold a reference
    // to its inputs so they don't get donated
    std::vector<array> inputs;
    if (arr.is_tracer()) {
      inputs = arr.inputs();
    }
    arr.primitive().eval_gpu(arr.inputs(), outputs);
  }

  auto& stream = arr.primitive().stream();
  auto& encoder = cu::get_command_encoder(stream);
  // Keep used buffers alive until kernel finishes running.
  for (auto& in : arr.inputs()) {
    // Except for the donated one.
    if (in.data_shared_ptr() != arr.data_shared_ptr()) {
      encoder.add_temporary(in);
    }
  }
  for (auto& s : arr.siblings()) {
    encoder.add_temporary(s);
  }

  if (encoder.needs_commit()) {
    scheduler::notify_new_task(stream);
    encoder.add_completed_handler(
        [stream]() { scheduler::notify_task_completion(stream); });
    encoder.commit();
  }
}

void finalize(Stream s) {
  nvtx3::scoped_range r("gpu::finalize");
  cu::get_command_encoder(s).commit();
}

void synchronize(Stream s) {
  nvtx3::scoped_range r("gpu::synchronize");
  cu::get_command_encoder(s).synchronize();
}

void clear_streams() {
  cu::get_command_encoders().clear();
  if (is_main_thread()) {
    cu::get_global_command_encoders().clear();
  }
}

} // namespace mlx::core::gpu
