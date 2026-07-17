// Copyright © 2026 Apple Inc.

#include "doctest/doctest.h"

#include "mlx/backend/metal/device.h"
#include "mlx/debug.h"
#include "mlx/debug_internal.h"
#include "mlx/mlx.h"

#include <thread>

using namespace mlx::core;

TEST_CASE("test metal debug group spans encoder boundaries") {
  auto stream = default_stream(Device::gpu);
  CHECK(debug::detail::groups(stream).empty());
  debug::push_group("logical_group", stream);

  auto first = add(array({1.0f}), array({2.0f}), stream);
  eval(first);
  auto& encoder = metal::get_command_encoder(stream);
  encoder.end_encoding();
  encoder.commit();

  auto second = multiply(first, array({3.0f}), stream);
  eval(second);
  debug::pop_group(stream);
  CHECK(debug::detail::groups(stream).empty());
  synchronize(stream);
}

TEST_CASE("test metal lazy debug scope crosses threads") {
  auto stream = new_thread_unsafe_stream(Device::gpu);
  auto arr = array({0.0f});
  {
    debug::ScopedLabel captured("captured");
    debug::ScopedGroup group("logical", stream);
    arr = add(array({1.0f}), array({2.0f}), stream);
  }
  std::thread worker([&] {
    debug::ScopedLabel unrelated("worker");
    eval(arr);
    synchronize(stream);
  });
  worker.join();
  CHECK(debug::detail::scope_context(arr) != nullptr);
}
