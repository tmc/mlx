// Copyright © 2026 Apple Inc.

#include "doctest/doctest.h"

#include "mlx/backend/metal/device.h"
#include "mlx/debug.h"
#include "mlx/mlx.h"

using namespace mlx::core;

TEST_CASE("test metal debug group spans encoder boundaries") {
  auto stream = default_stream(Device::gpu);
  debug::push_group("logical_group", stream);

  auto first = add(array({1.0f}), array({2.0f}), stream);
  eval(first);
  auto& encoder = metal::get_command_encoder(stream);
  encoder.end_encoding();
  encoder.commit();

  auto second = multiply(first, array({3.0f}), stream);
  eval(second);
  debug::pop_group(stream);
  synchronize(stream);
}
