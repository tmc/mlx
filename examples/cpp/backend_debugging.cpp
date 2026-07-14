// Copyright © 2026 Apple Inc.

#include "mlx/debug.h"
#include "mlx/mlx.h"

using namespace mlx::core;

int main() {
  auto x = array({1.0f, 2.0f, 3.0f, 4.0f}, {2, 2});
  auto y = array({5.0f, 6.0f, 7.0f, 8.0f}, {2, 2});
  auto stream = default_stream(default_device());

  debug::set_label(x, "input_x");
  debug::set_label(y, "input_y");
  debug::set_stream_label(stream, "example");
  debug::push_group("matrix", stream);
  {
    debug::ScopedLabel scope("forward");
    auto z = matmul(x, y, stream);
    eval(z);
  }
  debug::pop_group(stream);
  debug::remove_label(x);
  debug::remove_label(y);
}
