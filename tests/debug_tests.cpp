// Copyright © 2026 Apple Inc.

#include "doctest/doctest.h"

#include <thread>

#include "mlx/debug.h"
#include "mlx/debug_internal.h"
#include "mlx/mlx.h"

using namespace mlx::core;

TEST_CASE("test debug label scopes") {
  CHECK(debug::current_label().empty());
  debug::pop_label();

  debug::push_label("outer");
  CHECK_EQ(debug::current_label(), "outer");
  {
    debug::ScopedLabel inner("inner");
    CHECK_EQ(debug::current_label(), "outer:inner");
  }
  CHECK_EQ(debug::current_label(), "outer");
  debug::pop_label();
  CHECK(debug::current_label().empty());
}

TEST_CASE("test debug labels are thread local") {
  debug::push_label("main");
  std::thread worker([] {
    CHECK(debug::current_label().empty());
    debug::push_label("worker");
    CHECK_EQ(debug::current_label(), "worker");
    debug::pop_label();
  });
  worker.join();
  CHECK_EQ(debug::current_label(), "main");
  debug::pop_label();
}

TEST_CASE("test debug array labels") {
  auto arr = array({1.0f});
  CHECK_FALSE(debug::detail::has_labels());
  debug::set_label(arr, "input");
  CHECK(debug::detail::has_labels());
  CHECK_EQ(debug::detail::label(arr), "input");
  debug::remove_label(arr);
  CHECK_FALSE(debug::detail::has_labels());
  CHECK(debug::detail::label(arr).empty());
}

TEST_CASE("test debug label before materialization") {
  auto arr = add(array({1.0f}), array({2.0f}));
  debug::set_label(arr, "result");
  CHECK_EQ(debug::detail::label(arr), "result");
  eval(arr);
  CHECK_EQ(debug::detail::label(arr), "result");
  debug::remove_label(arr);
}

TEST_CASE("test debug labels follow array identity") {
  auto first = add(array({1.0f}), array({2.0f}));
  auto alias = first;
  debug::set_label(first, "result");
  CHECK_EQ(debug::detail::label(alias), "result");

  alias = add(array({3.0f}), array({4.0f}));
  CHECK(debug::detail::label(alias).empty());
  debug::remove_label(first);
}

TEST_CASE("test debug stream state") {
  auto stream = default_stream(Device::cpu);
  auto generation = debug::detail::stream_label_generation();
  debug::set_stream_label(stream, "inference");
  CHECK(debug::detail::stream_label_generation() > generation);
  CHECK_EQ(debug::detail::stream_label(stream), "inference");
  std::thread worker(
      [stream] { CHECK_EQ(debug::detail::stream_label(stream), "inference"); });
  worker.join();

  CHECK(debug::detail::groups(stream).empty());
  debug::push_group("forward", stream);
  const std::vector<std::string> expected{"forward"};
  CHECK_EQ(debug::detail::groups(stream), expected);
  debug::pop_group(stream);
  debug::pop_group(stream);
  CHECK(debug::detail::groups(stream).empty());
}
