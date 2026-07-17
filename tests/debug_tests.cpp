// Copyright © 2026 Apple Inc.

#include "doctest/doctest.h"

#include <thread>
#include <utility>

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

TEST_CASE("test debug array labels are pruned") {
  for (int i = 0; i < 32; ++i) {
    auto transient = add(array({1.0f}), array({2.0f}));
    debug::set_label(transient, "transient");
  }
  auto live = array({3.0f});
  debug::set_label(live, "live");
  CHECK_EQ(debug::detail::label(live), "live");
  debug::remove_label(live);
  CHECK_FALSE(debug::detail::has_labels());
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
  debug::remove_stream_label(stream);
  CHECK(debug::detail::stream_label(stream).empty());
  std::thread worker_after_remove(
      [stream] { CHECK(debug::detail::stream_label(stream).empty()); });
  worker_after_remove.join();

  CHECK(debug::detail::groups(stream).empty());
  debug::push_group("forward", stream);
  const std::vector<std::string> expected{"forward"};
  CHECK_EQ(debug::detail::groups(stream), expected);
  debug::detail::clear_groups(stream);
  CHECK(debug::detail::groups(stream).empty());
}

TEST_CASE("test debug scoped group") {
  auto stream = default_stream(Device::cpu);
  CHECK(debug::detail::groups(stream).empty());
  {
    debug::ScopedGroup group("forward", stream);
    CHECK_EQ(debug::detail::groups(stream), std::vector<std::string>{"forward"});
    auto moved = std::move(group);
    CHECK_EQ(debug::detail::groups(stream), std::vector<std::string>{"forward"});
  }
  CHECK(debug::detail::groups(stream).empty());
}

TEST_CASE("test lazy arrays capture debug scope") {
  auto stream = default_stream(Device::cpu);
  auto unscoped = add(array({1.0f}), array({2.0f}), stream);
  CHECK(debug::detail::scope_context(unscoped) == nullptr);
  debug::ScopedLabel label("forward");
  debug::ScopedGroup group("layer", stream);
  auto arr = add(array({1.0f}), array({2.0f}), stream);
  auto context = debug::detail::scope_context(arr);
  REQUIRE(context != nullptr);
  CHECK_EQ(context->label, "forward");
  CHECK_EQ(context->groups, std::vector<std::string>{"layer"});
}

TEST_CASE("test debug sequential groups have distinct generations") {
  auto stream = default_stream(Device::cpu);
  debug::push_group("layer0", stream);
  auto first = add(array({1.0f}), array({2.0f}), stream);
  debug::pop_group(stream);

  debug::push_group("layer1", stream);
  auto second = add(array({3.0f}), array({4.0f}), stream);
  debug::pop_group(stream);

  auto first_scope = debug::detail::scope_context(first);
  auto second_scope = debug::detail::scope_context(second);
  REQUIRE(first_scope != nullptr);
  REQUIRE(second_scope != nullptr);
  CHECK_NE(first_scope->group_generation, second_scope->group_generation);
  CHECK_EQ(first_scope->groups, std::vector<std::string>{"layer0"});
  CHECK_EQ(second_scope->groups, std::vector<std::string>{"layer1"});
}

TEST_CASE("test lazy debug scope crosses threads") {
  auto stream = default_stream(Device::cpu);
  auto arr = array({0.0f});
  {
    debug::ScopedLabel label("forward");
    debug::ScopedGroup group("layer", stream);
    arr = add(array({1.0f}), array({2.0f}), stream);
  }
  std::thread worker([&arr] {
    debug::ScopedLabel unrelated("worker");
    debug::detail::ScopedExecutionContext scope(debug::detail::scope_context(arr));
    auto* context = debug::detail::execution_scope();
    CHECK(context != nullptr);
    if (context != nullptr) {
      CHECK_EQ(context->label, "forward");
      CHECK_EQ(context->groups, std::vector<std::string>{"layer"});
    }
    {
      debug::detail::ScopedExecutionContext empty(nullptr);
      CHECK(debug::detail::execution_scope() == nullptr);
    }
    CHECK(debug::detail::execution_scope() != nullptr);
  });
  worker.join();
}
