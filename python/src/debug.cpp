// Copyright © 2026 Apple Inc.

#include <nanobind/nanobind.h>
#include <nanobind/stl/optional.h>
#include <nanobind/stl/string.h>

#include "mlx/array.h"
#include "mlx/debug.h"
#include "mlx/device.h"
#include "mlx/stream.h"

namespace mx = mlx::core;
namespace nb = nanobind;
using namespace nb::literals;

namespace {

class LabelContext {
 public:
  explicit LabelContext(std::string label) : label_(std::move(label)) {}

  void enter() {
    if (!active_) {
      mx::debug::push_label(label_);
      active_ = true;
    }
  }
  void exit() {
    if (active_) {
      mx::debug::pop_label();
      active_ = false;
    }
  }

 private:
  std::string label_;
  bool active_{false};
};

class GroupContext {
 public:
  GroupContext(std::string label, mx::Stream stream)
      : label_(std::move(label)), stream_(stream) {}

  void enter() {
    if (!active_) {
      mx::debug::push_group(label_, stream_);
      active_ = true;
    }
  }
  void exit() {
    if (active_) {
      mx::debug::pop_group(stream_);
      active_ = false;
    }
  }

 private:
  std::string label_;
  mx::Stream stream_;
  bool active_{false};
};

template <typename T>
void bind_context(nb::class_<T>& cls) {
  cls.def(
         "__enter__",
         [](T& context) -> T& {
           context.enter();
           return context;
         },
         nb::rv_policy::reference_internal)
      .def(
          "__exit__",
          [](T& context,
             const std::optional<nb::type_object>&,
             const std::optional<nb::object>&,
             const std::optional<nb::object>&) { context.exit(); },
          "exc_type"_a = nb::none(),
          "exc_value"_a = nb::none(),
          "traceback"_a = nb::none());
}

} // namespace

void init_debug(nb::module_& m) {
  auto debug = m.def_submodule("debug", "Backend-independent debugging.");

  debug.def("set_label", &mx::debug::set_label, "array"_a, "label"_a);
  debug.def("remove_label", &mx::debug::remove_label, "array"_a);
  debug.def(
      "set_stream_label", &mx::debug::set_stream_label, "stream"_a, "label"_a);
  debug.def(
      "push_group",
      [](std::string label, std::optional<mx::Stream> stream) {
        auto selected =
            stream.value_or(mx::default_stream(mx::default_device()));
        mx::debug::push_group(label, selected);
      },
      "label"_a,
      "stream"_a = nb::none());
  debug.def(
      "pop_group",
      [](std::optional<mx::Stream> stream) {
        auto selected =
            stream.value_or(mx::default_stream(mx::default_device()));
        mx::debug::pop_group(selected);
      },
      "stream"_a = nb::none());

  nb::class_<LabelContext> label_context(debug, "_LabelContext");
  bind_context(label_context);
  nb::class_<GroupContext> group_context(debug, "_GroupContext");
  bind_context(group_context);

  debug.def("label", [](std::string label) {
    return LabelContext(std::move(label));
  });
  debug.def(
      "group",
      [](std::string label, std::optional<mx::Stream> stream) {
        auto selected =
            stream.value_or(mx::default_stream(mx::default_device()));
        return GroupContext(std::move(label), selected);
      },
      "label"_a,
      "stream"_a = nb::none());
}
