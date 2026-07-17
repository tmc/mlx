// Copyright © 2026 Apple Inc.

#include "mlx/debug.h"

#include <atomic>
#include <memory>
#include <mutex>
#include <shared_mutex>
#include <unordered_map>
#include <vector>

#include "mlx/array.h"
#include "mlx/debug_internal.h"

namespace mlx::core::debug {

namespace detail {

struct ArrayAccess {
  static std::weak_ptr<const void> owner(const array& arr) {
    return arr.array_desc_;
  }
};

} // namespace detail

namespace {

struct StreamKey {
  Device::DeviceType type;
  int device;
  int stream;

  bool operator==(const StreamKey&) const = default;
};

struct StreamKeyHash {
  size_t operator()(const StreamKey& key) const {
    size_t h = static_cast<size_t>(key.type);
    h = h * 31 + static_cast<size_t>(key.device);
    return h * 31 + static_cast<size_t>(key.stream);
  }
};

StreamKey stream_key(Stream stream) {
  return {stream.device.type, stream.device.index, stream.index};
}

struct LabelStack {
  std::vector<std::string> stack;
  std::string current;
  uint64_t generation{0};

  void rebuild() {
    current.clear();
    for (size_t i = 0; i < stack.size(); ++i) {
      if (i != 0) {
        current += ':';
      }
      current += stack[i];
    }
  }
};

LabelStack& label_stack() {
  static thread_local LabelStack labels;
  return labels;
}

using GroupMap = std::unordered_map<StreamKey, LabelStack, StreamKeyHash>;

GroupMap& group_stacks() {
  static thread_local GroupMap groups;
  return groups;
}

std::atomic<uint64_t> group_generation_counter{0};

uint64_t next_group_generation() {
  return group_generation_counter.fetch_add(1, std::memory_order_relaxed) + 1;
}

struct ArrayLabel {
  std::weak_ptr<const void> owner;
  std::string label;
};

std::unordered_map<std::uintptr_t, ArrayLabel>& array_labels() {
  static std::unordered_map<std::uintptr_t, ArrayLabel> labels;
  return labels;
}

std::shared_mutex& array_labels_mutex() {
  static std::shared_mutex mutex;
  return mutex;
}

std::atomic<bool>& has_array_labels() {
  static std::atomic<bool> has_labels{false};
  return has_labels;
}

struct StreamLabels {
  std::mutex mutex;
  std::unordered_map<StreamKey, std::string, StreamKeyHash> labels;
  std::atomic<uint64_t> generation{0};
};

StreamLabels& stream_labels() {
  static StreamLabels labels;
  return labels;
}

bool same_owner(
    const std::weak_ptr<const void>& lhs,
    const std::weak_ptr<const void>& rhs) {
  return !lhs.owner_before(rhs) && !rhs.owner_before(lhs);
}

} // namespace

void push_label(const std::string& label) {
  auto& labels = label_stack();
  labels.stack.push_back(label);
  labels.rebuild();
  detail::scope_active = true;
}

void pop_label() {
  auto& labels = label_stack();
  if (!labels.stack.empty()) {
    labels.stack.pop_back();
    labels.rebuild();
  }
  detail::scope_active = !labels.current.empty() || !group_stacks().empty();
}

const std::string& current_label() {
  return label_stack().current;
}

void set_label(const array& arr, const std::string& label) {
  std::unique_lock lock(array_labels_mutex());
  for (auto it = array_labels().begin(); it != array_labels().end();) {
    if (it->second.owner.expired()) {
      it = array_labels().erase(it);
    } else {
      ++it;
    }
  }
  array_labels().insert_or_assign(
      arr.id(), ArrayLabel{detail::ArrayAccess::owner(arr), label});
  has_array_labels().store(true, std::memory_order_release);
}

void remove_label(const array& arr) {
  std::unique_lock lock(array_labels_mutex());
  array_labels().erase(arr.id());
  if (array_labels().empty()) {
    has_array_labels().store(false, std::memory_order_release);
  }
}

void set_stream_label(Stream stream, const std::string& label) {
  auto& labels = stream_labels();
  {
    std::lock_guard lock(labels.mutex);
    labels.labels.insert_or_assign(stream_key(stream), label);
  }
  labels.generation.fetch_add(1, std::memory_order_release);
}

void remove_stream_label(Stream stream) {
  auto& labels = stream_labels();
  {
    std::lock_guard lock(labels.mutex);
    labels.labels.erase(stream_key(stream));
  }
  labels.generation.fetch_add(1, std::memory_order_release);
}

void push_group(const std::string& label, Stream stream) {
  auto& groups = group_stacks()[stream_key(stream)];
  groups.stack.push_back(label);
  groups.generation = next_group_generation();
  groups.rebuild();
  detail::scope_active = true;
}

void pop_group(Stream stream) {
  auto it = group_stacks().find(stream_key(stream));
  if (it == group_stacks().end() || it->second.stack.empty()) {
    return;
  }
  it->second.stack.pop_back();
  it->second.generation = next_group_generation();
  it->second.rebuild();
  if (it->second.stack.empty()) {
    group_stacks().erase(it);
  }
  detail::scope_active = !label_stack().current.empty() || !group_stacks().empty();
}

namespace detail {

thread_local bool scope_active = false;
thread_local const ScopeContext* active_scope = nullptr;

const std::shared_ptr<const ScopeContext>& scope_context(const array& arr) {
  return arr.array_desc_->scope_context;
}

std::shared_ptr<const ScopeContext> capture_scope(Stream stream) {
  if (!detail::scope_active) {
    return nullptr;
  }
  auto context = std::make_shared<ScopeContext>(stream);
  context->label = label_stack().current;
  auto it = group_stacks().find(stream_key(stream));
  if (it != group_stacks().end()) {
    context->groups = it->second.stack;
    context->group_generation = it->second.generation;
  }
  if (context->label.empty() && context->groups.empty()) {
    return nullptr;
  }
  return context;
}

const ScopeContext* execution_scope() {
  return active_scope;
}

ScopedExecutionContext::ScopedExecutionContext(
    const std::shared_ptr<const ScopeContext>& context)
    : previous_(active_scope) {
  active_scope = context.get();
}

ScopedExecutionContext::~ScopedExecutionContext() {
  active_scope = previous_;
}

bool has_labels() {
  return has_array_labels().load(std::memory_order_relaxed);
}

bool has_groups() {
  return !group_stacks().empty();
}

std::string label(const array& arr) {
  if (!has_labels()) {
    return {};
  }
  auto owner = detail::ArrayAccess::owner(arr);
  {
    std::shared_lock lock(array_labels_mutex());
    auto it = array_labels().find(arr.id());
    if (it == array_labels().end()) {
      return {};
    }
    if (same_owner(it->second.owner, owner)) {
      return it->second.label;
    }
  }

  std::unique_lock lock(array_labels_mutex());
  auto it = array_labels().find(arr.id());
  if (it == array_labels().end()) {
    return {};
  }
  if (!same_owner(it->second.owner, owner)) {
    array_labels().erase(it);
    if (array_labels().empty()) {
      has_array_labels().store(false, std::memory_order_release);
    }
  }
  return {};
}

uint64_t stream_label_generation() {
  return stream_labels().generation.load(std::memory_order_relaxed);
}

std::string stream_label(Stream stream) {
  auto& labels = stream_labels();
  std::lock_guard lock(labels.mutex);
  auto it = labels.labels.find(stream_key(stream));
  return it == labels.labels.end() ? std::string{} : it->second;
}

const std::vector<std::string>& groups(Stream stream) {
  auto it = group_stacks().find(stream_key(stream));
  if (it == group_stacks().end()) {
    static thread_local const std::vector<std::string> empty;
    return empty;
  }
  return it->second.stack;
}

uint64_t group_generation(Stream stream) {
  auto it = group_stacks().find(stream_key(stream));
  return it == group_stacks().end() ? 0 : it->second.generation;
}

void clear_groups(Stream stream) {
  auto it = group_stacks().find(stream_key(stream));
  if (it == group_stacks().end()) {
    return;
  }
  next_group_generation();
  group_stacks().erase(it);
  detail::scope_active = !label_stack().current.empty() || !group_stacks().empty();
}

} // namespace detail
} // namespace mlx::core::debug
