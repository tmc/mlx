// Copyright © 2026 Apple Inc.

#include <algorithm>
#include <chrono>
#include <cstdlib>
#include <iostream>
#include <string>
#include <vector>

#include "mlx/debug.h"
#include "mlx/mlx.h"

using namespace mlx::core;

namespace {

enum class Mode { off, label, group, operation };

Mode parse_mode(const char* value) {
  const std::string mode(value);
  if (mode == "label") {
    return Mode::label;
  }
  if (mode == "group") {
    return Mode::group;
  }
  if (mode == "operation") {
    return Mode::operation;
  }
  return Mode::off;
}

double run(Mode mode, int iterations) {
  const auto stream = default_stream(Device::cpu);
  const auto start = std::chrono::steady_clock::now();
  if (mode == Mode::label) {
    debug::ScopedLabel scope("benchmark");
    for (int i = 0; i < iterations; ++i) {
      auto value = add(array({1.0f}), array({2.0f}), stream);
      (void)value;
    }
  } else if (mode == Mode::group) {
    debug::ScopedGroup scope("benchmark", stream);
    for (int i = 0; i < iterations; ++i) {
      auto value = add(array({1.0f}), array({2.0f}), stream);
      (void)value;
    }
  } else {
    for (int i = 0; i < iterations; ++i) {
      auto value = add(array({1.0f}), array({2.0f}), stream);
      if (mode == Mode::operation) {
        debug::set_label(value, "operation");
        debug::remove_label(value);
      }
      (void)value;
    }
  }
  return std::chrono::duration<double, std::micro>(
             std::chrono::steady_clock::now() - start)
      .count();
}

} // namespace

int main(int argc, char** argv) {
  const auto mode = parse_mode(argc > 1 ? argv[1] : "off");
  const int repetitions = argc > 2 ? std::atoi(argv[2]) : 20;
  const int iterations = argc > 3 ? std::atoi(argv[3]) : 1000;
  std::vector<double> samples;
  samples.reserve(repetitions);
  for (int i = 0; i < repetitions; ++i) {
    samples.push_back(run(mode, iterations));
  }
  std::sort(samples.begin(), samples.end());
  const auto median = samples[samples.size() / 2];
  std::cout << "mode=" << (argc > 1 ? argv[1] : "off")
            << " repetitions=" << repetitions << " iterations=" << iterations
            << " median_us=" << median << " min_us=" << samples.front()
            << " max_us=" << samples.back() << '\n';
}
