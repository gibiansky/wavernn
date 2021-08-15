#include "timer.h"

#include <iomanip>
#include <iostream>

namespace wavernn {

/// Enter a timer section. This resets the timer duration back to zero and
/// starts timing a new section. If this section has been encountered before,
/// the duration is accumulated.
///
/// @param key Name of the section. A char* is used instead of an std::string
/// to avoid overhead from memory allocation, copying, and comparison. This
/// method is intended to be used with string literals.
void Timer::start(const char *key) {
  // If the timer is disabled, every method should do nothing.
  if (!enabled) {
    return;
  }

  // Stop timing the previous section.
  stop();

  // Start timing a new section.
  currentSectionName = key;
  startTime = std::chrono::steady_clock::now();
}

/// Stop timing the current section. Starting a new section or printing output
/// does this implicitly, so this is only necessary if you intend to do
/// something you explicitly do not want to include in your timings.
void Timer::stop() {
  // If the timer is disabled, every method should do nothing.
  if (!enabled) {
    return;
  }

  // If no section is active, we can't stop timing.
  if (currentSectionName == nullptr) {
    return;
  }

  // Get the elapsed time in milliseconds since the last start() call.
  auto elapsedTime = std::chrono::steady_clock::now() - startTime;
  auto elapsedNanos =
      std::chrono::duration_cast<std::chrono::nanoseconds>(elapsedTime);
  float elapsedMs = elapsedNanos.count() / 1e6f;

  // If this is the first time we encounter this section, set it's time,
  // otherwise increment the existing time.
  if (timings.find(currentSectionName) == timings.end()) {
    timings[currentSectionName] = elapsedMs;
    keys.push_back(currentSectionName);
  } else {
    timings[currentSectionName] += elapsedMs;
  }
}

/// Log a display of time elapsed in each section to stdout.
/// The display will look roughly like this:
///
///    WaveRNN Kernel Timings
///    ======================
///    Section 1: 3 ms (0%)
///    Second Section: 193 ms (14%)
///    Sec 3: 1179 ms (86%)
///    ======================
///    Total: 1372 ms
///    ======================
void Timer::print() {
  // If the timer is disabled, every method should do nothing.
  if (!enabled) {
    return;
  }

  // Stop timing whatever section is active.
  stop();

  // Compute the total to be able to compute percentages.
  float total = 0.0f;
  for (const char *key : keys) {
    total += timings[key];
  }

  // Display the table with milliseconds and percentages.
  std::cout << std::setprecision(0) << std::fixed;
  std::cout << "WaveRNN Kernel Timings" << std::endl;
  std::cout << "======================" << std::endl;
  for (const char *key : keys) {
    std::cout << key << ": " << timings[key] << " ms ("
              << timings[key] / total * 100 << "%)" << std::endl;
  }
  std::cout << "======================" << std::endl;
  std::cout << "Total: " << total << " ms" << std::endl;
  std::cout << "======================" << std::endl;
}

}  // namespace wavernn
