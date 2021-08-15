#pragma once

#include <chrono>
#include <unordered_map>
#include <vector>

namespace wavernn {

/**
 * A timer utility, used for recording how long C++ inference kernels spend in
 * different parts of the kernel.  Timers allow you to separate your kernel
 * into sections, labeled with human-readable string names, and record the
 * duration of the sections as the kernel is running.  If a section is entered
 * multiple times, the times are aggregated.
 *
 * For example:
 *
 *     // Create the timer.
 *     bool enableTimer = true;
 *     Timer timer(enableTimer);
 *
 *     // Use the timer in a few sections.
 *     timer.start("Section 1");
 *     doSomething();
 *
 *     for(int i = 0; i < 1000; i++) {
 *         timer.start("Second Section");
 *         doSomethingElse();
 *
 *         timer.start("Sec 3");
 *         something();
 *     }
 *
 *     // Print a breakdown of times.
 *     timer.print();
 *
 * This example will print a table along the lines of:
 *
 *     WaveRNN Kernel Timings
 *     ======================
 *     Section 1: 3 ms (0%)
 *     Second Section: 193 ms (14%)
 *     Sec 3: 1179 ms (86%)
 *     ======================
 *     Total: 1372 ms
 *     ======================
 */
class Timer {
 public:
  /// Create a new timer.
  /// @param timing whether to enable this timer. If a timer is not enabled,
  /// its methods do nothing. Turn off timers to reduce overhead.
  explicit Timer(bool timing) : enabled(timing) {}

  /// Enter a timer section. This resets the timer duration back to zero and
  /// starts timing a new section. If this section has been encountered before,
  /// the duration is accumulated.
  ///
  /// @param key Name of the section. A char* is used instead of an std::string
  /// to avoid overhead from memory allocation, copying, and comparison. This
  /// method is intended to be used with string literals.
  void start(const char *key);

  /// Stop timing the current section. Starting a new section or printing output
  /// does this implicitly, so this is only necessary if you intend to do
  /// something you explicitly do not want to include in your timings.
  void stop();

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
  void print();

 private:
  /// Whether or not to enable this timer. When not enabled, all methods do
  /// nothing. This flag allows users to easily minimize the overhead incurred
  /// by the timer.
  bool enabled;

  /// The currently active section name. When no section is active, this is
  /// nullptr.
  ///
  /// This uses a char* instead of an std::string to minimize overhead.
  //// Working with std::string incurs overhead due to copying, memory
  //// allocation, and string comparison, while working with a char* is less
  //// safe but much more efficient.
  const char *currentSectionName = nullptr;

  /// The time at which the last section began.
  std::chrono::steady_clock::time_point startTime =
      std::chrono::steady_clock::now();

  /// Aggregated timings across the sections. Keys are string section names
  /// and values are times measured in milliseconds.
  std::unordered_map<const char *, float> timings;

  /// The list of keys that have been used. These are stored in a separate
  /// vector so that when the timings are printed out, they can be printed in
  /// the same order as the sections were encountered.
  std::vector<const char *> keys;
};
}  // namespace wavernn
