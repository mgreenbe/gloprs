// Copyright 2026 gloprs contributors
// Licensed under the Apache License, Version 2.0.

#include <atomic>
#include <cmath>
#include <iomanip>
#include <iostream>
#include <limits>

#include "ortools/util/time_limit.h"

int main() {
  using operations_research::TimeLimit;
  const double infinity = std::numeric_limits<double>::infinity();
  std::cout << std::unitbuf;
  std::cout << std::setprecision(17) << std::boolalpha;

  TimeLimit limit(infinity, 1.0);
  std::cout << "initial " << limit.GetElapsedDeterministicTime() << ' '
            << limit.GetDeterministicTimeLeft() << ' ' << limit.LimitReached()
            << '\n';
  limit.AdvanceDeterministicTime(0.25);
  std::cout << "advanced " << limit.GetElapsedDeterministicTime() << ' '
            << limit.GetDeterministicTimeLeft() << ' ' << limit.LimitReached()
            << '\n';
  limit.AdvanceDeterministicTime(0.75);
  std::cout << "reached " << limit.GetElapsedDeterministicTime() << ' '
            << limit.GetDeterministicTimeLeft() << ' ' << limit.LimitReached()
            << '\n';
  limit.ChangeDeterministicLimit(2.0);
  std::cout << "extended " << limit.GetDeterministicLimit() << ' '
            << limit.GetDeterministicTimeLeft() << ' ' << limit.LimitReached()
            << '\n';

  std::atomic<bool> primary(false);
  std::atomic<bool> secondary(false);
  limit.RegisterExternalBooleanAsLimit(&primary);
  limit.RegisterSecondaryExternalBooleanAsLimit(&secondary);
  primary = true;
  std::cout << "primary " << limit.LimitReached() << '\n';
  primary = false;
  secondary = true;
  std::cout << "secondary " << limit.LimitReached() << '\n';
  secondary = false;
  std::cout << "external_cleared " << limit.LimitReached() << '\n';

  TimeLimit global(infinity, 0.4);
  std::atomic<bool> global_external(false);
  global.RegisterExternalBooleanAsLimit(&global_external);
  limit.MergeWithGlobalTimeLimit(&global);
  std::cout << "merged " << limit.GetElapsedDeterministicTime() << ' '
            << limit.GetDeterministicLimit() << ' '
            << limit.GetDeterministicTimeLeft() << ' ' << limit.LimitReached()
            << '\n';
  global_external = true;
  std::cout << "merged_external " << limit.LimitReached() << '\n';

  limit.ResetHistory();
  std::cout << "infinite_time_left " << std::isinf(limit.GetTimeLeft()) << '\n';
  TimeLimit zero_wall(0.0, infinity);
  std::cout << "zero_wall " << zero_wall.LimitReached() << ' '
            << zero_wall.GetTimeLeft() << ' ' << zero_wall.LimitReached()
            << '\n';
}
