// Copyright 2026 gloprs contributors
// Licensed under the Apache License, Version 2.0.

#include <iostream>

#include "ortools/util/stats.h"

int main() {
  using namespace operations_research;
  StatsGroup stats("TraceStats");
  RatioDistribution ratio("ratio", &stats);
  DoubleDistribution value("value", &stats);
  IntegerDistribution count("count", &stats);
  for (const double x : {0.125, 0.5, 0.875}) ratio.Add(x);
  for (const double x : {-1.25e-9, 3.5e4, 2.0}) value.Add(x);
  for (const int x : {-2, 7, 10}) count.Add(x);
  std::cout << stats.StatString();
  stats.Reset();
  std::cout << "after_reset " << stats.StatString().size() << '\n';
}
