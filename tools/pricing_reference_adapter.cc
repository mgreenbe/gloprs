// Copyright 2026 gloprs contributors
// Licensed under the Apache License, Version 2.0.
// Differential adapter for GLOP's DynamicMaximum pricing structure.

#include <iomanip>
#include <iostream>

#include "absl/random/random.h"
#include "ortools/glop/pricing.h"
#include "ortools/lp_data/lp_types.h"

int main() {
  using namespace operations_research::glop;
  int cases;
  if (!(std::cin >> cases)) return 2;
  absl::BitGen random;
  for (int test = 0; test < cases; ++test) {
    int size;
    int operations;
    std::cin >> size >> operations;
    DynamicMaximum<ColIndex> maximum(random);
    maximum.ClearAndResize(ColIndex(size));
    std::cout << "case";
    for (int operation = 0; operation < operations; ++operation) {
      int code;
      std::cin >> code;
      if (code == 0) {
        int position;
        double value;
        std::cin >> position >> value;
        maximum.AddOrUpdate(ColIndex(position), value);
      } else if (code == 1) {
        int position;
        std::cin >> position;
        maximum.Remove(ColIndex(position));
      } else if (code == 2) {
        maximum.StartDenseUpdates();
      } else if (code == 3) {
        int position;
        double value;
        std::cin >> position >> value;
        maximum.DenseAddOrUpdate(ColIndex(position), value);
      } else {
        std::cout << ' ' << maximum.GetMaximum().value();
      }
    }
    std::cout << '\n';
  }
  return 0;
}
