// Copyright 2026 gloprs contributors
// Licensed under the Apache License, Version 2.0.
// Pinned Apple-libc++ equal-key ordering cases for the singleton crash.

#include <algorithm>
#include <iostream>
#include <numeric>
#include <vector>

int main(int argc, char** argv) {
  if (argc != 3) return 2;
  const int n = std::stoi(argv[1]);
  const int mode = std::stoi(argv[2]);
  std::vector<int> values(n);
  std::iota(values.begin(), values.end(), 0);
  auto key = [=](int index) {
    return mode == 0 ? index % 7 : mode == 1 ? (index * 37) % 19
                           : mode == 2 ? (index * index + 3 * index) % 11
                                       : index < n / 2 ? 0 : index % 3;
  };
  std::sort(values.begin(), values.end(), [&](int a, int b) { return key(a) < key(b); });
  for (int value : values) std::cout << value << '\n';
}
