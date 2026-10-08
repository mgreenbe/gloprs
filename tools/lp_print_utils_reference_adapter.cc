// Copyright 2026 gloprs contributors
// Licensed under the Apache License, Version 2.0.

#include <bit>
#include <cmath>
#include <cstdint>
#include <iostream>
#include <limits>
#include <string>

#include "absl/strings/str_cat.h"
#include "ortools/lp_data/lp_print_utils.h"

int main() {
  using operations_research::glop::Stringify;
  using operations_research::glop::StringifyMonomial;
  int count;
  if (!(std::cin >> count)) return 2;
  for (int i = 0; i < count; ++i) {
    uint64_t bits;
    std::cin >> std::hex >> bits;
    const double value = std::bit_cast<double>(bits);
    const std::string number = Stringify(value);
    const std::string monomial = StringifyMonomial(value, "x", false);
    const std::string default_number = absl::StrCat(value);
    const std::string rational = std::isfinite(value) && std::abs(value) <= 1e12
                                     ? operations_research::glop::StringifyRational(
                                           value, std::numeric_limits<double>::epsilon())
                                     : "SKIP";
    std::cout << number.size() << ' ' << number << '\n';
    std::cout << monomial.size() << ' ' << monomial << '\n';
    std::cout << default_number.size() << ' ' << default_number << '\n';
    std::cout << rational.size() << ' ' << rational << '\n';
  }
  return 0;
}
