#include <bit>
#include <cstdint>
#include <iostream>
#include <sstream>
#include <string>
#include "ortools/lp_data/lp_parser.h"

int main() {
  using namespace operations_research::glop;
  std::string mode, input; std::getline(std::cin, mode);
  std::ostringstream stream; stream << std::cin.rdbuf(); input = stream.str();
  if (!input.empty() && input.back() == '\n') input.pop_back();
  if (mode == "C") {
    const auto parsed = ParseConstraint(input);
    if (!parsed.ok()) { std::cout << "ERR " << parsed.status().message() << '\n'; return 0; }
    const auto& c = *parsed;
    std::cout << "OK " << c.name << ' ' << std::hex << std::bit_cast<uint64_t>(c.lower_bound)
              << ' ' << std::bit_cast<uint64_t>(c.upper_bound) << std::dec << ' ' << c.variable_names.size() << '\n';
    for (int i = 0; i < c.variable_names.size(); ++i)
      std::cout << c.variable_names[i] << ' ' << std::hex << std::bit_cast<uint64_t>(c.coefficients[i]) << std::dec << '\n';
  } else {
    LinearProgram lp; const bool ok = ParseLp(input, &lp);
    std::cout << (ok ? "OK\n" : "ERR\n") << lp.Dump();
  }
}
