#include <bit>
#include <cstdint>
#include <iostream>
#include <sstream>
#include <string>

#include "ortools/lp_data/lp_data.h"
#include "ortools/lp_data/sol_reader.h"

int main() {
  using namespace operations_research::glop;
  int n;
  if (!(std::cin >> n)) return 2;
  std::string line;
  std::getline(std::cin, line);
  LinearProgram model;
  for (int i = 0; i < n; ++i) {
    std::getline(std::cin, line);
    const ColIndex col = model.CreateNewVariable();
    if (line != "<empty>") model.SetVariableName(col, line);
  }
  std::ostringstream solution;
  solution << std::cin.rdbuf();
  const auto result = operations_research::ParseSolString(solution.str(), model);
  if (!result.ok()) {
    std::cout << "ERR " << result.status().message() << '\n';
    return 0;
  }
  std::cout << "OK";
  for (double value : *result) {
    std::cout << " " << std::hex << std::bit_cast<uint64_t>(value) << std::dec;
  }
  std::cout << '\n';
}
