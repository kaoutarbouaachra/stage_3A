/*
 * Software Name : EURO/ROADEF Challenge 2026 checker
 * SPDX-FileCopyrightText: Copyright (c) Orange SA
 * SPDX-License-Identifier: MIT
 *
 * This software is distributed under the MIT licence,
 * see the "LICENSE" file for more details or https://opensource.org/license/MIT
 *
 * Authors: see CONTRIBUTORS.md
 * Software description: checker for the EURO/ROADEF Challenge 2026
 */

/**
 * @file main.cpp
 * @brief Main entry point for the EURO/ROADEF Challenge 2026 checker.
 * @author Morgan Chopin (morgan.chopin@orange.com)
 *
 * This file contains the main checker program that validates the solutions
 * of the T-Adaptive Segment Routing (T-ASR) problem. It loads network topology, traffic matrices,
 * intervention scenarios, and segment routing paths, then performs validation checks
 * including budget constraints and segment routing simulations.
 *
 * @param argc Number of command-line arguments
 * @param argv Array of command-line argument strings
 * @return int Exit code (0 for success, -1 for failure)
 *
 */

#include "CLI11.hpp"
#include "checker.h"
#include "schemas.h"

/**
 * @brief Displays copyright and licensing information to standard output.
 *
 * Prints the checker version, authors information, copyright notice, and
 * MIT license information.
 *
 * @note This software is distributed under the MIT License.
 */
inline void copyright(bool use_ftxui = false) noexcept {
  if (use_ftxui) return;
  std::cout << CHECKER_NAME " (version : " CHECKER_VERSION ")\n";
  std::cout << CHECKER_COPYRIGHT << std::endl;
}

/**
 * @brief Main entry point for the network checker application.
 *
 * This program validates and evaluates traffic engineering solutions by simulating
 * segment routing with network interventions over multiple time slots.
 *
 * @param argc Number of command-line arguments
 * @param argv Array of command-line argument strings
 *
 * Required command-line arguments:
 * - --net: Path to network topology file
 * - --tm: Path to traffic matrix file
 * - --scenario: Path to interventions scenario file
 * - --srpaths: Path to segment routing paths file
 *
 * Optional flags:
 * - --verbose: Enable verbose logging output
 *
 * The program performs the following operations:
 * 1. Parses command-line arguments
 * 2. Loads and validates the traffic engineering instance (network + traffic matrix)
 * 3. Loads and validates the interventions scenario
 * 4. Loads and validates the segment routing paths solution
 * 5. Checks budget constraints for the routing scheme
 * 6. Simulates segment routing for each time slot with scheduled interventions
 * 7. Computes Maximum Link Utilization (MLU) for each time step
 *
 * @return 0 on success, -1 if validation fails at any step
 *
 */
int main(int argc, char** argv) {
  CLI::App app{CHECKER_NAME};

  std::string net_file;
  auto*       net_opt = app.add_option("--net", net_file, HELP_MSG_0)->check(CLI::ExistingFile);

  std::string tm_file;
  auto*       tm_opt = app.add_option("--tm", tm_file, HELP_MSG_1)->check(CLI::ExistingFile);

  std::string scenario_file;
  auto*       scenario_opt = app.add_option("--scenario", scenario_file, HELP_MSG_2)->check(CLI::ExistingFile);

  std::string srpaths_file;
  auto*       srpaths_opt = app.add_option("--srpaths", srpaths_file, HELP_MSG_3)->check(CLI::ExistingFile);

  bool verbose = false;
  app.add_flag("--verbose", verbose, HELP_MSG_4);

  std::string instance_name;
  auto*       instance_opt = app.add_option("--instance", instance_name, HELP_MSG_5);

  bool pretty = false;
  app.add_flag("--pretty", pretty, HELP_MSG_6);

  int i_max_decimal_places = 12;
  app.add_option("--max-decimal-places", i_max_decimal_places, HELP_MSG_7);

  // Make --instance and --net, --tm, --scenario, --srpaths mutually exclusive
  instance_opt->excludes(net_opt, tm_opt, scenario_opt, srpaths_opt);
  net_opt->excludes(instance_opt);
  tm_opt->excludes(instance_opt);
  scenario_opt->excludes(instance_opt);
  srpaths_opt->excludes(instance_opt);

  CLI11_PARSE(app, argc, argv);

  const bool use_ftxui = pretty && !verbose;

  if (verbose) {
    copyright(use_ftxui);

    nt::Logging::init(argc, argv);
    nt::Logging::setVerbosity(nt::Logging::NamedVerbosity::Verbosity_MAX);
    nt::Logging::addFile(CHECKER_LOGFILE, nt::Logging::FileMode::Truncate, nt::Logging::NamedVerbosity::Verbosity_MAX);
  } else {
    nt::Logging::setVerbosity(nt::Logging::NamedVerbosity::Verbosity_OFF);
  }

  if (!instance_name.empty()) {
    net_file = instance_name + "-net.json";
    tm_file = instance_name + "-tm.json";
    scenario_file = instance_name + "-scenario.json";
    srpaths_file = instance_name + "-srpaths.json";
  }

  if (net_file.empty() || tm_file.empty() || scenario_file.empty() || srpaths_file.empty()) {
    LOG_F(ERROR, ERR_MSG_23);
    if (net_file.empty()) LOG_F(ERROR, ERR_MSG_24);
    if (tm_file.empty()) LOG_F(ERROR, ERR_MSG_25);
    if (scenario_file.empty()) LOG_F(ERROR, ERR_MSG_26);
    if (srpaths_file.empty()) LOG_F(ERROR, ERR_MSG_27);
    LOG_F(ERROR, ERR_MSG_28);
    return -1;
  }

  Instance      inst;
  ResultBuilder result_builder(inst, use_ftxui);

  // Validate input files format and semantics before loading the instance
  if (!validateNetworkJsonFormat(net_file) || !validateTrafficMatrixJsonFormat(tm_file) || !validateScenarioJsonFormat(scenario_file) || !validateSrPathsJsonFormat(srpaths_file)) {
    result_builder.setValid(false);
    result_builder.display(i_max_decimal_places);
    return -1;
  }

  // Load TE instance
  if (!loadAndCheckTeInstance(net_file, tm_file, inst, result_builder)) {
    result_builder.setValid(false);
    result_builder.display(i_max_decimal_places);
    return -1;
  }

  // Load interventions scenario
  Scenario scenario;
  if (!loadAndCheckScenario(scenario_file, inst, scenario, result_builder)) {
    result_builder.setValid(false);
    result_builder.display(i_max_decimal_places);
    return -1;
  }

  // Load routing scheme
  RoutingScheme rs(inst);
  if (!loadAndCheckSrPaths(srpaths_file, scenario, inst, rs, result_builder)) {
    result_builder.setValid(false);
    result_builder.display(i_max_decimal_places);
    return -1;
  }

  // Validate budget constraints
  if (!checkBudgetConstraint(rs, inst, scenario, result_builder._i_total_cost)) {
    result_builder.setValid(false);
    result_builder.display(i_max_decimal_places);
    return -1;
  }

  // All validations passed
  result_builder.setValid(true);

  // Perform segment routing simulation
  if (!simulateSegmentRouting(inst, scenario, rs, result_builder)) {
    result_builder.setValid(false);
    result_builder.display(i_max_decimal_places);
    return -1;
  }

  result_builder.display(i_max_decimal_places);

  return 0;
}
