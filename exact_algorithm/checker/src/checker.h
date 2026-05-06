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
 * @file checker.h
 * @brief Main header file for the EURO/ROADEF Challenge 2026 checker.
 *
 * @author Morgan Chopin (morgan.chopin@orange.com)
 */

#ifndef _CHECKER_H_
#define _CHECKER_H_

#include "core/arrays.h"

#include "graphs/algorithms/connectivity.h"
#include "graphs/io/networkx_io.h"

#include "te/algorithms/cost_functions.h"
#include "te/algorithms/segment_routing.h"

#include "config.h"
#include "types.h"
#include "helpers.h"
#include "validation.h"

/**
 * @brief Loads and validates a Traffic Engineering (TE) instance from network and traffic matrix files.
 *
 * This function reads a network topology from a network file and traffic demands from a traffic matrix file,
 * validates the data, and populates the provided Instance object.
 *
 * @param net_file Path to the network topology file (expected to be in a format compatible with NetworkxIo)
 * @param tm_file Path to the traffic matrix file (expected to be a JSON file containing demand information)
 * @param inst Reference to an Instance object that will be populated with the loaded data
 * @param result_builder Reference to a ResultBuilder object for logging validation results
 *
 * @return true if both files were successfully loaded and validated, false otherwise
 *
 */
inline bool loadAndCheckTeInstance(const std::string& net_file, const std::string& tm_file, Instance& inst, ResultBuilder& result_builder) noexcept {
  LOG_F(INFO, INFO_MSG_6, net_file.c_str());

  nt::graphs::GraphProperties< Digraph > network_props;
  network_props.arcMap(WEIGHT_ATTR, inst.metrics);
  network_props.arcMap(CAPACITY_ATTR, inst.capacities);
  network_props.arcMap(DELAY_ATTR, inst.delays);
  network_props.nodeMap(ROUTER_ATTR, inst.names);

  if (!inst.reader.template readFile< false >(net_file.c_str(), inst.network, &network_props)) {
    LOG_F(ERROR, ERR_MSG_0);
    return false;
  }

  if (!nt::graphs::isBidirected(inst.network)) {
    LOG_F(ERROR, ERR_MSG_1);
    return false;
  }

  if (!inst.network.nodeNum() || !inst.network.arcNum()) {
    LOG_F(ERROR, ERR_MSG_2);
    return false;
  }

  LOG_F(INFO, INFO_MSG_0, net_file.c_str(), inst.network.nodeNum(), inst.network.arcNum());

  const Instance::NetworkxIo::IdNodeMap& id_node_map = inst.reader.idNodeMap();
  for (const auto& pair: id_node_map)
    inst.node_hashes[pair.second] = pair.first;

  // Load demand graphs from traffic matrix
  LOG_F(INFO, INFO_MSG_7, tm_file.c_str());

  nt::JSONDocument doc;
  if (!parseJsonFile(tm_file.c_str(), doc)) { return false; }

  if (!getIntMember(doc, NUM_TIMESTEPS_ATTR, inst.i_num_time_slots)) { return false; }

  if (inst.i_num_time_slots < 1) {
    LOG_F(ERROR, ERR_MSG_8, tm_file.c_str());
    return false;
  }

  if (!hasArrayMember(doc, DEMANDS_ATTR)) { return false; }
  const auto& demands_array = doc[DEMANDS_ATTR].GetArray();

  nt::DoubleDynamicArray values(inst.i_num_time_slots);
  inst.dvms.ensureAndFillEmplace(inst.i_num_time_slots, inst.demand_graph);

  int i_count = 0;
  for (auto& demand_object: demands_array) {
    int i_source = -1, i_target = -1;
    values.removeAll();

    for (nt::JsonObject::ConstMemberIterator iter = demand_object.MemberBegin(); iter != demand_object.MemberEnd(); ++iter) {
      if (iter->name.GetString()[0] == 'v' && iter->value.IsArray()) {
        for (const auto& value: iter->value.GetArray()) {
          const double v = value.GetDouble();
          values.add(v);
        }
      } else if (iter->name.GetString()[0] == 's' && iter->value.IsInt()) {
        i_source = iter->value.GetInt();
      } else if (iter->name.GetString()[0] == 't' && iter->value.IsInt()) {
        i_target = iter->value.GetInt();
      }
    }

    if (values.empty() || i_source < 0 || i_target < 0) {
      LOG_F(ERROR, ERR_MSG_6, tm_file.c_str());
      return false;
    }

    if (values.size() != static_cast< size_t >(inst.i_num_time_slots)) {
      LOG_F(ERROR, ERR_MSG_7, tm_file.c_str());
      return false;
    }

    const Node* p_source = inst.reader.nodeFromId(i_source);
    if (!p_source) {
      LOG_F(ERROR, ERR_MSG_21, i_source, i_target);
      return false;
    }

    const Node* p_target = inst.reader.nodeFromId(i_target);
    if (!p_target) {
      LOG_F(ERROR, ERR_MSG_22, i_source, i_target);
      return false;
    }

    const DemandArc demand_arc = inst.demand_graph.addArc(*p_source, *p_target);

    for (int t = 0; t < inst.i_num_time_slots; ++t) {
      DemandValueMap& dvm = inst.dvms[t];
      dvm[demand_arc] = values[t];
    }

    ++i_count;
  }

  if (!inst.demand_graph.nodeNum() || !inst.demand_graph.arcNum()) {
    LOG_F(ERROR, ERR_MSG_3);
    return false;
  }

  LOG_F(INFO, INFO_MSG_1, tm_file.c_str(), i_count);

  // Collect infos for output
  result_builder._net_file = net_file;
  result_builder._tm_file = tm_file;
  result_builder._i_num_interventions = 0;

  return true;
}

/**
 * @brief Loads and validates a Traffic Engineering (TE) instance from network and traffic matrix files without accumulating results.
 *
 * This function reads a network topology from a network file and traffic demands from a traffic matrix file,
 * validates the data, and populates the provided Instance object.
 *
 * @param net_file Path to the network topology file (expected to be in a format compatible with NetworkxIo)
 * @param tm_file Path to the traffic matrix file (expected to be a JSON file containing demand information)
 * @param inst Reference to an Instance object that will be populated with the loaded data
 * @param result_builder Reference to a ResultBuilder object for logging validation results
 *
 * @return true if both files were successfully loaded and validated, false otherwise
 *
 */
inline bool loadAndCheckTeInstance(const std::string& net_file, const std::string& tm_file, Instance& inst) noexcept {
  ResultBuilder dummy_result_builder(inst);
  return loadAndCheckTeInstance(net_file, tm_file, inst, dummy_result_builder);
}

/**
 * @brief Loads and validates scenario configuration from a JSON file.
 *
 * This function reads a scenario file containing intervention schedules and budget constraints
 * for the network optimization problem. It validates the JSON structure and ensures all
 * constraints are properly defined for each time step.
 *
 * @param scenario_file Path to the scenario JSON file
 * @param inst Reference to the Instance containing network topology and time slot information
 * @param scenario Reference to the Scenario object to populate with loaded data
 * @param result_builder Reference to a ResultBuilder object for logging validation results
 * 
 * @return true if scenario was successfully loaded and validated, false otherwise
 *
 */
inline bool loadAndCheckScenario(const std::string& scenario_file, const Instance& inst, Scenario& scenario, ResultBuilder& result_builder) noexcept {
  LOG_F(INFO, INFO_MSG_8, scenario_file.c_str());

  nt::JSONDocument doc;
  if (!parseJsonFile(scenario_file.c_str(), doc)) { return false; }

  if (!getIntMember(doc, MAX_SEGMENTS_ATTR, scenario.i_max_segments)) { return false; }

  if (scenario.i_max_segments < 0) {
    LOG_F(ERROR, ERR_MSG_9, scenario_file.c_str());
    return false;
  }

  if (!hasArrayMember(doc, BUDGET_ATTR)) { return false; }

  {
    const nt::JsonArray& budgets = doc[BUDGET_ATTR].GetArray();
    scenario.budget.ensureAndFillBytes(inst.i_num_time_slots, 0);
    for (auto& budget: budgets) {
      if (!budget.IsObject()) {
        LOG_F(ERROR, ERR_MSG_5);
        return false;
      }

      const auto& budget_data = budget.GetObject();

      int t, v;
      if (!getIntMember(budget_data, TIMESTEP_ATTR, t)) { return false; }

      if (!validateRange(t, 1, inst.i_num_time_slots, "Budget timeslot")) { return false; }

      if (!getIntMember(budget_data, VALUE_ATTR, v)) { return false; }

      if (v < 0) {
        LOG_F(ERROR, ERR_MSG_10, scenario_file.c_str());
        return false;
      }

      scenario.budget[t] = v;
    }
  }

  if (!hasArrayMember(doc, INTERVENTIONS_ATTR)) { return false; }

  {
    const nt::JsonArray& interventions = doc[INTERVENTIONS_ATTR].GetArray();
    scenario.interventions.ensureAndFill(inst.i_num_time_slots);
    for (auto& intervention: interventions) {
      if (!intervention.IsObject()) {
        LOG_F(ERROR, ERR_MSG_11, scenario_file.c_str());
        return false;
      }

      const auto& intervention_data = intervention.GetObject();

      int t;
      if (!getIntMember(intervention_data, TIMESTEP_ATTR, t)) { return false; }

      if (!validateRange(t, 1, inst.i_num_time_slots, "Intervention timeslot")) { return false; }

      if (!hasArrayMember(intervention_data, LINKS_ATTR)) { return false; }

      {
        const nt::JsonArray& links = intervention_data[LINKS_ATTR].GetArray();
        nt::IntDynamicArray& ids = scenario.interventions[t];
        ids.ensure(links.Size());
        for (auto& link: links)
          ids.add(link.GetInt());
      }
    }
  }

  LOG_F(INFO, INFO_MSG_2, scenario_file.c_str(), scenario.budget.size(), scenario.interventions.size());

  // Collect infos for output
  result_builder._scenario_file = scenario_file;
  result_builder._i_max_segments = scenario.i_max_segments;
  for (int t = 0; t < inst.i_num_time_slots; ++t)
    result_builder._i_num_interventions += (scenario.interventions[t].size() > 0);

  return true;
}

/**
 * @brief Loads and validates scenario configuration from a JSON file without accumulating results.
 *
 * This function reads a scenario file containing intervention schedules and budget constraints
 * for the network optimization problem. It validates the JSON structure and ensures all
 * constraints are properly defined for each time step.
 *
 * @param scenario_file Path to the scenario JSON file
 * @param inst Reference to the Instance containing network topology and time slot information
 * @param scenario Reference to the Scenario object to populate with loaded data
 * 
 * @return true if scenario was successfully loaded and validated, false otherwise
 *
 */
inline bool loadAndCheckScenario(const std::string& scenario_file, const Instance& inst, Scenario& scenario) noexcept {
  ResultBuilder dummy_result_builder(inst);
  return loadAndCheckScenario(scenario_file, inst, scenario, dummy_result_builder);
}

/**
 * @brief Loads and validates SR (Segment Routing) paths from a JSON file.
 *
 * This function reads a JSON file containing SR paths, validates its structure,
 * and populates the routing scheme with the loaded paths. Each SR path is
 * associated with a demand arc and time period.
 *
 * @param srpaths_file Path to the JSON file containing SR paths
 * @param scenario Reference to the scenario object (currently unused in function body)
 * @param inst Reference to the instance containing network and demand graph information
 * @param rs Reference to the routing scheme to be populated with SR paths
 * @param result_builder Reference to a ResultBuilder object for logging validation results
 *
 * @return true if the SR paths were successfully loaded and validated, false otherwise
 *
 * @note Check the subject of the challenge for the expected format of the SR paths JSON file
 * https://gitlab.com/Orange-OpenSource/network-optimization-tools/challenge-roadef-2026/-/blob/main/doc/Challenge_Orange_ROADEF_2026_Subject.pdf
 *
 */
inline bool loadAndCheckSrPaths(const std::string& srpaths_file, const Scenario& scenario, const Instance& inst, RoutingScheme& rs, ResultBuilder& result_builder) noexcept {
  LOG_F(INFO, INFO_MSG_5, srpaths_file.c_str());

  nt::JSONDocument doc;
  if (!parseJsonFile(srpaths_file.c_str(), doc)) { return false; }

  if (!hasArrayMember(doc, SRPATHS_ATTR)) { return false; }

  nt::IntDynamicArray waypoints;
  int                 i_num_srpaths = 0;
  int                 i_total_segments = 0;

  for (auto& srpath_object: doc[SRPATHS_ATTR].GetArray()) {
    if (!srpath_object.IsObject()) {
      LOG_F(ERROR, ERR_MSG_12, srpaths_file.c_str());
      return false;
    }

    int i_demand = -1, i_time = -1;
    waypoints.removeAll();

    for (nt::JsonObject::ConstMemberIterator iter = srpath_object.MemberBegin(); iter != srpath_object.MemberEnd(); ++iter) {
      if (iter->name.GetString()[0] == 'd' && iter->value.IsInt()) {
        i_demand = iter->value.GetInt();
      } else if (iter->name.GetString()[0] == 't' && iter->value.IsInt()) {
        i_time = iter->value.GetInt();
        if (!validateRange(i_time, 0, inst.i_num_time_slots, "Time period")) return false;
      } else if (iter->name.GetString()[0] == 'w' && iter->value.IsArray()) {
        const nt::JsonConstArray& waypoints_array = iter->value.GetArray();
        waypoints.ensure(waypoints_array.Size());
        for (auto& waypoint: waypoints_array) {
          const int id = waypoint.GetInt();
          waypoints.add(id);
        }
      } else {
        LOG_F(ERROR, ERR_MSG_13, srpaths_file.c_str());
        return false;
      }
    }

    if (i_demand < 0 || i_time < 0) {
      LOG_F(ERROR, ERR_MSG_13, srpaths_file.c_str());
      return false;
    }

    const DemandArc demand_arc = inst.demand_graph.arcFromId(i_demand);
    if (!validateId(inst.demand_graph, demand_arc, "Demand arc ID")) return false;

    SrPathBit& srpath = rs.getSrPath(i_time, demand_arc);
    srpath.init(inst.network, waypoints.size());

    Node last = inst.demand_graph.source(demand_arc);
    srpath.addSegment(last);
    for (const int i_node: waypoints) {
      const Node* p_wp = inst.reader.nodeFromId(i_node);
      if (!p_wp) {
        LOG_F(ERROR, ERR_MSG_29, i_node);
        return false;
      }
      srpath.addSegment(*p_wp, last);
      last = *p_wp;
    }
    srpath.finalize(inst.demand_graph.target(demand_arc));

    const int i_num_segments = srpath.segmentNum() - 1;
    if (i_num_segments > scenario.i_max_segments) {
      LOG_F(ERROR, ERR_MSG_14, srpaths_file.c_str(), srpath.segmentNum(), scenario.i_max_segments);
      return false;
    }

    i_total_segments += i_num_segments;      // Accumulate total number of segments across all SR paths
    i_num_srpaths += (i_num_segments > 1);   // Count only srpaths with more than one segment
  }

  LOG_F(INFO, INFO_MSG_3, srpaths_file.c_str(), i_num_srpaths);

  // Collect infos for output
  result_builder._srpaths_file = srpaths_file;
  result_builder._i_total_srpaths = i_num_srpaths;
  result_builder._i_total_segments = i_total_segments;

  return true;
}

/**
 * @brief Loads and validates SR (Segment Routing) paths from a JSON file without accumulating results.
 *
 * This function reads a JSON file containing SR paths, validates its structure,
 * and populates the routing scheme with the loaded paths. Each SR path is
 * associated with a demand arc and time period.
 *
 * @param srpaths_file Path to the JSON file containing SR paths
 * @param scenario Reference to the scenario object (currently unused in function body)
 * @param inst Reference to the instance containing network and demand graph information
 * @param rs Reference to the routing scheme to be populated with SR paths
 *
 * @return true if the SR paths were successfully loaded and validated, false otherwise
 *
 * @note Check the subject of the challenge for the expected format of the SR paths JSON file
 * https://gitlab.com/Orange-OpenSource/network-optimization-tools/challenge-roadef-2026/-/blob/main/doc/Challenge_Orange_ROADEF_2026_Subject.pdf
 * 
 * @note This overload is provided for cases where the caller does not need to collect detailed results and only wants to perform the simulation for validation purposes.
 *
 */
inline bool loadAndCheckSrPaths(const std::string& srpaths_file, const Scenario& scenario, const Instance& inst, RoutingScheme& rs) noexcept {
  ResultBuilder dummy_result_builder(inst);
  return loadAndCheckSrPaths(srpaths_file, scenario, inst, rs, dummy_result_builder);
}

// ============================================================================
// Segment Routing Simulation
// ============================================================================

/**
 * @brief Performs segment routing simulation for all time steps and computes several objective values (MLUs, inverse load costs).
 *
 * @param inst The instance containing network topology and demands
 * @param scenario The scenario containing interventions
 * @param rs The routing scheme with SR paths
 * @param result_builder Builder to accumulate MLU results
 *
 * @return true if the simulation completed successfully, false if any error occurred during routing (e.g., invalid SR paths, unsatisfied demands)
 *
 */
inline bool simulateSegmentRouting(Instance& inst, const Scenario& scenario, const RoutingScheme& rs, ResultBuilder& result_builder) noexcept {
  nt::DoubleDynamicArray sat(inst.network.arcNum());

  for (int t = 0; t < inst.i_num_time_slots; ++t) {
    // Apply interventions to save the original metrics of the network
    InterventionGuard intervention_guard(inst.network, inst.metrics, scenario.interventions[t]);

    SegmentRouting sr(inst.network, inst.metrics);

    const int i_num_routed = sr.run(inst.demand_graph, inst.dvms[t], rs.getSrPathsAt(t));

    if (i_num_routed != inst.demand_graph.arcNum()) {
      LOG_F(ERROR, ERR_MSG_56, i_num_routed, inst.demand_graph.arcNum(), t);
      return false;
    }

    sat.removeAll();
    double f_mlu = 0.;
    double f_sum_inv_load_cost = 0.;
    for (ArcIt arc(inst.network); arc != nt::INVALID; ++arc) {
      const double f_sat = sr.saturation(arc, inst.capacities[arc]);
      f_mlu = nt::max(f_mlu, f_sat);
      f_sum_inv_load_cost += nt::te::invArcLoadCost(f_sat);
      sat.add(f_sat);
      result_builder.addSaturationValue(t, arc, f_sat);
    }

    result_builder.addObjectiveValues(t, f_mlu, f_sum_inv_load_cost, nt::te::jainIndex(sat));

    LOG_F(INFO, INFO_MSG_4, t, f_mlu, f_sum_inv_load_cost);

    // Metrics are automatically restored to their original values by InterventionGuard destructor
  }

  return true;
}

/**
 * @brief Simulates segment routing without accumulating results.
 *
 * @param inst The instance containing network topology and demands
 * @param scenario The scenario containing interventions
 * @param rs The routing scheme with SR paths
 *
 * @return true if the simulation completed successfully, false if any error occurred during routing (e.g., invalid SR paths, unsatisfied demands)
 *
 * @note This overload is provided for cases where the caller does not need to collect detailed results and only wants to perform the simulation for validation purposes.
 */
inline bool simulateSegmentRouting(Instance& inst, const Scenario& scenario, const RoutingScheme& rs) noexcept {
  ResultBuilder dummy_result_builder(inst);
  return simulateSegmentRouting(inst, scenario, rs, dummy_result_builder);
}

#endif