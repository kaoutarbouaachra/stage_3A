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
 * @file validation.h
 * @brief Validation functions for the EURO/ROADEF Challenge 2026 checker.
 *
 * @author Qiao Zhang (qiao.zhang@orange.com)
 */

#ifndef _CHECKER_VALIDATION_H_
#define _CHECKER_VALIDATION_H_

#include <string>
#include <unordered_set>

#include "core/json.h"
#include "core/logging.h"
#include "helpers.h"
#include "schemas.h"

/**
 * @brief Validate the format of a network JSON file against a schema and perform additional semantic checks.
 *
 * @param net_file The path to the network JSON file to validate.
 * @param schema_file The path to the JSON schema file. Defaults to NETWORK_SCHEMA.
 * @return true If the network JSON file is valid, false otherwise.
 *
 */
inline bool validateNetworkJsonFormat(const std::string& net_file, const std::string& schema_file = "") noexcept {
  // Load and parse the schema
  nt::JSONDocument schema_doc;
  if (schema_file.empty()) {
    LOG_F(INFO, INFO_MSG_9, "embedded network schema");
    if (schema_doc.Parse(NETWORK_SCHEMA).HasParseError()) {
      LOG_F(ERROR, ERR_MSG_31, "embedded network schema");
      return false;
    }
  } else {
    LOG_F(INFO, INFO_MSG_9, schema_file.c_str());
    if (!parseJsonFile(schema_file.c_str(), schema_doc)) {
      LOG_F(ERROR, ERR_MSG_31, schema_file.c_str());
      return false;
    }
  }

  // Create schema validator
  nt::JSONSchemaDocument  schema(schema_doc);
  nt::JSONSchemaValidator validator(schema);

  // Load and parse the network file
  nt::JSONDocument net_doc;
  if (!parseJsonFile(net_file.c_str(), net_doc)) { return false; }

  // Validate against schema
  if (!net_doc.Accept(validator)) {
    rapidjson::StringBuffer sb;
    validator.GetInvalidSchemaPointer().StringifyUriFragment(sb);
    LOG_F(ERROR, ERR_MSG_32, sb.GetString());

    LOG_F(ERROR, ERR_MSG_33, validator.GetInvalidSchemaKeyword());

    sb.Clear();
    validator.GetInvalidDocumentPointer().StringifyUriFragment(sb);
    LOG_F(ERROR, ERR_MSG_34, sb.GetString());
    return false;
  }

  // Additional semantic validation beyond schema

  // 1. Check for duplicate node IDs
  std::unordered_set< int > node_ids;
  const auto&               nodes_array = net_doc["nodes"].GetArray();
  for (const auto& node: nodes_array) {
    const int node_id = node["id"].GetInt();
    if (!node_ids.insert(node_id).second) {
      LOG_F(ERROR, ERR_MSG_35, node_id);
      return false;
    }
  }

  // 2. Validate that all links reference existing node IDs
  const auto& links_array = net_doc["links"].GetArray();
  for (size_t i = 0; i < links_array.Size(); ++i) {
    const auto& link = links_array[i];
    const int   from_id = link["from"].GetInt();
    const int   to_id = link["to"].GetInt();

    if (node_ids.find(from_id) == node_ids.end() || node_ids.find(to_id) == node_ids.end()) {
      LOG_F(ERROR, ERR_MSG_36, i, from_id, to_id);
      return false;
    }
  }

  // 3. Check for duplicate links if multigraph is false
  if (net_doc.HasMember("multigraph") && !net_doc["multigraph"].GetBool()) {
    std::unordered_set< std::string > link_pairs;
    for (const auto& link: links_array) {
      const int from_id = link["from"].GetInt();
      const int to_id = link["to"].GetInt();

      // Create a unique key for the link pair
      std::string link_key = std::to_string(from_id) + "_" + std::to_string(to_id);

      if (!link_pairs.insert(link_key).second) {
        LOG_F(ERROR, ERR_MSG_37, from_id, to_id);
        return false;
      }
    }
  }

  LOG_F(INFO, INFO_MSG_10);
  return true;
}

/**
 * @brief Validate the format of a traffic matrix JSON file against a schema and perform additional semantic checks.
 *
 * @param tm_file The path to the traffic matrix JSON file to validate.
 * @param schema_file The path to the JSON schema file. Defaults to TRAFFIC_MATRIX_SCHEMA.
 * @return true If the traffic matrix JSON file is valid, false otherwise.
 */
inline bool validateTrafficMatrixJsonFormat(const std::string& tm_file, const std::string& schema_file = "") noexcept {
  // Load and parse the schema
  nt::JSONDocument schema_doc;
  if (schema_file.empty()) {
    LOG_F(INFO, INFO_MSG_11, "embedded traffic matrix schema");
    if (schema_doc.Parse(TRAFFIC_MATRIX_SCHEMA).HasParseError()) {
      LOG_F(ERROR, ERR_MSG_38, "embedded traffic matrix schema");
      return false;
    }
  } else {
    LOG_F(INFO, INFO_MSG_11, schema_file.c_str());
    if (!parseJsonFile(schema_file.c_str(), schema_doc)) {
      LOG_F(ERROR, ERR_MSG_38, schema_file.c_str());
      return false;
    }
  }

  // Create schema validator
  rapidjson::SchemaDocument  schema(schema_doc);
  rapidjson::SchemaValidator validator(schema);

  // Load and parse the traffic matrix file
  nt::JSONDocument tm_doc;
  if (!parseJsonFile(tm_file.c_str(), tm_doc)) { return false; }

  // Validate against schema
  if (!tm_doc.Accept(validator)) {
    rapidjson::StringBuffer sb;
    validator.GetInvalidSchemaPointer().StringifyUriFragment(sb);
    LOG_F(ERROR, ERR_MSG_39, sb.GetString());

    LOG_F(ERROR, ERR_MSG_40, validator.GetInvalidSchemaKeyword());

    sb.Clear();
    validator.GetInvalidDocumentPointer().StringifyUriFragment(sb);
    LOG_F(ERROR, ERR_MSG_41, sb.GetString());
    return false;
  }

  // Additional semantic validation beyond schema

  const int   num_time_slots = tm_doc["num_time_slots"].GetInt();
  const auto& demands_array = tm_doc["demands"].GetArray();

  // 1. Check that all value arrays have size equal to num_time_slots
  for (size_t i = 0; i < demands_array.Size(); ++i) {
    const auto& demand = demands_array[i];
    const auto& values_array = demand["v"].GetArray();

    if (static_cast< int >(values_array.Size()) != num_time_slots) {
      LOG_F(ERROR, ERR_MSG_42, i, num_time_slots, values_array.Size());
      return false;
    }
  }

  // 2. Check for duplicate demands (same source and target)
  std::unordered_set< std::string > demand_pairs;
  for (const auto& demand: demands_array) {
    const int source = demand["s"].GetInt();
    const int target = demand["t"].GetInt();

    // Create a unique key for the demand pair
    std::string demand_key = std::to_string(source) + "_" + std::to_string(target);

    if (!demand_pairs.insert(demand_key).second) {
      LOG_F(ERROR, ERR_MSG_43, source, target);
      return false;
    }
  }

  LOG_F(INFO, INFO_MSG_12);
  return true;
}

/**
 * @brief Validate the format of a scenario JSON file against a schema and perform additional semantic checks.
 *
 * @param scenario_file The path to the scenario JSON file to validate.
 * @param schema_file The path to the JSON schema file. Defaults to SCENARIO_SCHEMA.
 * @return true If the scenario JSON file is valid, false otherwise.
 */
inline bool validateScenarioJsonFormat(const std::string& scenario_file, const std::string& schema_file = "") noexcept {
  // Load and parse the schema
  nt::JSONDocument schema_doc;
  if (schema_file.empty()) {
    LOG_F(INFO, INFO_MSG_13, "embedded scenario schema");
    if (schema_doc.Parse(SCENARIO_SCHEMA).HasParseError()) {
      LOG_F(ERROR, ERR_MSG_44, "embedded scenario schema");
      return false;
    }
  } else {
    LOG_F(INFO, INFO_MSG_13, schema_file.c_str());
    if (!parseJsonFile(schema_file.c_str(), schema_doc)) {
      LOG_F(ERROR, ERR_MSG_44, schema_file.c_str());
      return false;
    }
  }

  // Create schema validator
  rapidjson::SchemaDocument  schema(schema_doc);
  rapidjson::SchemaValidator validator(schema);

  // Load and parse the scenario file
  nt::JSONDocument scenario_doc;
  if (!parseJsonFile(scenario_file.c_str(), scenario_doc)) { return false; }

  // Validate against schema
  if (!scenario_doc.Accept(validator)) {
    rapidjson::StringBuffer sb;
    validator.GetInvalidSchemaPointer().StringifyUriFragment(sb);
    LOG_F(ERROR, ERR_MSG_45, sb.GetString());

    LOG_F(ERROR, ERR_MSG_46, validator.GetInvalidSchemaKeyword());

    sb.Clear();
    validator.GetInvalidDocumentPointer().StringifyUriFragment(sb);
    LOG_F(ERROR, ERR_MSG_47, sb.GetString());
    return false;
  }

  // Additional semantic validation beyond schema

  // 1. Check budget entries: no duplicates, no timestep 0
  if (scenario_doc.HasMember("budget")) {
    const auto&               budget_array = scenario_doc["budget"].GetArray();
    std::unordered_set< int > budget_timesteps;

    for (const auto& budget_entry: budget_array) {
      const int timestep = budget_entry["t"].GetInt();

      // Check for timestep 0 (not allowed)
      if (timestep == 0) {
        LOG_F(ERROR, ERR_MSG_50);
        return false;
      }

      // Check for duplicate timestep in budget
      if (!budget_timesteps.insert(timestep).second) {
        LOG_F(ERROR, ERR_MSG_48, timestep);
        return false;
      }
    }
  }

  // 2. Check intervention entries: no duplicates, no timestep 0
  if (scenario_doc.HasMember("interventions")) {
    const auto&               interventions_array = scenario_doc["interventions"].GetArray();
    std::unordered_set< int > intervention_timesteps;

    for (const auto& intervention_entry: interventions_array) {
      const int timestep = intervention_entry["t"].GetInt();

      // Check for timestep 0 (not allowed)
      if (timestep == 0) {
        LOG_F(ERROR, ERR_MSG_50);
        return false;
      }

      // Check for duplicate timestep in interventions
      if (!intervention_timesteps.insert(timestep).second) {
        LOG_F(ERROR, ERR_MSG_49, timestep);
        return false;
      }
    }
  }

  LOG_F(INFO, INFO_MSG_14);
  return true;
}

/**
 * @brief Validate the format of an SR paths JSON file against a schema and perform additional semantic checks.
 *
 * @param srpaths_file The path to the SR paths JSON file to validate.
 * @param schema_file The path to the JSON schema file. Defaults to SRPATHS_SCHEMA.
 * @return true If the SR paths JSON file is valid, false otherwise.
 */
inline bool validateSrPathsJsonFormat(const std::string& srpaths_file, const std::string& schema_file = "") noexcept {
  // Load and parse the schema
  nt::JSONDocument schema_doc;
  if (schema_file.empty()) {
    LOG_F(INFO, INFO_MSG_15, "embedded SR paths schema");
    if (schema_doc.Parse(SRPATHS_SCHEMA).HasParseError()) {
      LOG_F(ERROR, ERR_MSG_51, "embedded SR paths schema");
      return false;
    }
  } else {
    LOG_F(INFO, INFO_MSG_15, schema_file.c_str());
    if (!parseJsonFile(schema_file.c_str(), schema_doc)) {
      LOG_F(ERROR, ERR_MSG_51, schema_file.c_str());
      return false;
    }
  }

  // Create schema validator
  rapidjson::SchemaDocument  schema(schema_doc);
  rapidjson::SchemaValidator validator(schema);

  // Load and parse the SR paths file
  nt::JSONDocument srpaths_doc;
  if (!parseJsonFile(srpaths_file.c_str(), srpaths_doc)) { return false; }

  // Validate against schema
  if (!srpaths_doc.Accept(validator)) {
    rapidjson::StringBuffer sb;
    validator.GetInvalidSchemaPointer().StringifyUriFragment(sb);
    LOG_F(ERROR, ERR_MSG_52, sb.GetString());

    LOG_F(ERROR, ERR_MSG_53, validator.GetInvalidSchemaKeyword());

    sb.Clear();
    validator.GetInvalidDocumentPointer().StringifyUriFragment(sb);
    LOG_F(ERROR, ERR_MSG_54, sb.GetString());
    return false;
  }

  // Additional semantic validation beyond schema

  // Check for duplicate (demand, timestep) pairs
  if (srpaths_doc.HasMember("srpaths")) {
    const auto&                       srpaths_array = srpaths_doc["srpaths"].GetArray();
    std::unordered_set< std::string > demand_time_pairs;

    for (const auto& srpath_entry: srpaths_array) {
      const int demand_id = srpath_entry["d"].GetInt();
      const int timestep = srpath_entry["t"].GetInt();

      // Create a unique key for (demand, timestep) pair
      std::string dt_key = std::to_string(demand_id) + "_" + std::to_string(timestep);

      // Check for duplicate (demand, timestep) pair
      if (!demand_time_pairs.insert(dt_key).second) {
        LOG_F(ERROR, ERR_MSG_55, demand_id, timestep);
        return false;
      }
    }
  }

  LOG_F(INFO, INFO_MSG_16);
  return true;
}

#endif
