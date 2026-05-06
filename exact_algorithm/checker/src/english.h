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
 * @file english.h
 * @brief English language header file for the EURO/ROADEF Challenge 2026 checker.
 *
 * @author Morgan Chopin (morgan.chopin@orange.com)
 */

#ifndef _CHECKER_ENGLISH_H_
#define _CHECKER_ENGLISH_H_

// Icons for status messages
#define ICON_CROSS "❌"
#define ICON_CHECK "✅"

// Help messages
#define HELP_MSG_0 "JSON file describing the network topology."
#define HELP_MSG_1 "JSON file describing the traffic matrix."
#define HELP_MSG_2 "JSON file describing the interventions scenario."
#define HELP_MSG_3 "JSON file describing the SR-paths associated to the demands."
#define HELP_MSG_4 "Activate output logs"
#define HELP_MSG_5 "Instance name prefix (alternative to specifying individual files). Automatically loads <name>-net.json, <name>-tm.json, <name>-scenario.json and <name>-srpaths.json."
#define HELP_MSG_6 "Enable pretty output with charts"
#define HELP_MSG_7 "Sets the maximum number of decimal places for floating-point values (default: 12)."

// Error messages
#define ERR_MSG_0 ICON_CROSS " Cannot read the input files."
#define ERR_MSG_1 ICON_CROSS " Input network is NOT bidirected."
#define ERR_MSG_2 ICON_CROSS " Input network is empty or has no arcs."
#define ERR_MSG_3 ICON_CROSS " Demand graph is empty or has no demands."
#define ERR_MSG_4 ICON_CROSS " Budget constraint is violated (cost is at least {} > budget : {}) at time step {}."
#define ERR_MSG_5 ICON_CROSS " An budget in the budgets list is not a valid object."
#define ERR_MSG_6 ICON_CROSS " {} : Bad demand format"
#define ERR_MSG_7 ICON_CROSS " {} : Demand values array size does not match number of time slots"
#define ERR_MSG_8 ICON_CROSS " {} : 'num_time_slots' must be a positive integer."
#define ERR_MSG_9 ICON_CROSS " {} : 'max_segments' must be a non negative integer."
#define ERR_MSG_10 ICON_CROSS " {} : Budget value must be a non negative integer."
#define ERR_MSG_11 ICON_CROSS " {} : An intervention in the interventions list is not a valid object."
#define ERR_MSG_12 ICON_CROSS " {} : An sr-path in the sr-paths list is not a valid object."
#define ERR_MSG_13 ICON_CROSS " {} : Bad srpath format"
#define ERR_MSG_14 ICON_CROSS " {} : SR path has more segments ({}) than the maximum allowed ({})"
#define ERR_MSG_15 ICON_CROSS " Parse Error: Document is not a valid JSON file. Error code : {} (offset: {})"
#define ERR_MSG_16 ICON_CROSS " Document is invalid, it must be an object"
#define ERR_MSG_17 ICON_CROSS " {} attribute is missing or is not an integer"
#define ERR_MSG_18 ICON_CROSS " {} attribute is missing or is not an array"
#define ERR_MSG_19 ICON_CROSS " {} '{}' is out of bounds [{},{})"
#define ERR_MSG_20 ICON_CROSS " Every demand should be assigned an SR path"
#define ERR_MSG_21 ICON_CROSS " Source node of demand {} -> {} not found in the network"
#define ERR_MSG_22 ICON_CROSS " Target node of demand {} -> {} not found in the network"
#define ERR_MSG_23 ICON_CROSS " Missing required file(s):"
#define ERR_MSG_24 "  --net (network topology file)"
#define ERR_MSG_25 "  --tm (traffic matrix file)"
#define ERR_MSG_26 "  --scenario (scenario file)"
#define ERR_MSG_27 "  --srpaths (SR paths solution file)"
#define ERR_MSG_28 " Please provide all required files or use --instance <name>."
#define ERR_MSG_29 ICON_CROSS " Waypoint {} not found in the network"
#define ERR_MSG_30 ICON_CROSS " Cannot load network schema file '{}'"
#define ERR_MSG_31 ICON_CROSS " Network schema file '{}' is not found or not a valid JSON"
#define ERR_MSG_32 ICON_CROSS " Network JSON validation failed. Invalid schema at: {}"
#define ERR_MSG_33 ICON_CROSS " Network JSON validation failed. Invalid keyword: {}"
#define ERR_MSG_34 ICON_CROSS " Network JSON validation failed. Invalid document at: {}"
#define ERR_MSG_35 ICON_CROSS " Network JSON: Duplicate node ID {} found"
#define ERR_MSG_36 ICON_CROSS " Network JSON: Link #{} references undefined node ID (from={}, to={})"
#define ERR_MSG_37 ICON_CROSS " Network JSON: Duplicate link found (from={}, to={}). Multigraph not allowed."
#define ERR_MSG_38 ICON_CROSS " Traffic matrix schema file '{}' is not found or not a valid JSON"
#define ERR_MSG_39 ICON_CROSS " Traffic matrix JSON validation failed. Invalid schema at: {}"
#define ERR_MSG_40 ICON_CROSS " Traffic matrix JSON validation failed. Invalid keyword: {}"
#define ERR_MSG_41 ICON_CROSS " Traffic matrix JSON validation failed. Invalid document at: {}"
#define ERR_MSG_42 ICON_CROSS " Traffic matrix: Demand #{} has inconsistent value array size (expected {}, got {})"
#define ERR_MSG_43 ICON_CROSS " Traffic matrix: Duplicate demand found (s={}, t={})"
#define ERR_MSG_44 ICON_CROSS " Scenario schema file '{}' is not found or not a valid JSON"
#define ERR_MSG_45 ICON_CROSS " Scenario JSON validation failed. Invalid schema at: {}"
#define ERR_MSG_46 ICON_CROSS " Scenario JSON validation failed. Invalid keyword: {}"
#define ERR_MSG_47 ICON_CROSS " Scenario JSON validation failed. Invalid document at: {}"
#define ERR_MSG_48 ICON_CROSS " Scenario: Duplicate budget entry for timestep {}"
#define ERR_MSG_49 ICON_CROSS " Scenario: Duplicate intervention entry for timestep {}"
#define ERR_MSG_50 ICON_CROSS " Scenario: Budget/Intervention at timestep 0 is not allowed (t must be ≥ 1)"
#define ERR_MSG_51 ICON_CROSS " SR paths schema file '{}' is not found or not a valid JSON"
#define ERR_MSG_52 ICON_CROSS " SR paths JSON validation failed. Invalid schema at: {}"
#define ERR_MSG_53 ICON_CROSS " SR paths JSON validation failed. Invalid keyword: {}"
#define ERR_MSG_54 ICON_CROSS " SR paths JSON validation failed. Invalid document at: {}"
#define ERR_MSG_55 ICON_CROSS " SR paths: Duplicate SR path entry for demand {} at timestep {}"
#define ERR_MSG_56 ICON_CROSS " Only {} out of {} demands were routed at time {}. This may indicate an issue with the provided SR paths or interventions."
#define ERR_MSG_57 ICON_CROSS " The checker is not built with FTXUI. Please set ENABLE_FTXUI to 1 and rebuild."

// Info messages
#define INFO_MSG_0 ICON_CHECK " Network topology '{}' successfully loaded ({} nodes, {} arcs)"
#define INFO_MSG_1 ICON_CHECK " Demand graph '{}' successfully loaded ({} demands)"
#define INFO_MSG_2 ICON_CHECK " Input scenario '{}' loaded successfully (budget size {}, interventions {})"
#define INFO_MSG_3 ICON_CHECK " Input srpaths '{}' loaded successfully ({} srpaths)"

#define INFO_MSG_4 "Maximum Link Utilization (MLU) at {} : {} (inverse load score: {})"
#define INFO_MSG_5 "Loading SR paths from '{}'..."
#define INFO_MSG_6 "Loading network topology from '{}'..."
#define INFO_MSG_7 "Loading traffic matrix from '{}'..."
#define INFO_MSG_8 "Loading scenario from '{}'..."
#define INFO_MSG_9 "Validating network JSON format using schema '{}'..."
#define INFO_MSG_10 ICON_CHECK " Network JSON format validation successful"
#define INFO_MSG_11 "Validating traffic matrix JSON format using schema '{}'..."
#define INFO_MSG_12 ICON_CHECK " Traffic matrix JSON format validation successful"
#define INFO_MSG_13 "Validating scenario JSON format using schema '{}'..."
#define INFO_MSG_14 ICON_CHECK " Scenario JSON format validation successful"
#define INFO_MSG_15 "Validating SR paths JSON format using schema '{}'..."
#define INFO_MSG_16 ICON_CHECK " SR paths JSON format validation successful"

#endif