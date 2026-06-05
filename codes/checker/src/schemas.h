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
 * @file schemas.h
 * @brief JSON schemas for the EURO/ROADEF Challenge 2026 checker.
 *
 * @author Morgan Chopin (morgan.chopin@orange.com)
 */

#ifndef _CHECKER_SCHEMAS_H_
#define _CHECKER_SCHEMAS_H_

inline const char* NETWORK_SCHEMA = R"json({
  "$schema": "http://json-schema.org/draft-07/schema#",
  "title": "NetworkX Network Topology Schema",
  "description": "JSON schema for validating NetworkX-format network topology files used in ROADEF Challenge 2026",
  "type": "object",
  "properties": {
    "directed": {
      "type": "boolean",
      "description": "Indicates whether the graph is directed"
    },
    "multigraph": {
      "type": "boolean",
      "description": "Indicates whether multiple arcs between nodes are allowed"
    },
    "nodes": {
      "type": "array",
      "description": "Array of node objects",
      "items": {
        "type": "object",
        "properties": {
          "id": {
            "type": "integer",
            "description": "Unique identifier of the node"
          },
          "name": {
            "type": "string",
            "description": "Name of the node (label, optional)"
          }
        },
        "required": ["id"],
        "additionalProperties": true
      },
      "minItems": 1
    },
    "links": {
      "type": "array",
      "description": "Array of link objects",
      "items": {
        "type": "object",
        "properties": {
          "id": {
            "type": "integer",
            "description": "Unique identifier of the link"
          },
          "from": {
            "type": "integer",
            "description": "ID of the source node"
          },
          "to": {
            "type": "integer",
            "description": "ID of the destination node"
          },
          "metric": {
            "type": "number",
            "description": "Metric value (IGP weight)"
          },
          "capacity": {
            "type": "number",
            "description": "Maximum capacity of the link"
          }
        },
        "required": ["id", "from", "to", "metric", "capacity"],
        "additionalProperties": true
      },
      "minItems": 1
    }
  },
  "required": ["directed", "nodes", "links"],
  "additionalProperties": true
})json";

inline const char* TRAFFIC_MATRIX_SCHEMA = R"json({
  "$schema": "http://json-schema.org/draft-07/schema#",
  "title": "Traffic Matrix Schema",
  "description": "JSON schema for validating traffic matrix files used in ROADEF Challenge 2026",
  "type": "object",
  "properties": {
    "num_time_slots": {
      "type": "integer",
      "minimum": 1,
      "description": "Number of time slots over which demand is defined"
    },
    "demands": {
      "type": "array",
      "description": "Array of demand objects",
      "items": {
        "type": "object",
        "properties": {
          "v": {
            "type": "array",
            "description": "Traffic values for each time slot",
            "items": {
              "type": "number",
              "minimum": 0
            },
            "minItems": 1
          },
          "s": {
            "type": "integer",
            "minimum": 0,
            "description": "ID of the source node"
          },
          "t": {
            "type": "integer",
            "minimum": 0,
            "description": "ID of the destination node"
          }
        },
        "required": ["v", "s", "t"],
        "additionalProperties": true
      },
      "minItems": 1
    }
  },
  "required": ["num_time_slots", "demands"],
  "additionalProperties": true
})json";

inline const char* SCENARIO_SCHEMA = R"json({
  "$schema": "http://json-schema.org/draft-07/schema#",
  "title": "Intervention Scenario Schema",
  "description": "JSON schema for validating intervention scenario files used in ROADEF Challenge 2026",
  "type": "object",
  "properties": {
    "max_segments": {
      "type": "integer",
      "minimum": 0,
      "description": "Maximum number of segments allowed in any SR path"
    },
    "budget": {
      "type": "array",
      "description": "Budget constraints per time step",
      "items": {
        "type": "object",
        "properties": {
          "t": {
            "type": "integer",
            "minimum": 1,
            "description": "Time step index (starts at 1, timestep 0 has infinite budget)"
          },
          "value": {
            "type": "integer",
            "minimum": 0,
            "description": "Maximum number of routing changes allowed at this time step"
          }
        },
        "required": ["t", "value"],
        "additionalProperties": false
      },
      "minItems": 0
    },
    "interventions": {
      "type": "array",
      "description": "List of network interventions (link failures)",
      "items": {
        "type": "object",
        "properties": {
          "t": {
            "type": "integer",
            "minimum": 1,
            "description": "Time step when intervention occurs (no interventions at t=0)"
          },
          "links": {
            "type": "array",
            "description": "Array of link IDs that are offline at this time step",
            "items": {
              "type": "integer",
              "minimum": 0
            },
            "minItems": 0
          }
        },
        "required": ["t", "links"],
        "additionalProperties": false
      },
      "minItems": 0
    }
  },
  "required": ["max_segments", "budget", "interventions"],
  "additionalProperties": true
})json";

inline const char* SRPATHS_SCHEMA = R"json({
  "$schema": "http://json-schema.org/draft-07/schema#",
  "title": "Segment Routing Paths Schema",
  "description": "JSON schema for validating SR paths solution files used in ROADEF Challenge 2026",
  "type": "object",
  "properties": {
    "srpaths": {
      "type": "array",
      "description": "Array of segment routing path objects",
      "items": {
        "type": "object",
        "properties": {
          "d": {
            "type": "integer",
            "minimum": 0,
            "description": "Demand ID (must exist in traffic matrix)"
          },
          "t": {
            "type": "integer",
            "minimum": 0,
            "description": "Time slot index"
          },
          "w": {
            "type": "array",
            "description": "Waypoint node IDs (path: source → w[0] → ... → w[n-1] → target)",
            "items": {
              "type": "integer",
              "minimum": 0
            },
            "minItems": 0
          }
        },
        "required": ["d", "t", "w"],
        "additionalProperties": false
      },
      "minItems": 0
    }
  },
  "required": ["srpaths"],
  "additionalProperties": true
})json";

#endif
