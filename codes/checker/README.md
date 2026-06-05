# EURO/ROADEF Challenge 2026 - Checker

**Official validation tool for the EURO/ROADEF Challenge 2026 - "Keep the Flow!"**

## Table of Contents

- [Installation](#installation)
- [Building from Source](#building-from-source)
- [Usage](#usage)
- [Output Format](#output-format)
- [License](#license)

## Installation

### Pre-compiled Binary (Recommended)

For Linux x86-64 systems, pre-compiled binaries are available on the [releases page](https://gitlab.com/Orange-OpenSource/network-optimization-tools/challenge-roadef-2026/-/releases).

**To install:**

1. Download the latest `checker-v<version>-x86-64_linux` binary from the releases page
2. Make it executable:
   ```bash
   chmod +x checker-v<version>-x86-64_linux
   ```
3. Optionally, create a symlink or move it to your PATH:
   ```bash
   ln -s checker-v<version>-x86-64_linux checker
   ```

The pre-compiled binary is statically linked and has no external dependencies.

### Building from Source

If you prefer to build from source or need a different platform, see the section below.

## Building from Source

### Prerequisites

Before building the checker, ensure you have the following installed:

1. **C++20 Compiler**
   - GCC 11.0+ or Clang 14.0+ (Linux/macOS)

2. **Networktools Library**
   - The checker depends on the [networktools](https://gitlab.com/Orange-OpenSource/network-optimization-tools/networktools) library
   - To install the library, simply clone the repository into your home directory (or any other folder): `git clone https://gitlab.com/Orange-OpenSource/network-optimization-tools/networktools.git`

### Build Instructions

1. **Navigate to the src directory:**
   ```bash
   cd checker/src
   ```

2. **Build the checker:**
   ```bash
   # Basic build (English language, no FTXUI)
   make
   
   # If networktools is in a custom location (i.e not in your home directory)
   make IDIRNT=/path/to/networktools
   
   # Enable FTXUI support (for pretty terminal output)
   make ENABLE_FTXUI=1
   
   # Build with French language
   make CHECKER_LANG=FR
   
   # Combine options
   make ENABLE_FTXUI=1 CHECKER_LANG=EN IDIRNT=/custom/path/networktools
   ```

   **Build Options:**
   - `IDIRNT=/path` - Path to networktools src/networktools directory (default: `~/networktools`)
   - `ENABLE_FTXUI=1` - Enable FTXUI terminal UI support (default: `0`, disabled)
   - `CHECKER_LANG=XX` - Set language (`EN` or `FR`, default: `EN`)

3. **Create a symlink (optional):**
   ```bash
   make checker
   ```
   This creates a `checker` symlink to the versioned binary.

### Build Output

After successful compilation, you'll have:
- `checker-v<version>-x86-64_linux` - A statically-linked checker binary
- `checker` - Symlink to the binary (if created)

### Cleaning Build Artifacts

```bash
# Remove compiled binaries only
make clean

# Remove all build artifacts including FTXUI
make clean-all
```

## Usage

### Command Syntax

#### Option 1: Specify Individual Files

```bash
./checker --net <network_file> --tm <traffic_matrix> \
          --scenario <scenario_file> --srpaths <solution_file> \
          [--verbose] [--pretty] [--max-decimal-places <n>]
```

#### Option 2: Use Instance Prefix

```bash
./checker --instance <instance_name> \
          [--verbose] [--pretty] [--max-decimal-places <n>]
```

### Required Arguments

**When using individual files:**

| Flag | Description | Format |
|------|-------------|--------|
| `--net` | Network topology file | JSON |
| `--tm` | Traffic matrix file | JSON  |
| `--scenario` | Scenario configuration | JSON |
| `--srpaths` | SR paths solution | JSON  |

**When using instance prefix:**

| Flag | Description |
|------|-------------|
| `--instance` | Instance name prefix. Automatically loads `<name>-net.json`, `<name>-tm.json`, `<name>-scenario.json`, and `<name>-srpaths.json` |

Note that if `--instance` is specified, all the options `--net`, `--tm`, `--scenario` and `--srpaths` are ignored.

### Optional Arguments

| Flag | Description |
|------|-------------|
| `--verbose` | Enable detailed logging to console and `checker.log` |
| `--pretty` | Enable formatted output display |
| `--max-decimal-places` | Maximum decimal places for floating-point output (default: 12) |
| `--help` | Display help messages |

Note that `--verbose` and `--pretty` are exclusive options. If both options are specified, then `--pretty` is ignored.

### Examples

#### Example 1: Using Individual Files

```bash
./checker \
  --net inputs/simple-net.json \
  --tm inputs/simple-tm.json \
  --scenario inputs/simple-scenario.json \
  --srpaths inputs/simple-srpaths.json
```

#### Example 2: Using Instance Prefix

```bash
# Automatically loads inputs/simple/simple-net.json, inputs/simple/simple-tm.json, 
# inputs/simple/simple-scenario.json, and inputs/simple/simple-srpaths.json
./checker --instance inputs/simple/simple
```

#### Example 3: With Verbose Output

```bash
./checker --instance inputs/simple/simple --verbose
```

## Output Format

The checker outputs results in JSON format to standard output. The structure varies depending on whether the solution is valid or invalid.

### Valid Solution Output

When the solution passes all validation checks, the JSON output contains the following fields:


| Field | Type | Description |
|-------|------|-------------|
| `valid` | boolean | Indicates whether the solution is valid (`true`) or invalid (`false`) |
| `total_cost` | integer | Total cost of routing changes across all time slots |
| `total_srpaths` | integer | Total number of SR paths defined in the solution |
| `total_segments` | integer | Total number of segments across all SR paths |
| `objectives` | array | Array of objective values for each time slot |
| `objectives[].t` | integer | Time slot index |
| `objectives[].mlu` | float | Maximum Link Utilization for this time slot |
| `objectives[].inv_load_cost` | float | Inverse load cost for this time slot |
| `objectives[].jain_index` | float | Jain's fairness index for this time slot |
| `saturations` | array | Array of link saturation values (sorted by saturation, descending) |
| `saturations[].t` | integer | Time slot index |
| `saturations[].from` | string | Source node identifier |
| `saturations[].to` | string | Destination node identifier |
| `saturations[].sat` | float | Link saturation value between 0.0 and 1.0 |

### Invalid Solution Output

When the solution fails validation checks, the JSON output is minimal:

```json
{
  "valid": false
}
```

Check the verbose output or `checker.log` file for detailed error messages explaining why the solution failed validation.

## Contributors

See [CONTRIBUTORS.md](CONTRIBUTORS.md) for the list of project contributors.

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

Copyright (c) 2026 Orange SA
