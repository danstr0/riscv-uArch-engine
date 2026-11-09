/**
 * @file config.hpp
 * @brief Configuration file parser.
 *
 * Reads a TOML-like configuration file that specifies:
 * - Architecture parameters (cache, pipeline, system).
 * - Initial memory contents (arrays, fill regions).
 * - Initial register values.
 * - Sweep axes (parameters with multiple values).
 *
 * @par File format
 * @code
 *   # Comments start with # or ;
 *
 *   [memory]
 *   size = 256K
 *
 *   [memory.init]
 *   0x8000 = [1, 2, 3, 4]     # u32 array
 *   0x9000 = fill(0xFF, 256)  # fill 256 bytes with 0xFF
 *   0xA000 = "hello"          # ASCII string
 *
 *   [cache.l1]
 *   size = 4K
 *   assoc = 8
 *   line = 64
 *   replacement = lru
 *   write = writeback
 *
 *   [cache.l2]
 *   size = 16K
 *   assoc = 8
 *
 *   [cache.l3]
 *   size = 64K
 *   assoc = 8
 *
 *   [pipeline]
 *   forwarding = partial
 *   branch_pred = bimodal_2bit
 *   mispredict_penalty = 3
 *
 *   [registers]
 *   a0 = 8
 *   a1 = 0x8000
 *   sp = 0xFFF0
 *
 *   [sweep]
 *   cache.l1.size = [4K, 8K, 16K]
 *   pipeline.forwarding = [none, partial]
 * @endcode
 */

#pragma once

#include "types.hpp"

#include <map>
#include <string>
#include <variant>
#include <vector>

namespace riscv {

/// Memory initialization entry.
struct MemInitEntry
{
    addr_t address;

    enum class Kind { WORDS, BYTES, FILL, STRING } kind;

    std::vector<u32> words;       ///< For @c Kind::WORDS - @c u32 values.
    std::vector<u8>  bytes;       ///< For @c Kind::BYTES - raw bytes.
    u8               fill_val;    ///< For @c Kind::FILL - fill byte.
    u32              fill_count;  ///< For @c Kind::FILL - number of bytes.
    std::string      str;         ///< For @c Kind::STRING - ASCII data.
};

/// Register initialization.
struct RegInit
{
    u32 reg_idx;
    u32 value;
};

/// Sweep axis.
struct SweepAxis
{
    std::string              param;   ///< Qualified name: "cache.l1.size".
    std::vector<std::string> values;
};

/// Per-core configuration (stepper only).
struct CoreConfig
{
    std::string          program;   ///< Path to .s file (empty = use main source).
    addr_t               pc = 0;    ///< Starting PC / load address.
    std::vector<RegInit> reg_init;  ///< Per-core register overrides.
};

/// Parsed configuration.
struct SimConfig
{
    /// @name Memory
    /// @{
    u32 mem_size_kb = 64;
    std::vector<MemInitEntry> mem_init;
    /// @}

    /// @name L1 cache
    /// @{
    u32         l1_size        = 4 * 1024; // 0 = disabled
    u32         l1_assoc       = 8;
    u32         l1_line        = 64;
    std::string l1_replacement = "lru";
    std::string l1_write       = "writeback";
    std::string l1_write_alloc = "allocate";
    /// @}

    /// @name L2 cache
    /// @{
    u32 l2_size  = 0; // 0 = disabled
    u32 l2_assoc = 8;
    /// @}

    /// @name L3 cache
    /// @{
    u32 l3_size  = 0; // 0 = disabled
    u32 l3_assoc = 16;
    /// @}

    /// @name Pipeline
    /// @{
    std::string forwarding         = "partial";
    std::string branch_pred        = "bimodal_2bit";
    u32         mispredict_penalty = 3;
    /// @}

    /// @name System
    /// @{
    std::string cpu_type = "pipelined";  ///< simple | pipelined | multicore
    u64 max_cycles = 10'000'000;
    u32 num_cores  = 1;
    /// @}

    /// @name Registers
    /// @{
    std::vector<RegInit> reg_init;  ///< Applied to all cores as baseline.
    /// @}

    /// @name Per-core
    /// @{
    std::vector<CoreConfig> cores;  ///< Stepper only; indexed by core ID.
    /// @}

    /// @name Sweep
    /// @{
    std::vector<SweepAxis> sweeps;
    /// @}
};

/// @name Parser
/// @{
struct ConfigError
{
    u32         line;
    std::string message;
};

struct ConfigResult
{
    bool                     ok = false;
    SimConfig                config;
    std::vector<ConfigError> errors;
};

/// Parse a configuration file from a string.
[[nodiscard]] ConfigResult parse_config(std::string_view source);

/// Parse a configuration file from disk.
[[nodiscard]] ConfigResult parse_config_file(const char* path);

/**
 * @brief Apply a SimConfig's architecture parameters to a set of values.
 *
 * @param param  Qualified parameter name (e.g., "cache.l1.size").
 * @param value  Value string.
 * @param cfg    SimConfig to modify.
 */
void apply_config_param(SimConfig& cfg, const std::string& param,
                        const std::string& value);

/**
 * @brief Validate a parsed @c SimConfig.
 *
 * Appends errors to @p errors.
 *
 * @return @c true if the config is valid.
 */
[[nodiscard]] bool validate_config(const SimConfig& cfg,
                                   std::vector<ConfigError>& errors);

/// Check if a parameter name is recognized.
[[nodiscard]] bool is_valid_param(const std::string& param);

/**
 * @brief Check if a value is valid for a given parameter.
 *
 * On failure, writes a human-readable message to @p error_msg.
 */
[[nodiscard]] bool is_valid_param_value(const std::string& param,
                                        const std::string& value,
                                        std::string& error_msg);
/// @}

} // namespace riscv
