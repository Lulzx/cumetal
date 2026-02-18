#include "cumetal/passes/printf_lower.h"

#include <algorithm>
#include <cctype>
#include <cstddef>
#include <map>
#include <string>
#include <string_view>
#include <utility>

namespace cumetal::passes {
namespace {

std::string trim(std::string_view text) {
    std::size_t begin = 0;
    while (begin < text.size() && std::isspace(static_cast<unsigned char>(text[begin])) != 0) {
        ++begin;
    }
    std::size_t end = text.size();
    while (end > begin && std::isspace(static_cast<unsigned char>(text[end - 1])) != 0) {
        --end;
    }
    return std::string(text.substr(begin, end - begin));
}

std::string lowercase(std::string value) {
    std::transform(value.begin(), value.end(), value.begin(), [](unsigned char c) {
        return static_cast<char>(std::tolower(c));
    });
    return value;
}

bool is_printf_symbol(const std::string& token) {
    const std::string lowered = lowercase(token);
    return lowered.find("vprintf") != std::string::npos || lowered.find("printf") != std::string::npos;
}

std::size_t find_printf_callee_index(const std::vector<std::string>& operands) {
    for (std::size_t i = 0; i < operands.size(); ++i) {
        if (is_printf_symbol(trim(operands[i]))) {
            return i;
        }
    }
    return operands.size();
}

std::vector<std::string> split_call_args(std::string text) {
    text = trim(text);
    if (text.size() >= 2 && text.front() == '(' && text.back() == ')') {
        text = text.substr(1, text.size() - 2);
    }

    std::vector<std::string> args;
    std::string current;
    int bracket_depth = 0;
    bool in_quote = false;
    bool escaped = false;

    for (char c : text) {
        if (in_quote) {
            current.push_back(c);
            if (escaped) {
                escaped = false;
            } else if (c == '\\') {
                escaped = true;
            } else if (c == '"') {
                in_quote = false;
            }
            continue;
        }

        if (c == '"') {
            in_quote = true;
            current.push_back(c);
            continue;
        }
        if (c == '[' || c == '(' || c == '{') {
            ++bracket_depth;
            current.push_back(c);
            continue;
        }
        if (c == ']' || c == ')' || c == '}') {
            if (bracket_depth > 0) {
                --bracket_depth;
            }
            current.push_back(c);
            continue;
        }
        if (c == ',' && bracket_depth == 0) {
            const std::string arg = trim(current);
            if (!arg.empty()) {
                args.push_back(arg);
            }
            current.clear();
            continue;
        }

        current.push_back(c);
    }

    const std::string tail = trim(current);
    if (!tail.empty()) {
        args.push_back(tail);
    }
    return args;
}

bool is_quoted_string(const std::string& token) {
    return token.size() >= 2 && token.front() == '"' && token.back() == '"';
}

std::string unescape_string_literal(const std::string& token) {
    if (!is_quoted_string(token)) {
        return token;
    }

    std::string out;
    out.reserve(token.size());
    for (std::size_t i = 1; i + 1 < token.size(); ++i) {
        char c = token[i];
        if (c == '\\' && i + 2 < token.size()) {
            const char escaped = token[++i];
            switch (escaped) {
                case 'n':
                    out.push_back('\n');
                    break;
                case 't':
                    out.push_back('\t');
                    break;
                case '\\':
                    out.push_back('\\');
                    break;
                case '"':
                    out.push_back('"');
                    break;
                default:
                    out.push_back('\\');
                    out.push_back(escaped);
                    break;
            }
            continue;
        }
        out.push_back(c);
    }
    return out;
}

bool fail_or_warn(bool strict,
                  const std::string& message,
                  std::vector<std::string>* warnings,
                  std::string* error) {
    if (strict) {
        if (error != nullptr) {
            *error = message;
        }
        return true;
    }
    if (warnings != nullptr) {
        warnings->push_back(message);
    }
    return false;
}

}  // namespace

PrintfLowerResult lower_printf_calls(const cumetal::ptx::EntryFunction& entry,
                                     const PrintfLowerOptions& options) {
    PrintfLowerResult result;

    std::map<std::string, std::uint32_t> format_ids;

    for (const auto& instruction : entry.instructions) {
        if (instruction.opcode.rfind("call", 0) != 0) {
            continue;
        }

        const std::size_t callee_index = find_printf_callee_index(instruction.operands);
        if (callee_index == instruction.operands.size()) {
            continue;
        }

        if (callee_index + 1 >= instruction.operands.size()) {
            const std::string message = "printf_lower: missing argument tuple at line " +
                                        std::to_string(instruction.line);
            if (fail_or_warn(options.strict, message, &result.warnings, &result.error)) {
                return result;
            }
            continue;
        }

        const std::vector<std::string> args = split_call_args(instruction.operands[callee_index + 1]);
        if (args.empty()) {
            const std::string message = "printf_lower: empty argument tuple at line " +
                                        std::to_string(instruction.line);
            if (fail_or_warn(options.strict, message, &result.warnings, &result.error)) {
                return result;
            }
            continue;
        }

        const std::string format_token_raw = trim(args.front());
        if (format_token_raw.empty()) {
            const std::string message = "printf_lower: empty format token at line " +
                                        std::to_string(instruction.line);
            if (fail_or_warn(options.strict, message, &result.warnings, &result.error)) {
                return result;
            }
            continue;
        }

        bool literal = false;
        bool truncated = false;
        std::string canonical_token = format_token_raw;
        if (is_quoted_string(format_token_raw)) {
            literal = true;
            canonical_token = unescape_string_literal(format_token_raw);
            if (canonical_token.size() > options.max_format_length) {
                canonical_token.resize(options.max_format_length);
                truncated = true;
                result.warnings.push_back(
                    "printf_lower: format literal truncated to " +
                    std::to_string(options.max_format_length) + " bytes at line " +
                    std::to_string(instruction.line));
            }
        }

        std::uint32_t format_id = 0;
        const auto existing = format_ids.find(canonical_token);
        if (existing == format_ids.end()) {
            format_id = static_cast<std::uint32_t>(result.formats.size());
            format_ids[canonical_token] = format_id;
            result.formats.push_back(
                {.id = format_id, .token = canonical_token, .literal = literal, .truncated = truncated});
        } else {
            format_id = existing->second;
            if (truncated && format_id < result.formats.size()) {
                result.formats[format_id].truncated = true;
            }
        }

        PrintfLoweredCall call;
        call.source_line = instruction.line;
        call.source_opcode = instruction.opcode;
        call.format_id = format_id;
        call.format_token = canonical_token;
        call.arguments.assign(args.begin() + 1, args.end());
        result.calls.push_back(std::move(call));
    }

    result.ok = true;
    return result;
}

}  // namespace cumetal::passes
