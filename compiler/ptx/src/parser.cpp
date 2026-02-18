#include "cumetal/ptx/parser.h"

#include <algorithm>
#include <cctype>
#include <regex>
#include <sstream>
#include <utility>
#include <unordered_set>

namespace cumetal::ptx {
namespace {

std::string strip_comments(std::string_view text) {
    std::string out;
    out.reserve(text.size());

    enum class State { kNormal, kLineComment, kBlockComment };
    State state = State::kNormal;

    for (std::size_t i = 0; i < text.size(); ++i) {
        const char c = text[i];
        const char next = (i + 1 < text.size()) ? text[i + 1] : '\0';

        if (state == State::kLineComment) {
            if (c == '\n') {
                state = State::kNormal;
                out.push_back(c);
            }
            continue;
        }

        if (state == State::kBlockComment) {
            if (c == '*' && next == '/') {
                state = State::kNormal;
                ++i;
            }
            continue;
        }

        if (c == '/' && next == '/') {
            state = State::kLineComment;
            ++i;
            continue;
        }

        if (c == '/' && next == '*') {
            state = State::kBlockComment;
            ++i;
            continue;
        }

        out.push_back(c);
    }

    return out;
}

bool parse_number(const std::string& token, int* out) {
    if (out == nullptr || token.empty()) {
        return false;
    }
    int value = 0;
    for (const char c : token) {
        if (!std::isdigit(static_cast<unsigned char>(c))) {
            return false;
        }
        value = value * 10 + (c - '0');
    }
    *out = value;
    return true;
}

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

bool extract_balanced_body(const std::string& text,
                           std::size_t open_brace_index,
                           std::string* body,
                           std::size_t* body_start,
                           std::size_t* body_end) {
    if (body == nullptr || body_start == nullptr || body_end == nullptr ||
        open_brace_index >= text.size() || text[open_brace_index] != '{') {
        return false;
    }

    int depth = 1;
    const std::size_t start = open_brace_index + 1;
    std::size_t end = std::string::npos;
    for (std::size_t i = start; i < text.size(); ++i) {
        const char c = text[i];
        if (c == '{') {
            ++depth;
        } else if (c == '}') {
            --depth;
            if (depth == 0) {
                end = i;
                break;
            }
        }
    }

    if (end == std::string::npos || end < start) {
        return false;
    }

    *body = text.substr(start, end - start);
    *body_start = start;
    *body_end = end;
    return true;
}

bool is_supported_opcode(const std::string& opcode) {
    if (opcode.empty()) {
        return false;
    }
    const std::size_t dot = opcode.find('.');
    const std::string root = (dot == std::string::npos) ? opcode : opcode.substr(0, dot);

    static const std::unordered_set<std::string> kSupportedRoots = {
        "abs",   "add",  "and",  "atom", "bar", "bra", "call", "cvt",  "cvta", "div",
        "fma",   "ld",   "mad",  "max",  "min", "mov", "mul",  "neg",  "or",   "rem",
        "ret",   "set",  "setp", "shl",  "shr", "st",  "sub",  "vote", "xor",
    };
    return kSupportedRoots.contains(root);
}

std::vector<std::string> split_operands(const std::string& text) {
    std::vector<std::string> operands;
    std::string current;
    int bracket_depth = 0;
    for (const char c : text) {
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
            const std::string token = trim(current);
            if (!token.empty()) {
                operands.push_back(token);
            }
            current.clear();
            continue;
        }
        current.push_back(c);
    }
    const std::string tail = trim(current);
    if (!tail.empty()) {
        operands.push_back(tail);
    }
    return operands;
}

std::vector<std::string> split_tokens_ws(const std::string& text) {
    std::istringstream stream(text);
    std::vector<std::string> tokens;
    std::string token;
    while (stream >> token) {
        tokens.push_back(token);
    }
    return tokens;
}

bool is_ptx_type_token(const std::string& token) {
    if (token.size() < 2 || token[0] != '.') {
        return false;
    }

    static const std::unordered_set<std::string> kNonTypeQualifiers = {
        ".param",
        ".ptr",
        ".align",
    };
    return !kNonTypeQualifiers.contains(token);
}

std::string sanitize_param_name(std::string name) {
    while (!name.empty() && (name.back() == ',' || name.back() == ';' || name.back() == ')')) {
        name.pop_back();
    }
    return name;
}

int count_line_number_at_offset(const std::string& text, std::size_t offset) {
    int line = 1;
    const std::size_t clamped = std::min(offset, text.size());
    for (std::size_t i = 0; i < clamped; ++i) {
        if (text[i] == '\n') {
            ++line;
        }
    }
    return line;
}

void parse_instructions(const std::string& body,
                        int start_line,
                        EntryFunction* entry,
                        std::vector<std::string>* warnings) {
    if (entry == nullptr || warnings == nullptr) {
        return;
    }

    int line = start_line;

    std::istringstream stream(body);
    std::string raw_line;
    while (std::getline(stream, raw_line)) {
        const int current_line = line++;
        std::string line_text = trim(raw_line);
        if (line_text.empty()) {
            continue;
        }
        if (line_text == "{" || line_text == "}") {
            continue;
        }
        if (line_text.back() == ':') {
            continue;
        }
        if (!line_text.empty() && line_text[0] == '.') {
            continue;
        }
        if (!line_text.empty() && line_text.back() == ';') {
            line_text.pop_back();
            line_text = trim(line_text);
        }
        if (line_text.empty()) {
            continue;
        }

        EntryFunction::Instruction instruction;
        instruction.line = current_line;

        if (line_text[0] == '@') {
            const std::size_t ws = line_text.find_first_of(" \t");
            if (ws == std::string::npos) {
                continue;
            }
            instruction.predicate = line_text.substr(0, ws);
            line_text = trim(line_text.substr(ws + 1));
        }

        if (line_text.empty()) {
            continue;
        }

        const std::size_t ws = line_text.find_first_of(" \t");
        if (ws == std::string::npos) {
            instruction.opcode = line_text;
        } else {
            instruction.opcode = line_text.substr(0, ws);
            instruction.operands = split_operands(line_text.substr(ws + 1));
        }

        instruction.supported = is_supported_opcode(instruction.opcode);
        if (!instruction.supported) {
            warnings->push_back("unsupported opcode '" + instruction.opcode + "' at line " +
                                std::to_string(instruction.line));
        }
        entry->instructions.push_back(std::move(instruction));
    }
}

}  // namespace

ParseResult parse_ptx(std::string_view text) {
    return parse_ptx(text, ParseOptions{});
}

ParseResult parse_ptx(std::string_view text, const ParseOptions& options) {
    ParseResult result;
    const std::string source = strip_comments(text);

    std::smatch match;
    const std::regex version_regex(R"(\.version\s+([0-9]+)\.([0-9]+))");
    if (std::regex_search(source, match, version_regex) && match.size() >= 3) {
        (void)parse_number(match[1].str(), &result.module.version_major);
        (void)parse_number(match[2].str(), &result.module.version_minor);
    }

    const std::regex target_regex(R"(\.target\s+([A-Za-z0-9_\.]+))");
    if (std::regex_search(source, match, target_regex) && match.size() >= 2) {
        result.module.target = match[1].str();
    }

    const std::regex entry_regex(
        R"(\.entry\s+([A-Za-z_.$][A-Za-z0-9_.$]*)\s*\(([\s\S]*?)\)\s*(?:\.[^\n{}]*\s*)*\{)");
    const std::regex param_decl_regex(R"(\.param\s+([^,\n\)]+))");

    std::sregex_iterator iter(source.begin(), source.end(), entry_regex);
    const std::sregex_iterator end;

    for (; iter != end; ++iter) {
        if (iter->size() < 3) {
            continue;
        }

        EntryFunction entry;
        entry.name = (*iter)[1].str();

        const std::string params_blob = (*iter)[2].str();
        std::sregex_iterator param_iter(params_blob.begin(), params_blob.end(), param_decl_regex);
        for (; param_iter != end; ++param_iter) {
            if (param_iter->size() < 2) {
                continue;
            }

            const std::string decl = trim((*param_iter)[1].str());
            if (decl.empty()) {
                continue;
            }

            const std::vector<std::string> tokens = split_tokens_ws(decl);
            if (tokens.empty()) {
                continue;
            }

            std::string type = ".u32";
            for (const std::string& token : tokens) {
                if (is_ptx_type_token(token)) {
                    type = token;
                    break;
                }
            }

            std::string name = sanitize_param_name(tokens.back());
            if (name.empty()) {
                continue;
            }

            entry.params.push_back({.type = type, .name = name});
        }

        const std::size_t open_brace = static_cast<std::size_t>(iter->position(0) + iter->length(0) - 1);
        std::string body;
        std::size_t body_start = 0;
        std::size_t body_end = 0;
        if (!extract_balanced_body(source, open_brace, &body, &body_start, &body_end)) {
            result.error = "malformed entry body for '" + entry.name + "'";
            return result;
        }
        const int start_line = count_line_number_at_offset(source, body_start);
        parse_instructions(body, start_line, &entry, &result.warnings);

        result.module.entries.push_back(std::move(entry));
    }

    if (result.module.entries.empty()) {
        result.error = "no .entry definitions found";
        return result;
    }

    if (options.strict && !result.warnings.empty()) {
        result.error = result.warnings.front();
        return result;
    }

    result.ok = true;
    return result;
}

}  // namespace cumetal::ptx
