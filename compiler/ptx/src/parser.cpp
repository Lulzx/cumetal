#include "cumetal/ptx/parser.h"

#include <cctype>
#include <regex>
#include <utility>

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

}  // namespace

ParseResult parse_ptx(std::string_view text) {
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
        R"(\.entry\s+([A-Za-z_.$][A-Za-z0-9_.$]*)\s*\(([\s\S]*?)\)\s*\{)");
    const std::regex param_regex(R"(\.param\s+([A-Za-z0-9_\.]+)\s+([A-Za-z0-9_.$]+))");

    std::sregex_iterator iter(source.begin(), source.end(), entry_regex);
    const std::sregex_iterator end;

    for (; iter != end; ++iter) {
        if (iter->size() < 3) {
            continue;
        }

        EntryFunction entry;
        entry.name = (*iter)[1].str();

        const std::string params_blob = (*iter)[2].str();
        std::sregex_iterator param_iter(params_blob.begin(), params_blob.end(), param_regex);
        for (; param_iter != end; ++param_iter) {
            if (param_iter->size() < 3) {
                continue;
            }
            entry.params.push_back({.type = (*param_iter)[1].str(), .name = (*param_iter)[2].str()});
        }

        result.module.entries.push_back(std::move(entry));
    }

    if (result.module.entries.empty()) {
        result.error = "no .entry definitions found";
        return result;
    }

    result.ok = true;
    return result;
}

}  // namespace cumetal::ptx
