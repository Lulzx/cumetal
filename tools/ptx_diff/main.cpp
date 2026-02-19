#include "cumetal/passes/intrinsic_lower.h"
#include "cumetal/ptx/lower_to_metal.h"
#include "cumetal/ptx/parser.h"

#include <cstdio>
#include <cstring>
#include <fstream>
#include <iostream>
#include <sstream>
#include <string>

namespace {

void print_usage(const char* argv0) {
    std::fprintf(stderr,
                 "usage: %s [--entry <name>] [--strict] [--metal] [ptx_file]\n"
                 "\n"
                 "  Shows the CuMetal PTX→AIR intrinsic lowering diff for a PTX kernel.\n"
                 "  If ptx_file is omitted, reads from stdin.\n"
                 "\n"
                 "  --entry <name>   Process only the named entry (default: first entry)\n"
                 "  --strict         Fail on any untranslated instruction\n"
                 "  --metal          Also attempt generic PTX→Metal source emission\n",
                 argv0);
}

std::string read_file(const char* path) {
    std::ifstream f(path);
    if (!f) {
        return "";
    }
    std::ostringstream buf;
    buf << f.rdbuf();
    return buf.str();
}

std::string read_stdin() {
    std::ostringstream buf;
    buf << std::cin.rdbuf();
    return buf.str();
}

}  // namespace

int main(int argc, char** argv) {
    std::string entry_name;
    bool strict = false;
    bool show_metal = false;
    const char* ptx_file = nullptr;

    for (int i = 1; i < argc; ++i) {
        if (std::strcmp(argv[i], "--entry") == 0 && i + 1 < argc) {
            entry_name = argv[++i];
        } else if (std::strcmp(argv[i], "--strict") == 0) {
            strict = true;
        } else if (std::strcmp(argv[i], "--metal") == 0) {
            show_metal = true;
        } else if (std::strcmp(argv[i], "--help") == 0 || std::strcmp(argv[i], "-h") == 0) {
            print_usage(argv[0]);
            return 0;
        } else if (argv[i][0] != '-') {
            ptx_file = argv[i];
        } else {
            std::fprintf(stderr, "error: unknown option '%s'\n", argv[i]);
            print_usage(argv[0]);
            return 1;
        }
    }

    const std::string ptx_source = ptx_file ? read_file(ptx_file) : read_stdin();
    if (ptx_source.empty()) {
        std::fprintf(stderr, "error: empty or unreadable PTX input\n");
        return 1;
    }

    cumetal::ptx::ParseOptions parse_opts;
    parse_opts.strict = strict;
    const auto parsed = cumetal::ptx::parse_ptx(ptx_source, parse_opts);

    if (!parsed.ok) {
        std::fprintf(stderr, "error: PTX parse failed: %s\n", parsed.error.c_str());
        return 1;
    }

    for (const auto& warn : parsed.warnings) {
        std::fprintf(stderr, "parse warning: %s\n", warn.c_str());
    }

    if (parsed.module.entries.empty()) {
        std::fprintf(stderr, "error: no .entry functions found in PTX\n");
        return 1;
    }

    // Select entry
    const cumetal::ptx::EntryFunction* entry = nullptr;
    if (entry_name.empty()) {
        entry = &parsed.module.entries[0];
    } else {
        for (const auto& e : parsed.module.entries) {
            if (e.name == entry_name) {
                entry = &e;
                break;
            }
        }
        if (entry == nullptr) {
            std::fprintf(stderr, "error: entry '%s' not found in PTX\n", entry_name.c_str());
            return 1;
        }
    }

    std::printf("Entry: %s\n", entry->name.c_str());
    std::printf("Params: %zu\n", entry->params.size());
    for (std::size_t i = 0; i < entry->params.size(); ++i) {
        std::printf("  [%zu] %s%s\n", i, entry->params[i].name.c_str(),
                    entry->params[i].is_pointer ? " (pointer)" : " (scalar)");
    }
    std::printf("\n");

    // Run intrinsic lowering
    cumetal::passes::IntrinsicLowerOptions lower_opts;
    lower_opts.strict = strict;
    const auto lowered = cumetal::passes::lower_intrinsics(*entry, lower_opts);

    if (!lowered.ok) {
        std::fprintf(stderr, "error: intrinsic lowering failed: %s\n", lowered.error.c_str());
        return 1;
    }

    // Print diff
    std::printf("Instruction lowering diff (%zu instructions):\n",
                entry->instructions.size());

    for (std::size_t i = 0; i < lowered.instructions.size(); ++i) {
        const auto& orig = entry->instructions[i];
        const auto& lo = lowered.instructions[i];

        if (lo.translated) {
            std::printf("  [%3zu] %-32s → %s\n", i, orig.opcode.c_str(),
                        lo.opcode.c_str());
        } else if (!orig.supported) {
            std::printf("  [%3zu] %-32s ! UNSUPPORTED\n", i, orig.opcode.c_str());
        } else {
            std::printf("  [%3zu] %-32s   (passthrough)\n", i, orig.opcode.c_str());
        }
    }

    if (!lowered.warnings.empty()) {
        std::printf("\nLowering warnings (%zu):\n", lowered.warnings.size());
        for (const auto& w : lowered.warnings) {
            std::printf("  %s\n", w.c_str());
        }
    }

    // Optional: generic Metal emission
    if (show_metal) {
        std::printf("\nGeneric PTX→Metal emission:\n");
        cumetal::ptx::LowerToMetalOptions metal_opts;
        metal_opts.entry_name = entry->name;
        const auto metal_result =
            cumetal::ptx::lower_ptx_to_metal_source(ptx_source, metal_opts);

        if (!metal_result.ok) {
            std::printf("  error: %s\n", metal_result.error.c_str());
        } else if (!metal_result.matched) {
            std::printf("  (not matched by generic emitter — uses hardcoded or LLVM path)\n");
            for (const auto& w : metal_result.warnings) {
                std::printf("  warning: %s\n", w.c_str());
            }
        } else {
            std::printf("  matched — emitted Metal source (%zu bytes):\n",
                        metal_result.metal_source.size());
            std::printf("---\n%s---\n", metal_result.metal_source.c_str());
        }
    }

    return 0;
}
