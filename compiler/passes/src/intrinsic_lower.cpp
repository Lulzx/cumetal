#include "cumetal/passes/intrinsic_lower.h"

#include <utility>

namespace cumetal::passes {
namespace {

bool map_special_register_mov(const cumetal::ptx::EntryFunction::Instruction& instruction,
                              LoweredInstruction* lowered) {
    if (lowered == nullptr || instruction.opcode.rfind("mov", 0) != 0 || instruction.operands.size() < 2) {
        return false;
    }

    const std::string& src = instruction.operands[1];
    std::string mapped;
    if (src == "%tid.x") {
        mapped = "air.thread_position_in_threadgroup.x";
    } else if (src == "%tid.y") {
        mapped = "air.thread_position_in_threadgroup.y";
    } else if (src == "%tid.z") {
        mapped = "air.thread_position_in_threadgroup.z";
    } else if (src == "%ctaid.x") {
        mapped = "air.threadgroup_position_in_grid.x";
    } else if (src == "%ctaid.y") {
        mapped = "air.threadgroup_position_in_grid.y";
    } else if (src == "%ctaid.z") {
        mapped = "air.threadgroup_position_in_grid.z";
    } else if (src == "%ntid.x") {
        mapped = "air.threads_per_threadgroup.x";
    } else if (src == "%ntid.y") {
        mapped = "air.threads_per_threadgroup.y";
    } else if (src == "%ntid.z") {
        mapped = "air.threads_per_threadgroup.z";
    } else if (src == "%nctaid.x") {
        mapped = "air.threadgroups_per_grid.x";
    } else if (src == "%nctaid.y") {
        mapped = "air.threadgroups_per_grid.y";
    } else if (src == "%nctaid.z") {
        mapped = "air.threadgroups_per_grid.z";
    } else {
        return false;
    }

    lowered->opcode = "air.read.special_register";
    lowered->operands = {instruction.operands[0], mapped};
    lowered->translated = true;
    return true;
}

bool map_barrier(const cumetal::ptx::EntryFunction::Instruction& instruction, LoweredInstruction* lowered) {
    if (lowered == nullptr || instruction.opcode.rfind("bar.sync", 0) != 0) {
        return false;
    }
    lowered->opcode = "air.threadgroup_barrier";
    lowered->operands = instruction.operands;
    lowered->translated = true;
    return true;
}

bool map_math(const cumetal::ptx::EntryFunction::Instruction& instruction, LoweredInstruction* lowered) {
    if (lowered == nullptr) {
        return false;
    }

    if (instruction.opcode.rfind("add", 0) == 0) {
        lowered->opcode = "llvm.add";
    } else if (instruction.opcode.rfind("sub", 0) == 0) {
        lowered->opcode = "llvm.sub";
    } else if (instruction.opcode.rfind("mul", 0) == 0) {
        lowered->opcode = "llvm.mul";
    } else if (instruction.opcode.rfind("div", 0) == 0) {
        lowered->opcode = "llvm.div";
    } else if (instruction.opcode.rfind("rem", 0) == 0) {
        lowered->opcode = "llvm.rem";
    } else if (instruction.opcode.rfind("and", 0) == 0) {
        lowered->opcode = "llvm.and";
    } else if (instruction.opcode.rfind("or", 0) == 0) {
        lowered->opcode = "llvm.or";
    } else if (instruction.opcode.rfind("xor", 0) == 0) {
        lowered->opcode = "llvm.xor";
    } else if (instruction.opcode.rfind("not", 0) == 0) {
        lowered->opcode = "llvm.not";
    } else if (instruction.opcode.rfind("shl", 0) == 0) {
        lowered->opcode = "llvm.shl";
    } else if (instruction.opcode.rfind("shr", 0) == 0) {
        lowered->opcode = "llvm.shr";
    } else if (instruction.opcode.rfind("selp", 0) == 0) {
        lowered->opcode = "llvm.select";
    } else if (instruction.opcode.rfind("rcp", 0) == 0) {
        lowered->opcode = "llvm.rcp";
    } else if (instruction.opcode.rfind("neg", 0) == 0) {
        const bool is_float = instruction.opcode.find(".f32") != std::string::npos ||
                              instruction.opcode.find(".f64") != std::string::npos;
        lowered->opcode = is_float ? "llvm.fneg" : "llvm.neg";
    } else if (instruction.opcode.rfind("fma", 0) == 0) {
        lowered->opcode = "llvm.fma";
    } else if (instruction.opcode.rfind("mad", 0) == 0) {
        const bool is_float = instruction.opcode.find(".f32") != std::string::npos ||
                              instruction.opcode.find(".f64") != std::string::npos;
        lowered->opcode = is_float ? "llvm.fma" : "llvm.mad";
    } else if (instruction.opcode.rfind("max", 0) == 0) {
        const bool is_float = instruction.opcode.find(".f32") != std::string::npos ||
                              instruction.opcode.find(".f64") != std::string::npos;
        lowered->opcode = is_float ? "llvm.fmax" : "llvm.max";
    } else if (instruction.opcode.rfind("min", 0) == 0) {
        const bool is_float = instruction.opcode.find(".f32") != std::string::npos ||
                              instruction.opcode.find(".f64") != std::string::npos;
        lowered->opcode = is_float ? "llvm.fmin" : "llvm.min";
    } else if (instruction.opcode.rfind("abs", 0) == 0) {
        const bool is_float = instruction.opcode.find(".f32") != std::string::npos ||
                              instruction.opcode.find(".f64") != std::string::npos;
        lowered->opcode = is_float ? "llvm.fabs" : "llvm.abs";
    } else if (instruction.opcode.rfind("sqrt", 0) == 0) {
        lowered->opcode = "llvm.sqrt";
    } else if (instruction.opcode.rfind("rsqrt", 0) == 0) {
        lowered->opcode = "llvm.rsqrt";
    } else if (instruction.opcode.rfind("ex2", 0) == 0) {
        lowered->opcode = "llvm.exp2";
    } else if (instruction.opcode.rfind("lg2", 0) == 0) {
        lowered->opcode = "llvm.log2";
    } else if (instruction.opcode.rfind("sin", 0) == 0) {
        lowered->opcode = "llvm.sin";
    } else if (instruction.opcode.rfind("cos", 0) == 0) {
        lowered->opcode = "llvm.cos";
    } else {
        return false;
    }

    lowered->operands = instruction.operands;
    lowered->translated = true;
    return true;
}

bool map_warp_primitives(const cumetal::ptx::EntryFunction::Instruction& instruction,
                          LoweredInstruction* lowered) {
    if (lowered == nullptr) {
        return false;
    }

    const std::string& op = instruction.opcode;

    // shfl: shuffle within a SIMD-group (warp)
    // shfl.sync.{idx,down,up,bfly}.b32 dst, src, sel, clamp, membermask
    if (op.rfind("shfl", 0) == 0) {
        if (op.find(".idx.") != std::string::npos) {
            lowered->opcode = "air.simdgroup.shuffle";
        } else if (op.find(".down.") != std::string::npos) {
            lowered->opcode = "air.simdgroup.shuffle_down";
        } else if (op.find(".up.") != std::string::npos) {
            lowered->opcode = "air.simdgroup.shuffle_up";
        } else if (op.find(".bfly.") != std::string::npos) {
            lowered->opcode = "air.simdgroup.shuffle_xor";
        } else {
            return false;
        }
        lowered->operands = instruction.operands;
        lowered->translated = true;
        return true;
    }

    // vote: warp-wide predicate reduction (ballot / any / all)
    // vote.[sync.]{ballot.b32,any.pred,all.pred}
    if (op.rfind("vote", 0) == 0) {
        if (op.find(".ballot.") != std::string::npos) {
            lowered->opcode = "air.simdgroup.ballot";
        } else if (op.find(".any.") != std::string::npos) {
            lowered->opcode = "air.simdgroup.any";
        } else if (op.find(".all.") != std::string::npos) {
            lowered->opcode = "air.simdgroup.all";
        } else {
            return false;
        }
        lowered->operands = instruction.operands;
        lowered->translated = true;
        return true;
    }

    // bar.warp.sync → simdgroup barrier (__syncwarp)
    if (op.rfind("bar.warp.sync", 0) == 0) {
        lowered->opcode = "air.simdgroup.barrier";
        lowered->operands = instruction.operands;
        lowered->translated = true;
        return true;
    }

    return false;
}

}  // namespace

IntrinsicLowerResult lower_intrinsics(const cumetal::ptx::EntryFunction& entry,
                                      const IntrinsicLowerOptions& options) {
    IntrinsicLowerResult result;
    for (const auto& instruction : entry.instructions) {
        LoweredInstruction lowered;
        lowered.opcode = instruction.opcode;
        lowered.operands = instruction.operands;
        lowered.translated = false;

        bool translated = false;
        translated = translated || map_special_register_mov(instruction, &lowered);
        translated = translated || map_barrier(instruction, &lowered);
        translated = translated || map_warp_primitives(instruction, &lowered);
        translated = translated || map_math(instruction, &lowered);

        if (!translated) {
            if (instruction.opcode.rfind("ret", 0) == 0 || instruction.opcode.rfind("ld", 0) == 0 ||
                instruction.opcode.rfind("st", 0) == 0 || instruction.opcode.rfind("setp", 0) == 0 ||
                instruction.opcode.rfind("bra", 0) == 0 || instruction.opcode.rfind("cvt", 0) == 0 ||
                instruction.opcode.rfind("cvta", 0) == 0 || instruction.opcode.rfind("mov", 0) == 0 ||
                instruction.opcode.rfind("call", 0) == 0 || instruction.opcode.rfind("atom", 0) == 0 ||
                instruction.opcode.rfind("selp", 0) == 0) {
                // keep instruction as-is; lowering not needed yet for this stage.
            } else {
                result.warnings.push_back("intrinsic_lower: no mapping for opcode '" + instruction.opcode + "'");
            }
        }

        result.instructions.push_back(std::move(lowered));
    }

    if (options.strict && !result.warnings.empty()) {
        result.error = result.warnings.front();
        return result;
    }

    result.ok = true;
    return result;
}

}  // namespace cumetal::passes
