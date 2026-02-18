add_test(
    NAME ptx_sweep_supported_ops
    COMMAND bash
            ${CMAKE_CURRENT_SOURCE_DIR}/run_supported_ops.sh
            $<TARGET_FILE:cumetal-ptx2llvm>
            ${CMAKE_CURRENT_BINARY_DIR}
)

add_test(
    NAME ptx_sweep_unsupported_ops
    COMMAND bash
            ${CMAKE_CURRENT_SOURCE_DIR}/run_unsupported_ops.sh
            $<TARGET_FILE:cumetal-ptx2llvm>
            ${CMAKE_CURRENT_BINARY_DIR}
)
