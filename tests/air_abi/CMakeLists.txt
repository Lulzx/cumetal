if(NOT APPLE)
    message(STATUS "Skipping air_abi tests: Apple platform required")
    return()
endif()

add_executable(cumetal_metal_load_test
    metallib_load_test.mm
)

target_link_libraries(cumetal_metal_load_test
    PRIVATE
        "-framework Foundation"
        "-framework Metal"
)

target_compile_options(cumetal_metal_load_test
    PRIVATE
        -fobjc-arc
        -Wall
        -Wextra
)

add_test(
    NAME air_abi_metal_load
    COMMAND cumetal_metal_load_test ${CMAKE_CURRENT_SOURCE_DIR}/reference/reference.metallib
)

set_tests_properties(air_abi_metal_load PROPERTIES SKIP_RETURN_CODE 77)

add_test(
    NAME air_abi_emit_validate_experimental
    COMMAND ${CMAKE_CURRENT_SOURCE_DIR}/run_experimental_validation.sh
            $<TARGET_FILE:cumetal-air-emitter>
            $<TARGET_FILE:air_validate>
            ${CMAKE_CURRENT_SOURCE_DIR}/reference/vector_add_air.ll
            ${CMAKE_CURRENT_BINARY_DIR}/vector_add.experimental.metallib
)

add_test(
    NAME air_abi_validate_reference
    COMMAND ${CMAKE_CURRENT_SOURCE_DIR}/run_reference_validation.sh
            ${CMAKE_SOURCE_DIR}/scripts/generate_reference_metallib.sh
            $<TARGET_FILE:air_validate>
            $<TARGET_FILE:air_inspect>
            ${CMAKE_CURRENT_SOURCE_DIR}/reference/reference.metallib
)

add_test(
    NAME air_abi_emit_validate_xcrun
    COMMAND ${CMAKE_CURRENT_SOURCE_DIR}/run_xcrun_emitter_validation.sh
            $<TARGET_FILE:cumetal-air-emitter>
            $<TARGET_FILE:air_validate>
            ${CMAKE_CURRENT_SOURCE_DIR}/reference/vector_add.metal
            ${CMAKE_CURRENT_BINARY_DIR}/vector_add.xcrun.metallib
)

add_test(
    NAME air_abi_emit_load_xcrun
    COMMAND ${CMAKE_CURRENT_SOURCE_DIR}/run_xcrun_emit_and_load.sh
            $<TARGET_FILE:cumetal-air-emitter>
            $<TARGET_FILE:cumetal_metal_load_test>
            ${CMAKE_CURRENT_SOURCE_DIR}/reference/vector_add.metal
            ${CMAKE_CURRENT_BINARY_DIR}/vector_add.xcrun.load.metallib
)

add_test(
    NAME air_abi_cumetalc_emit_load_xcrun
    COMMAND ${CMAKE_CURRENT_SOURCE_DIR}/run_xcrun_emit_and_load.sh
            $<TARGET_FILE:cumetalc>
            $<TARGET_FILE:cumetal_metal_load_test>
            ${CMAKE_CURRENT_SOURCE_DIR}/reference/vector_add.metal
            ${CMAKE_CURRENT_BINARY_DIR}/vector_add.cumetalc.xcrun.load.metallib
)

add_test(
    NAME air_abi_cumetalc_positional_emit_load_xcrun
    COMMAND bash
            ${CMAKE_CURRENT_SOURCE_DIR}/run_cumetalc_positional_emit_and_load.sh
            $<TARGET_FILE:cumetalc>
            $<TARGET_FILE:cumetal_metal_load_test>
            ${CMAKE_CURRENT_SOURCE_DIR}/reference/vector_add.metal
            ${CMAKE_CURRENT_BINARY_DIR}/vector_add.cumetalc.positional.xcrun.load.metallib
)

add_test(
    NAME air_abi_validate_negative
    COMMAND ${CMAKE_CURRENT_SOURCE_DIR}/run_negative_validation.sh
            $<TARGET_FILE:air_validate>
            ${CMAKE_CURRENT_BINARY_DIR}/negative
)

set_tests_properties(
    air_abi_validate_reference
    air_abi_emit_validate_xcrun
    air_abi_emit_load_xcrun
    air_abi_cumetalc_emit_load_xcrun
    air_abi_cumetalc_positional_emit_load_xcrun
    PROPERTIES
        SKIP_RETURN_CODE 77
)
