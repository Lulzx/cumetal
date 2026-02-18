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
