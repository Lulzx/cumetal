if(NOT APPLE)
    message(STATUS "Skipping functional tests: Apple platform required")
    return()
endif()

add_executable(cumetal_runtime_vector_add_test
    runtime_vector_add_test.cpp
)

add_executable(cumetal_runtime_stream_vector_add_test
    runtime_stream_vector_add_test.cpp
)

add_executable(cumetal_runtime_async_memops_test
    runtime_async_memops_test.cpp
)

add_executable(cumetal_runtime_event_test
    runtime_event_test.cpp
)

add_executable(cumetal_runtime_stream_wait_event_test
    runtime_stream_wait_event_test.cpp
)

add_executable(cumetal_runtime_stream_query_test
    runtime_stream_query_test.cpp
)

add_executable(cumetal_runtime_axpy_offset_test
    runtime_axpy_offset_test.cpp
)

target_link_libraries(cumetal_runtime_vector_add_test
    PRIVATE
        cumetal_runtime
)

target_link_libraries(cumetal_runtime_stream_vector_add_test
    PRIVATE
        cumetal_runtime
)

target_link_libraries(cumetal_runtime_async_memops_test
    PRIVATE
        cumetal_runtime
)

target_link_libraries(cumetal_runtime_event_test
    PRIVATE
        cumetal_runtime
)

target_link_libraries(cumetal_runtime_stream_wait_event_test
    PRIVATE
        cumetal_runtime
)

target_link_libraries(cumetal_runtime_stream_query_test
    PRIVATE
        cumetal_runtime
)

target_link_libraries(cumetal_runtime_axpy_offset_test
    PRIVATE
        cumetal_runtime
)

target_compile_options(cumetal_runtime_vector_add_test
    PRIVATE
        -Wall
        -Wextra
        -Wpedantic
)

target_compile_options(cumetal_runtime_stream_vector_add_test
    PRIVATE
        -Wall
        -Wextra
        -Wpedantic
)

target_compile_options(cumetal_runtime_async_memops_test
    PRIVATE
        -Wall
        -Wextra
        -Wpedantic
)

target_compile_options(cumetal_runtime_event_test
    PRIVATE
        -Wall
        -Wextra
        -Wpedantic
)

target_compile_options(cumetal_runtime_stream_wait_event_test
    PRIVATE
        -Wall
        -Wextra
        -Wpedantic
)

target_compile_options(cumetal_runtime_stream_query_test
    PRIVATE
        -Wall
        -Wextra
        -Wpedantic
)

target_compile_options(cumetal_runtime_axpy_offset_test
    PRIVATE
        -Wall
        -Wextra
        -Wpedantic
)

add_test(
    NAME functional_runtime_vector_add
    COMMAND ${CMAKE_CURRENT_SOURCE_DIR}/run_runtime_vector_add.sh
            ${CMAKE_SOURCE_DIR}/scripts/generate_reference_metallib.sh
            $<TARGET_FILE:cumetal_runtime_vector_add_test>
            ${CMAKE_SOURCE_DIR}/tests/air_abi/reference/reference.metallib
)

set_tests_properties(functional_runtime_vector_add PROPERTIES SKIP_RETURN_CODE 77)

add_test(
    NAME functional_runtime_stream_vector_add
    COMMAND ${CMAKE_CURRENT_SOURCE_DIR}/run_runtime_vector_add.sh
            ${CMAKE_SOURCE_DIR}/scripts/generate_reference_metallib.sh
            $<TARGET_FILE:cumetal_runtime_stream_vector_add_test>
            ${CMAKE_SOURCE_DIR}/tests/air_abi/reference/reference.metallib
)

set_tests_properties(functional_runtime_stream_vector_add PROPERTIES SKIP_RETURN_CODE 77)

add_test(
    NAME functional_runtime_async_memops
    COMMAND ${CMAKE_CURRENT_SOURCE_DIR}/run_runtime_vector_add.sh
            ${CMAKE_SOURCE_DIR}/scripts/generate_reference_metallib.sh
            $<TARGET_FILE:cumetal_runtime_async_memops_test>
            ${CMAKE_SOURCE_DIR}/tests/air_abi/reference/reference.metallib
)

set_tests_properties(functional_runtime_async_memops PROPERTIES SKIP_RETURN_CODE 77)

add_test(
    NAME functional_runtime_event
    COMMAND ${CMAKE_CURRENT_SOURCE_DIR}/run_runtime_vector_add.sh
            ${CMAKE_SOURCE_DIR}/scripts/generate_reference_metallib.sh
            $<TARGET_FILE:cumetal_runtime_event_test>
            ${CMAKE_SOURCE_DIR}/tests/air_abi/reference/reference.metallib
)

set_tests_properties(functional_runtime_event PROPERTIES SKIP_RETURN_CODE 77)

add_test(
    NAME functional_runtime_stream_wait_event
    COMMAND ${CMAKE_CURRENT_SOURCE_DIR}/run_runtime_vector_add.sh
            ${CMAKE_SOURCE_DIR}/scripts/generate_reference_metallib.sh
            $<TARGET_FILE:cumetal_runtime_stream_wait_event_test>
            ${CMAKE_SOURCE_DIR}/tests/air_abi/reference/reference.metallib
)

set_tests_properties(functional_runtime_stream_wait_event PROPERTIES SKIP_RETURN_CODE 77)

add_test(
    NAME functional_runtime_stream_query
    COMMAND ${CMAKE_CURRENT_SOURCE_DIR}/run_runtime_vector_add.sh
            ${CMAKE_SOURCE_DIR}/scripts/generate_reference_metallib.sh
            $<TARGET_FILE:cumetal_runtime_stream_query_test>
            ${CMAKE_SOURCE_DIR}/tests/air_abi/reference/reference.metallib
)

set_tests_properties(functional_runtime_stream_query PROPERTIES SKIP_RETURN_CODE 77)

add_test(
    NAME functional_runtime_axpy_offset
    COMMAND ${CMAKE_CURRENT_SOURCE_DIR}/run_runtime_axpy.sh
            $<TARGET_FILE:cumetal_runtime_axpy_offset_test>
            ${CMAKE_CURRENT_SOURCE_DIR}/reference/axpy.metal
            ${CMAKE_CURRENT_BINARY_DIR}/axpy.metallib
)

set_tests_properties(functional_runtime_axpy_offset PROPERTIES SKIP_RETURN_CODE 77)
