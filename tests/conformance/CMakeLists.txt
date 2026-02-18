if(NOT APPLE)
    message(STATUS "Skipping conformance tests: Apple platform required")
    return()
endif()

add_test(
    NAME conformance_phase4_functional
    COMMAND ${CMAKE_CURRENT_SOURCE_DIR}/run_conformance_suite.sh
            ${CMAKE_BINARY_DIR}
            90
            ^functional_
)

set_tests_properties(conformance_phase4_functional PROPERTIES SKIP_RETURN_CODE 77)
