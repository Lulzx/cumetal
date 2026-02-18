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

set_tests_properties(conformance_phase4_functional PROPERTIES
    SKIP_RETURN_CODE 77
    TIMEOUT 1800
)

set(CUMETAL_LLMC_DIR_ENV "$ENV{CUMETAL_LLMC_DIR}")
set(CUMETAL_LLMC_DIR_AUTO "")
if(EXISTS "${CMAKE_SOURCE_DIR}/../llm.c")
    set(CUMETAL_LLMC_DIR_AUTO "${CMAKE_SOURCE_DIR}/../llm.c")
elseif(EXISTS "${CMAKE_SOURCE_DIR}/llm.c")
    set(CUMETAL_LLMC_DIR_AUTO "${CMAKE_SOURCE_DIR}/llm.c")
endif()

if((CUMETAL_LLMC_DIR_ENV AND EXISTS "${CUMETAL_LLMC_DIR_ENV}") OR CUMETAL_LLMC_DIR_AUTO)
    add_test(
        NAME conformance_llmc_gpt2fp32cu
        COMMAND ${CMAKE_CURRENT_SOURCE_DIR}/run_llmc_gpt2fp32cu.sh
    )

    set_tests_properties(conformance_llmc_gpt2fp32cu PROPERTIES
        SKIP_RETURN_CODE 77
        TIMEOUT 1800
    )
else()
    message(STATUS "Skipping llm.c conformance registration (set CUMETAL_LLMC_DIR to enable)")
endif()
