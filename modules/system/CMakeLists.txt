cmake_minimum_required(VERSION 3.9)

project (HCLIB_SYSTEM)

SET_PROPERTY(GLOBAL PROPERTY TARGET_SUPPORTS_SHARED_LIBS TRUE)

SET(MODULE_NAME "system")

list(APPEND CMAKE_PREFIX_PATH "$ENV{HCLIB_ROOT}")

find_package(hclib REQUIRED)

FILE(GLOB SOURCES "${CMAKE_CURRENT_LIST_DIR}/src/*.cpp" "${CMAKE_CURRENT_LIST_DIR}/src/*.c")
FILE(GLOB HEADERS "${CMAKE_CURRENT_LIST_DIR}/inc/*.h" "${CMAKE_CURRENT_LIST_DIR}/../common/*.h")

string(APPEND CMAKE_CXX_FLAGS " -g -O3 ")

ADD_LIBRARY("hclib_${MODULE_NAME}" MODULE "${SOURCES}")

target_link_libraries("hclib_${MODULE_NAME}" PRIVATE hclib::hclib)

target_compile_features("hclib_${MODULE_NAME}" PUBLIC cxx_std_11)

target_include_directories("hclib_${MODULE_NAME}"
    PUBLIC
    $<BUILD_INTERFACE:${CMAKE_CURRENT_LIST_DIR}/inc>
    $<INSTALL_INTERFACE:include>)

target_compile_definitions("hclib_${MODULE_NAME}" PRIVATE ${COMPILE_DEFINITIONS})

if (CMAKE_INSTALL_PREFIX_INITIALIZED_TO_DEFAULT)
    set (CMAKE_INSTALL_PREFIX "$ENV{HCLIB_ROOT}" CACHE PATH "default install path" FORCE )
endif()

install(TARGETS "hclib_${MODULE_NAME}" EXPORT "hclib_${MODULE_NAME}-targets"
        INCLUDES DESTINATION include
        ARCHIVE DESTINATION lib
        LIBRARY DESTINATION lib)

install(EXPORT "hclib_${MODULE_NAME}-targets"
    FILE "hclib_${MODULE_NAME}Targets.cmake"
    NAMESPACE hclib::
    DESTINATION cmake)

install(FILES ${HEADERS} DESTINATION include)
