cmake_minimum_required (VERSION 3.9)

# Silence an error about the version option to project
# this will need to be changed if we ever bump the version to 3.0 and put the
# version stuff in the project command itself
#if (NOT CMAKE_VERSION VERSION_LESS 3.0)
#  cmake_policy(SET CMP0048 OLD)
#endif()
#cmake_policy(SET CMP0025 NEW)
#cmake_policy(SET CMP0028 OLD)

project (HCLIB)
# The version number.
set (HCLIB_VERSION_MAJOR 0)
set (HCLIB_VERSION_MINOR 1)


#include (CheckCXXCompilerFlag)
#include (CheckCXXSourceCompiles)

enable_testing()


option(HCLIB_ENABLE_VERBOSE     "whether verbose HC runtime"          OFF)
option(HCLIB_ENABLE_STATS       "whether to turn on HC runtime stats" OFF)
option(HCLIB_ENABLE_HWLOC       "whether to enable HWLoc"             OFF)
option(HCLIB_ENABLE_PRODUCTION  "disable assertions and stats"        OFF)
option(HCLIB_PRIVATIZE_C_STD_11    "whether to publicize c_std_11 compiler features" OFF)

set(COMPILE_DEFINITIONS "")

if (HCLIB_ENABLE_HWLOC)
    list(APPEND COMPILE_DEFINITIONS "USE_HWLOC ")
endif(HCLIB_ENABLE_HWLOC)

if (HCLIB_ENABLE_STATS)
    list(APPEND COMPILE_DEFINITIONS "HCLIB_STATS ")
endif(HCLIB_ENABLE_STATS)

if (HCLIB_ENABLE_VERBOSE)
    list(APPEND COMPILE_DEFINITIONS "VERBOSE ")
endif(HCLIB_ENABLE_VERBOSE)

if (NOT HCLIB_ENABLE_PRODUCTION)
    list(APPEND COMPILE_DEFINITIONS "HC_ASSERTION_CHECK ")
endif()

message("COMPILE_DEFINITIONS: " ${COMPILE_DEFINITIONS})

set(HCLIB_MODULES "system" CACHE STRING "A semicolon separated list of modules to compile")

if (CMAKE_INSTALL_PREFIX_INITIALIZED_TO_DEFAULT)
    set (CMAKE_INSTALL_PREFIX "${CMAKE_CURRENT_SOURCE_DIR}/hclib-install" CACHE PATH "default install path" FORCE )
endif()

configure_file(
 ${CMAKE_CURRENT_SOURCE_DIR}/inc/hclib_config.h.in
 ${CMAKE_CURRENT_BINARY_DIR}/inc/hclib_config.h @ONLY
)

add_subdirectory(src)

foreach(_mod ${HCLIB_MODULES})
    add_subdirectory("modules/${_mod}")
endforeach()

include(CMakePackageConfigHelpers)
configure_package_config_file(cmake/hclibConfig.cmake.in
                              ${CMAKE_CURRENT_BINARY_DIR}/hclibConfig.cmake 
                              INSTALL_DESTINATION cmake)
write_basic_package_version_file(${CMAKE_CURRENT_BINARY_DIR}/hclibConfigVersion.cmake
                                 VERSION "${HCLIB_VERSION_MAJOR}.${HCLIB_VERSION_MINOR}"
                                 COMPATIBILITY SameMajorVersion)

install(
  FILES 
  ${CMAKE_CURRENT_BINARY_DIR}/hclibConfig.cmake
  ${CMAKE_CURRENT_BINARY_DIR}/hclibConfigVersion.cmake
  DESTINATION cmake 
)



install(
  FILES 
  ${CMAKE_CURRENT_BINARY_DIR}/inc/hclib_config.h
  DESTINATION include
)

FILE(GLOB INCL_LIST ${CMAKE_CURRENT_SOURCE_DIR}/inc/*.h)
install(
  FILES ${INCL_LIST}
  DESTINATION include
)

