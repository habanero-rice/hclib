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

configure_file(
 ${CMAKE_CURRENT_SOURCE_DIR}/inc/hclib_config.h.in
 ${CMAKE_CURRENT_BINARY_DIR}/inc/hclib_config.h @ONLY
)

add_subdirectory(src)

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
