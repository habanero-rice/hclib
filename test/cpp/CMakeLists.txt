
cmake_minimum_required (VERSION 3.8)
project (HCLIB_async0)

list(APPEND CMAKE_PREFIX_PATH "$ENV{HCLIB_ROOT}")

add_executable(async0
  async0.cpp
)

find_package(hclib REQUIRED)

target_link_libraries(async0 hclib::hclib)

