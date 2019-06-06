
cmake_minimum_required (VERSION 3.9)
project (HCLIB_async0)

add_executable(async0
  async0.cpp
)

find_package(hclib REQUIRED system)

target_link_libraries(async0 hclib::hclib hclib::hclib_system)

