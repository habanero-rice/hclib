
cmake_minimum_required (VERSION 3.9)
project (HCLIB_async0)

add_executable(async0
  async0.cpp
  $ENV{HCLIB_ROOT}/../modules/system/src/hclib_system.cpp
)

find_package(hclib REQUIRED)

target_link_libraries(async0 hclib::hclib)

set(HCLIB_MOD_SYS_INC
    $ENV{HCLIB_ROOT}/../modules/system/inc
)
target_include_directories(async0 PUBLIC ${HCLIB_MOD_SYS_INC})

