
cmake_minimum_required (VERSION 3.9)
project (HCLIB_asyncAwait0)

add_executable(asyncAwait0
  asyncAwait0.cpp
  $ENV{HCLIB_ROOT}/../modules/system/src/hclib_system.cpp
)

find_package(hclib REQUIRED)

target_link_libraries(asyncAwait0 hclib::hclib)

set(HCLIB_MOD_SYS_INC
    $ENV{HCLIB_ROOT}/../modules/system/inc
)
target_include_directories(asyncAwait0 PUBLIC ${HCLIB_MOD_SYS_INC})

