enable_language(C ASM)

set(SOURCES
  hclib-runtime.c 
  hclib-deque.c 
  hclib-promise.c 
  hclib-timer.c 
  hclib_cpp.cpp 
  hclib.c 
  hclib-tree.c 
  hclib-locality-graph.c 
  hclib_module.c
  hclib-fptr-list.c 
  hclib-mem.c 
  hclib-instrument.c 
  hclib_atomic.c
  jsmn/jsmn.c
)

set(HEADERS
 fcontext/fcontext.h
 jsmn/jsmn.h
)

FILE(GLOB INC_LIST inc/*.h)
list(APPEND HEADERS ${INC_LIST})

if (CMAKE_SYSTEM_NAME STREQUAL "Darwin")
  message("Compiling for Mac OS X (Darwin)")
  list(APPEND SOURCES
    fcontext/jump_x86_64_sysv_macho_gas.S
    fcontext/make_x86_64_sysv_macho_gas.S)
elseif (CMAKE_SYSTEM_NAME STREQUAL "Linux")
  message("Compiling for Linux")
  list(APPEND SOURCES
    fcontext/jump_x86_64_sysv_elf_gas.S
    fcontext/make_x86_64_sysv_elf_gas.S)
endif()

add_library(hclib SHARED ${SOURCES})

if(HCLIB_PRIVATIZE_C_STD_11)
  target_compile_features(hclib PRIVATE c_std_11)
else()
  target_compile_features(hclib PUBLIC c_std_11)
endif(HCLIB_PRIVATIZE_C_STD_11)

target_compile_features(hclib PUBLIC cxx_std_11)

#set(CMAKE_BUILD_TYPE "Release")
#set(CMAKE_BUILD_TYPE "Debug")
string(APPEND CMAKE_C_FLAGS " -g -O3 ")
string(APPEND CMAKE_CXX_FLAGS " -g -O3 ")

set(INCLUDE_DIRS
 ${CMAKE_SOURCE_DIR}/src/inc
 ${CMAKE_SOURCE_DIR}/inc
 ${CMAKE_SOURCE_DIR}/src/fcontext
 ${CMAKE_SOURCE_DIR}/src/jsmn
 ${CMAKE_BINARY_DIR}/inc
)

foreach(DIR ${INCLUDE_DIRS})
target_include_directories(hclib PUBLIC
  $<BUILD_INTERFACE:${DIR}>
  $<INSTALL_INTERFACE:include>
)
endforeach()

install(TARGETS hclib EXPORT hclib
        INCLUDES DESTINATION include
        ARCHIVE DESTINATION lib
        LIBRARY DESTINATION lib)

install(EXPORT hclib
        FILE hclibTargets.cmake
        NAMESPACE hclib::
        DESTINATION cmake)


target_compile_definitions(hclib PRIVATE ${COMPILE_DEFINITIONS})

if (HCLIB_ENABLE_HWLOC)
find_path(HWLOC_INCLUDE NAMES hwloc.h PATHS "$ENV{HWLOC_HOME}/include")
target_include_directories(hclib PUBLIC ${HWLOC_INCLUDE})
find_library(HWLOC_LIBRARY NAMES hwloc PATHS "$ENV{HWLOC_HOME}/lib" "$ENV{HWLOC_HOME}/lib64")
target_link_libraries(hclib PUBLIC ${HWLOC_LIBRARY})
endif(HCLIB_ENABLE_HWLOC)

if (CMAKE_SYSTEM_NAME STREQUAL "Linux")
  target_link_libraries(hclib PUBLIC pthread)
  target_link_libraries(hclib PUBLIC dl)
  target_link_libraries(hclib PUBLIC rt)
endif()

install(FILES ${HEADERS} DESTINATION include)



