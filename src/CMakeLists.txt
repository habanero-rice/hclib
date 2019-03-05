
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
  fcontext/jump_ppc64_sysv_macho_gas.S 
  fcontext/make_ppc64_sysv_macho_gas.S)
endif()

add_library(hclib ${SOURCES})

target_compile_features(hclib PUBLIC cxx_std_11)

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


if (HCLIB_ENABLE_HWLOC)
target_compile_definitions(hclib USE_HWLOC)
target_include_directories(hclib ${HWLOC_HOME}/include)
target_link_directories(hclib PUBLIC ${HWLOC_HOME}/lib)
target_link_libraries(hclib PUBLIC hwloc)
endif(HCLIB_ENABLE_HWLOC)

if (HCLIB_ENABLE_STATS)
target_compile_definitions(hclib HCLIB_STATS)
endif(HCLIB_ENABLE_STATS)

if (HCLIB_ENABLE_VERBOSE)
target_compile_definitions(hclib VERBOSE)
endif(HCLIB_ENABLE_VERBOSE)

if (HCLIB_ENABLE_PRODUCTION)
target_compile_definitions(hclib HC_ASSERTION_CHECK)
endif()

install(FILES ${HEADERS} DESTINATION include)



