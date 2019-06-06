SET(MODULE_NAME "system")

FILE(GLOB SOURCES "${CMAKE_CURRENT_LIST_DIR}/src/*.cpp" "${CMAKE_CURRENT_LIST_DIR}/src/*.c")
FILE(GLOB HEADERS "${CMAKE_CURRENT_LIST_DIR}/inc/*.h")

ADD_LIBRARY("hclib_${MODULE_NAME}" MODULE "${SOURCES}")

target_link_libraries("hclib_${MODULE_NAME}" PRIVATE hclib)

target_compile_features("hclib_${MODULE_NAME}" PUBLIC c_std_11)
target_compile_features("hclib_${MODULE_NAME}" PUBLIC cxx_std_11)

target_include_directories("hclib_${MODULE_NAME}"
    PUBLIC
    $<BUILD_INTERFACE:${CMAKE_CURRENT_LIST_DIR}/inc>
    $<INSTALL_INTERFACE:include>)

target_compile_definitions(hclib PRIVATE COMPILE_DEFINITIONS)

install(TARGETS "hclib_${MODULE_NAME}" EXPORT "hclib_${MODULE_NAME}-targets"
        INCLUDES DESTINATION include
        ARCHIVE DESTINATION lib
        LIBRARY DESTINATION lib)

install(EXPORT "hclib_${MODULE_NAME}-targets"
    FILE "hclib_${MODULE_NAME}Targets.cmake"
    NAMESPACE hclib::
    DESTINATION cmake)

install(FILES ${HEADERS} DESTINATION include)
