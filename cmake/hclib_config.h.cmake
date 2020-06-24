
include(CheckIncludeFiles)
include(CheckCXXSymbolExists)
include(CheckCXXSourceCompiles)

check_include_files(aio.h HAVE_AIO_H)
#check_cxx_symbol_exists(std::is_trivially_copyable<int> "type_traits" HAVE_CXX11_TRIVIAL_COPY_CHECK)
check_cxx_source_compiles(
    "#include<type_traits>
    int main() {
        return std::is_trivially_copyable<int>::value;
    }" HAVE_CXX11_TRIVIAL_COPY_CHECK )
check_include_files(dlfcn.h HAVE_DLFCN_H)
check_include_files(inttypes.h HAVE_INTTYPES_H)
check_include_files(memory.h HAVE_MEMORY_H)
check_include_files(stdint.h HAVE_STDINT_H)
check_include_files(stdlib.h HAVE_STDLIB_H)
check_include_files(strings.h HAVE_STRINGS_H)
check_include_files(string.h HAVE_STRING_H)
check_include_files(sys/mman.h HAVE_SYS_MMAN_H)
check_include_files(sys/stat.h HAVE_SYS_STAT_H)
check_include_files(sys/types.h HAVE_SYS_TYPES_H)
check_include_files(unistd.h HAVE_UNISTD_H)
check_include_files("stdlib.h;stdarg.h;string.h;float.h" STDC_HEADERS)

set(LT_OBJDIR ".libs/")

set(PACKAGE "hclib")
set(VERSION "0.1")
set(PACKAGE_BUGREPORT "jmg3@rice.edu")
set(PACKAGE_NAME "${PACKAGE}")
set(PACKAGE_VERSION "${VERSION}")
set(PACKAGE_STRING "${PACKAGE_NAME} ${PACKAGE_VERSION}")
set(PACKAGE_TARNAME "${PACKAGE_NAME}")
set(PACKAGE_URL "https://habanero-rice.github.io/hclib")

configure_file(
 ${CMAKE_CURRENT_SOURCE_DIR}/inc/hclib_config.h.cmake.in
 ${CMAKE_CURRENT_BINARY_DIR}/inc/hclib_config.h @ONLY
)

