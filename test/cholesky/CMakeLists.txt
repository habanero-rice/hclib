
cmake_minimum_required (VERSION 3.9)
project (HCLIB_Cholesky)

add_executable(cholesky
  cholesky.cpp 
  sequential_cholesky.cpp 
  trisolve.cpp 
  update_diagonal.cpp 
  update_nondiagonal.cpp
)

find_package(hclib REQUIRED)

target_link_libraries(cholesky hclib::hclib)

