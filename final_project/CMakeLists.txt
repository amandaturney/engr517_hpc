# ------------------------------------------------------------------------------
# You need to make sure to run "module load cmake" on juliet to get 3.24.
# ------------------------------------------------------------------------------
cmake_minimum_required(VERSION 3.24)
project(SHORS CXX)

find_package(OpenMP REQUIRED)

include(FetchContent)
FetchContent_Declare(fmt
  GIT_REPOSITORY     https://github.com/fmtlib/fmt.git
  GIT_TAG            8.1.1)
FetchContent_MakeAvailable(fmt)

add_executable(shors shors.cpp)
target_link_libraries(shors PRIVATE fmt::fmt OpenMP::OpenMP_CXX)
