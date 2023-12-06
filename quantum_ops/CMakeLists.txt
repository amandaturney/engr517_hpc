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

FetchContent_Declare(docopt
  GIT_REPOSITORY     https://github.com/docopt/docopt.cpp.git
  GIT_TAG            master)
FetchContent_MakeAvailable(docopt)

add_executable(quantum_ops quantum_ops.cpp)
target_link_libraries(quantum_ops PRIVATE fmt::fmt OpenMP::OpenMP_CXX docopt)
