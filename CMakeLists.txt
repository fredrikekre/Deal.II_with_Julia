cmake_minimum_required(VERSION 2.8.12)

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++17")

# rpaths
set(CMAKE_MACOSX_RPATH 1)
set(CMAKE_SKIP_BUILD_RPATH FALSE)
set(CMAKE_BUILD_WITH_INSTALL_RPATH TRUE)
set(CMAKE_INSTALL_RPATH_USE_LINK_PATH TRUE)

set(CMAKE_MODULE_PATH ${CMAKE_CURRENT_SOURCE_DIR}/cmake)
find_package(Julia REQUIRED)

# Path to the Julia source file
add_compile_options(-DJULIA_SOURCE_FILE="${CMAKE_CURRENT_SOURCE_DIR}/src/hyperelasticity.jl")

# Target name and source files
set(TARGET "hyperelasticity")
set(TARGET_SRC src/${TARGET}.cc)

find_package(deal.II 9.5.0
  HINTS ${deal.II_DIR} ${DEAL_II_DIR} ../ ../../ $ENV{DEAL_II_DIR}
  )
if(NOT ${deal.II_FOUND})
  message(FATAL_ERROR "\n"
    "*** Could not locate a (sufficiently recent) version of deal.II. ***\n\n"
    "You may want to either pass a flag -DDEAL_II_DIR=/path/to/deal.II to cmake\n"
    "or set an environment variable \"DEAL_II_DIR\" that contains this path."
    )
endif()

deal_ii_initialize_cached_variables()
project(${TARGET})

deal_ii_invoke_autopilot()

target_include_directories(${TARGET} PUBLIC
  "$<BUILD_INTERFACE:${Julia_INCLUDE_DIRS}>"
)
target_link_libraries(${TARGET} $<BUILD_INTERFACE:${Julia_LIBRARY}>)
