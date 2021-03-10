cmake_minimum_required(VERSION 3.14)
project(PeleRad CXX)

add_library(PeleRad INTERFACE)
set_target_properties(PeleRad PROPERTIES INTERFACE_COMPILE_FEATURES cxx_std_14)
add_dependencies(PeleRad record_hash)

add_library(PeleRad::PeleRad ALIAS PeleRad)

option(PELERAD_ENABLE_MPI "Enable MPI" OFF)

if(PELERAD_ENABLE_MPI)
  find_package(MPI REQUIRED)
  target_link_libraries(PeleRad INTERFACE MPI::MPI_CXX)
endif()

find_package(Boost 1.67.0 REQUIRED COMPONENTS filesystem system iostreams program_options unit_test_framework)

option(PELERAD_ENABLE_CUDA "CUDA build" OFF)
if(PELERAD_GPU_BACKEND="CUDA")
  enable_language(CUDA)
  set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} --expt-relaxed-constexpr --expt-extended-lambda -Wno-deprecated-gpu-targets ${NVCC_ARCH_FLAGS}")
  if(POLICY CMP0104)
      cmake_policy(SET CMP0104 OLD)
  endif()
  include(${AMREX_HOME_DIR}/Tools/CMake/AMReX_SetupCUDA.cmake)
endif()

target_include_directories(PeleRad INTERFACE
  $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/src>
  $<BUILD_INTERFACE:${CMAKE_CURRENT_BINARY_DIR}/include>
  $<INSTALL_INTERFACE:include>
)

install(TARGETS PeleRad
  EXPORT PeleRadTargets
  ARCHIVE LIBRARY PUBLIC_HEADER
)

install(EXPORT PeleRadTargets
  NAMESPACE PeleRad::
  DESTINATION lib/cmake/PeleRad
)

if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE "Debug" CACHE STRING
    "None was specified to CMAKE_BUILD_TYPE, setting build type to Debug."
    FORCE)
endif()

option(PELERAD_ENABLE_TESTS "Enalbe tests" OFF)

if(${PELERAD_ENABLE_TESTS})
  enable_testing()
  add_subdirectory(tests)
endif()

include(${CMAKE_SOURCE_DIR}/cmake/SetAmrexOptions.cmake)
add_subdirectory(${AMREX_HOME_DIR} amrex)
