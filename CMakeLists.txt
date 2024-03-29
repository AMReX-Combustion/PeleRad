cmake_minimum_required(VERSION 3.14)
project(PeleRad CXX)

add_library(PeleRad INTERFACE)
set_target_properties(PeleRad PROPERTIES INTERFACE_COMPILE_FEATURES cxx_std_17)
add_dependencies(PeleRad record_hash)

add_library(PeleRad::PeleRad ALIAS PeleRad)

option(PELERAD_ENABLE_MPI "Enable MPI" OFF)
if(PELERAD_ENABLE_MPI)
  find_package(MPI REQUIRED)
  target_link_libraries(PeleRad INTERFACE MPI::MPI_CXX)
endif()


include(${CMAKE_SOURCE_DIR}/cmake/SetAmrexOptions.cmake)

add_subdirectory(${AMREX_HOME_DIR} amrex)

if(PELERAD_ENABLE_CUDA)
  set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} --expt-relaxed-constexpr --expt-extended-lambda -Wno-deprecated-gpu-targets ${NVCC_ARCH_FLAGS}")
  if(POLICY CMP0104)
      cmake_policy(SET CMP0104 OLD)
  endif()
  enable_language(CUDA)
endif()

if(PELERAD_ENABLE_HIP)
  add_definitions(-DPELERAD_USE_HIP=ON)
  set(CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} -lstdc++fs")
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

configure_file(${CMAKE_CURRENT_SOURCE_DIR}/src/PeleRad_Config.hpp.in 
               ${CMAKE_CURRENT_BINARY_DIR}/include/PeleRad_Config.hpp)

configure_file(${CMAKE_CURRENT_SOURCE_DIR}/cmake/PeleRadSettings.cmake.in 
               ${CMAKE_CURRENT_BINARY_DIR}/PeleRadSettings.cmake
              @ONLY)

include(CMakePackageConfigHelpers)
configure_package_config_file(cmake/PeleRadConfig.cmake.in
  ${CMAKE_CURRENT_BINARY_DIR}/PeleRadConfig.cmake
  INSTALL_DESTINATION lib/cmake/PeleRad
)

install(FILES
  ${CMAKE_CURRENT_BINARY_DIR}/PeleRadConfig.cmake
  ${CMAKE_CURRENT_BINARY_DIR}/PeleRadSettings.cmake
  DESTINATION lib/cmake/PeleRad
)

install(DIRECTORY ${PROJECT_SOURCE_DIR}/src/ DESTINATION include
        FILES_MATCHING PATTERN "*.hpp")
install(DIRECTORY ${PROJECT_BINARY_DIR}/include/ DESTINATION include
        FILES_MATCHING PATTERN "*.hpp")

if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE "Debug" CACHE STRING
    "None was specified to CMAKE_BUILD_TYPE, setting build type to Debug."
    FORCE)
endif()

option(PELERAD_ENABLE_TESTS "Enable tests" OFF)

if(${PELERAD_ENABLE_TESTS})
  find_package(Boost 1.74.0 REQUIRED COMPONENTS program_options unit_test_framework)
  enable_testing()
  add_subdirectory(tests)
endif()


