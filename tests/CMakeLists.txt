find_package(Boost 1.67.0 REQUIRED COMPONENTS unit_test_framework)

include(${AMREX_HOME_DIR}/Tools/CMake/AMReXTargetHelpers.cmake)

if(PELERAD_ENABLE_MPI)
add_executable(PeleRad_AmrexParallelFor.exe tstAmrexParallelFor.cpp ut_main.cpp)
target_include_directories(PeleRad_AmrexParallelFor.exe PUBLIC ${AMREX_HOME_DIR}/Src/Base)
target_link_libraries(PeleRad_AmrexParallelFor.exe PRIVATE PeleRad Boost::unit_test_framework amrex MPI::MPI_CXX)
target_compile_definitions(PeleRad_AmrexParallelFor.exe PRIVATE BOOST_TEST_DYN_LINK PELERAD_MPI_UNIT_TEST)
add_test(NAME PeleRad_AmrexParallelFor_Test COMMAND ${MPIEXEC_EXECUTABLE} ${MPIEXEC_NUMPROC_FLAG} ${MPIEXEC_MAX_NUMPROCS} ${MPIEXEC_PREFLAGS} ./PeleRad_AmrexParallelFor.exe ${MPIEXEC_POSTFLAGS})
else()
add_executable(PeleRad_AmrexParallelFor.exe tstAmrexParallelFor.cpp ut_main.cpp)
target_include_directories(PeleRad_AmrexParallelFor.exe PUBLIC ${AMREX_HOME_DIR}/Src/Base)
target_link_libraries(PeleRad_AmrexParallelFor.exe PRIVATE PeleRad Boost::unit_test_framework amrex)
target_compile_definitions(PeleRad_AmrexParallelFor.exe PRIVATE BOOST_TEST_DYN_LINK)
add_test(NAME PeleRad_AmrexParallelFor_Test COMMAND ./PeleRad_AmrexParallelFor.exe)
endif()

if (PELERAD_ENABLE_CUDA)
  setup_target_for_cuda_compilation(PeleRad_AmrexParallelFor.exe)
endif ()


