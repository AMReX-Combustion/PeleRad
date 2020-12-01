include(${AMREX_HOME_DIR}/Tools/CMake/AMReXTargetHelpers.cmake)

add_executable(PeleRad_BoostPropertyTree.exe tstBoostPropertyTree.cpp ut_main.cpp)
target_include_directories(PeleRad_BoostPropertyTree.exe PUBLIC ${AMREX_HOME_DIR}/Src/Base)
target_link_libraries(PeleRad_BoostPropertyTree.exe PRIVATE PeleRad Boost::unit_test_framework amrex)
target_compile_definitions(PeleRad_BoostPropertyTree.exe PRIVATE BOOST_TEST_DYN_LINK)
add_test(NAME PeleRad_BoostPropertyTree_Test COMMAND ./PeleRad_BoostPropertyTree.exe)

add_executable(PeleRad_BoostReadDatabase.exe tstBoostReadDatabase.cpp ut_main.cpp)
target_include_directories(PeleRad_BoostReadDatabase.exe PUBLIC ${AMREX_HOME_DIR}/Src/Base)
target_link_libraries(PeleRad_BoostReadDatabase.exe PRIVATE PeleRad Boost::unit_test_framework amrex ${Boost_FILESYSTEM_LIBRARY} ${Boost_SYSTEM_LIBRARY} ${Boost_IOSTREAMS_LIBRARY})
target_compile_definitions(PeleRad_BoostReadDatabase.exe PRIVATE BOOST_TEST_DYN_LINK)
set_property(TARGET PeleRad_BoostReadDatabase.exe APPEND PROPERTY COMPILE_DEFINITION DATABASE_PATH=${CMAKE_SOURCE_DIR}/data/kpDB/)
add_test(NAME PeleRad_BoostReadDatabase_Test COMMAND ./PeleRad_BoostReadDatabase.exe)

if(PELERAD_ENABLE_MPI)
add_executable(PeleRad_AmrexParallelFor.exe tstAmrexParallelFor.cpp ut_main.cpp)
target_include_directories(PeleRad_AmrexParallelFor.exe PUBLIC ${AMREX_HOME_DIR}/Src/Base)
target_link_libraries(PeleRad_AmrexParallelFor.exe PRIVATE PeleRad Boost::unit_test_framework amrex MPI::MPI_CXX)
target_compile_definitions(PeleRad_AmrexParallelFor.exe PRIVATE BOOST_TEST_DYN_LINK)
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


