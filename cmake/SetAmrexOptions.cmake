set(USE_XSDK_DEFAULTS OFF)
set(AMReX_SPACEDIM "${PELERAD_DIM}" CACHE STRING "Number of physical dimensions" FORCE)
set(AMReX_MPI ${PELERAD_ENABLE_MPI})
set(AMReX_OMP ${PELERAD_ENABLE_OPENMP})
set(AMReX_EB ${PELERAD_ENABLE_EB})
set(AMReX_SUNDIALS ${PELERAD_ENABLE_SUNDIALS})
set(AMReX_PARTICLES ${PELERAD_ENABLE_PARTICLES})
set(AMReX_CUDA ${PELERAD_ENABLE_CUDA})
set(AMReX_CUDA_ARCH ${PELERAD_CUDA_ARCH})
set(AMReX_DPCPP ${PELERAD_ENABLE_DPCPP})
set(AMReX_HIP ${PELERAD_ENABLE_HIP})
set(AMReX_PLOTFILE_TOOLS ${PELERAD_ENABLE_FCOMPARE})
set(AMReX_LINEAR_SOLVERS ${PELERAD_ENABLE_LINEARSOLVERS})
set(AMReX_HYPRE ${PELERAD_ENABLE_HYPRE})
set(AMReX_FORTRAN OFF)
set(AMReX_FORTRAN_INTERFACES OFF)
set(AMReX_PIC OFF)
set(AMReX_PRECISION DOUBLE)
set(AMReX_AMRDATA OFF)
set(AMReX_ASCENT OFF)
set(AMReX_SENSEI OFF)
set(AMReX_CONDUIT OFF)
set(AMReX_FPE OFF)
set(AMReX_ASSERTIONS OFF)
set(AMReX_BASE_PROFILE OFF)
set(AMReX_TINY_PROFILE OFF)
set(AMReX_TRACE_PROFILE OFF)
set(AMReX_MEM_PROFILE OFF)
set(AMReX_COMM_PROFILE OFF)
set(AMReX_BACKTRACE OFF)
set(AMReX_PROFPARSER OFF)
set(AMReX_ACC OFF)

if(PELERAD_ENABLE_CUDA)
  set(AMReX_GPU_BACKEND CUDA CACHE STRING "AMReX GPU type" FORCE)
  set(AMReX_ERROR_CROSS_EXECUTION_SPACE_CALL ON)
  set(AMReX_ERROR_CAPTURE_THIS ON)
endif()
