#!/bin/bash -l

MYPWD=${PWD}

module load PrgEnv-cray cmake rocm/5.2.0 craype-x86-trento craype-accel-amd-gfx90a cray-libsci/21.08.1.2

# git clone --recursive --branch development https://github.com/AMReX-Combustion/PeleLMeX.git
git clone --recursive --branch pr_rad https://github.com/wjge/PeleLMeX.git
git clone --branch main https://github.com/AMReX-Combustion/PeleRad.git
git clone --branch eb_robin https://github.com/WeiqunZhang/amrex.git
git clone --branch pr_rad https://github.com/wjge/PeleMP.git

export PELELMEX_HOME=${MYPWD}/PeleLMeX
export AMREX_HOME=${MYPWD}/amrex
export PELE_PHYSICS_HOME=${MYPWD}/PeleLMeX/Submodules/PelePhysics
export PELEMP_HOME=${MYPWD}/PeleMP
export AMREX_HYDRO_HOME=${MYPWD}/PeleLMeX/Submodules/AMReX-Hydro
export PELERAD_HOME=${MYPWD}/PeleRad

cd ${PELEMP_HOME}/Exec/SootTests/PeleLMeX/laminar_flame

# Build LMeX laminar_flame case
#Build SUNDIALS and MAGMA
make -j8 COMP=clang USE_HIP=TRUE USE_MPI=TRUE TINY_PROFILE=TRUE PELE_USE_MAGMA=TRUE HOSTNAME=frontier TPLrealclean
make -j8 COMP=clang USE_HIP=TRUE USE_MPI=TRUE TINY_PROFILE=TRUE PELE_USE_MAGMA=TRUE HOSTNAME=frontier TPL
   
#Build PeleLMeX
make -j8 COMP=clang USE_HIP=TRUE USE_MPI=TRUE TINY_PROFILE=TRUE PELE_USE_MAGMA=TRUE HOSTNAME=frontier realclean
make -j8 COMP=clang USE_HIP=TRUE USE_MPI=TRUE TINY_PROFILE=TRUE PELE_USE_MAGMA=TRUE HOSTNAME=frontier

echo "pelerad.kppath = "$PELERAD_HOME/data/kpDB/"" >> first-input
./PeleLMeX2d.hip.x86-trento.TPROF.MPI.HIP.ex first-input
