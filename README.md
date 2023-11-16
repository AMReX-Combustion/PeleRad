# PeleRad

[![CI](https://github.com/AMReX-Combustion/PeleRad/actions/workflows/linux.yml/badge.svg)](https://github.com/AMReX-Combustion/PeleRad)
[![AMReX Badge](https://img.shields.io/static/v1?label=%22powered%20by%22&message=%22AMReX%22&color=%22blue%22)](https://amrex-codes.github.io/amrex/)
[![Exascale Computing Project](https://img.shields.io/badge/supported%20by-ECP-blue)](https://www.exascaleproject.org/research-project/combustion-pele/)

## Table of Contents
- [Description](#description)
- [Installation](#installation)
- [Usage](#usage)
- [License](#license)

## Description
PeleRad is a module for modeling radiative transfer in reacting flows. It supports GPU performance portability and adaptive mesh refinement through AMReX and is currently coupled to PeleLMeX.

## Installation
1. Clone the repository: `git clone https://github.com/AMReX-Combustion/PeleRad.git`
2. The example CMake scripts for Summit CUDA build and Frontier HIP build are in the '/scripts' folder.
3. To enable to unit tests, PeleRad needs to build with the Boost library.

To build with the PeleLMeX flow solver, GNUMake is prefered.
One example script to build a test case on Frontier is provided in the '/scripts' folder.

To run the script:
'./build_PeleLMeX_Fontier.sh' 

## Usage
PeleRad is designed for modeling radiative transfer in reacting flows. To activate and use PeleRad in conjunction with PeleLMeX, follow these steps:
1. Add 'USE_PELERAD = TRUE' in the GNUmakefile in the case.
2. If using the clang compiler on Frontier, the c++ file system needs to be linked manually by adding 'LIBRARIES += -lstdc++fs'.
3. Add the pelerad input keywords to the input file in the case folder. Example keywords can be found in '/inputs/inputs.egLMeX'
4. The path of the spectral database ($PELERAD_HOME/data/kpDB/) is needed for the pelerad.kppath keywords.

## License
The license file is located at
https://github.com/AMReX-Combustion/PeleMP/blob/master/license.txt


