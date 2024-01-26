# PeleRad

[![CI](https://github.com/AMReX-Combustion/PeleRad/actions/workflows/linux.yml/badge.svg)](https://github.com/AMReX-Combustion/PeleRad)
[![AMReX Badge](https://img.shields.io/static/v1?label=%22powered%20by%22&message=%22AMReX%22&color=%22blue%22)](https://amrex-codes.github.io/amrex/)
[![Exascale Computing Project](https://img.shields.io/badge/supported%20by-ECP-blue)](https://www.exascaleproject.org/research-project/combustion-pele/)

# WARNING

The production version of PeleRad have been moved to [PelePhysics](https://github.com/AMReX-Combustion/PelePhysics) and this repository is now only for experimental developments.
Further development is continuing within PelePhysics. The tests are in PelePhysics, while the example cases are part of PeleLMeX.

## Table of Contents
- [Description](#description)
- [Installation](#installation)
- [Usage](#usage)
- [License](#license)
- [Citation](#citation)

## Description
PeleRad was a module for modeling radiative transfer in reacting flows. It supports GPU performance portability and adaptive mesh refinement through AMReX and is coupled to PeleLMeX.

## Installation
1. Clone the repository: `git clone https://github.com/AMReX-Combustion/PeleRad.git`
2. The example CMake scripts for Summit CUDA build and Frontier HIP build are in the '/scripts' folder.
3. To enable the unit tests, PeleRad needs to build with the Boost library.

To build with the PeleLMeX flow solver, GNUMake is prefered.
One example script to build a test case on Frontier is provided in the '/scripts' folder.

To run the script:
'./build_PeleLMeX_Fontier.sh'

## Usage
PeleRad is designed for modeling radiative transfer in reacting flows. To activate and use PeleRad in conjunction with PeleLMeX, follow these steps:
1. Add 'USE_PELERAD = TRUE' in the GNUmakefile in the case folder.
2. If using the clang compiler on Frontier, link the c++ file system by adding 'LIBRARIES += -lstdc++fs'.
3. Add the pelerad input keywords to the amrex-style input file in the case folder. Example keywords can be found in '/inputs/inputs.egLMeX'
4. Specify the path of the spectral database ($PELERAD_HOME/data/kpDB/) is needed for the 'pelerad.kppath' keywords in the input file.

## License
The license file is located at
https://github.com/AMReX-Combustion/PeleMP/blob/master/license.txt

## Citation
To cite PeleMP, please use the following [Journal of Fluids Engineering article](https://doi.org/10.1115/1.4064494):
```
@article{owen2023pelemp,
  title={PeleMP: The Multiphysics Solver for the Combustion Pele Adaptive Mesh Refinement Code Suite},
  author={Owen, Landon D and Ge, Wenjun and Rieth, Martin and Arienti, Marco and Esclapez, Lucas and S Soriano, Bruno and Mueller, Michael E and Day, Marc and Sankaran, Ramanan and Chen, Jacqueline H},
  journal={Journal of Fluids Engineering},
  pages={1--41},
  year={2023}
}
```
