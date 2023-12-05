# GBTCurvedPipeLinear
FEM implementation of GBT for analysis of thin-walled curved pipe members 


GBTCurvedPipeLinear is a Python code for stress and deformation analysis of curved thin-walled circular pipes based on the Generalized Beam Theory (GBT). This code is based on the paper: Generalized Beam Theory formulation for thin-walled pipes with circular axis, https://doi.org/10.1016/j.tws.2020.107243



## Getting started

We recomend to use Anaconda/Spyder editor and library manager 


## Dependencies
 * Python 3.7
 * Libraries: Numpy, Scipy, matplotlib, panda, sympy and math

## Code structure
 Ansys_shell_FEM_results_to_compare\
 * contains Ansys results for comparison 
 

 Python
 * GBT_classes_list.py-> include all GBT classes, solver and postprocessing
 * example_under_pressure_GBT_linear_curve.py ->	preprocessing, example  


## Numerical Example
The numerical example is developed to validate and illustrate the application and capabilities of the linear GBT formulation and its numerical implementation. Here, a short cantilever pipe is considered as a numerical example with the physical properties and boundary conditions shown below.

![example](https://github.com/AbinetKH/GBTStraightPipeLinear/blob/master/doc/example.png)



### The generalized modal amplitude vector
![The generalized modal amplitude vector](https://github.com/AbinetKH/GBTStraightPipeLinear/blob/master/doc/dispVector.png)

### Force vector
![Force vector](https://github.com/AbinetKH/GBTStraightPipeLinear/blob/master/doc/externalForceVector.png)

### Element stiffness matrix
![Element stiffness matrix](https://github.com/AbinetKH/GBTStraightPipeLinear/blob/master/doc/stiffnessmatrix.png)

### Deformation shape of a short cantilever pipe
![Deformation shape of a short cantilever pipe](https://github.com/AbinetKH/GBTStraightPipeLinear/blob/master/doc/plot.png)

Citation
--------

If you use any of the GBTStraightPipeLinear packages to produce scientific articles, please cite: https://e-pub.uni-weimar.de/opus4/frontdoor/index/index/docId/4572
