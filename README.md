# Consistent_MPFA_discretization_for_flow_in_the_presence_of_gravity
https://zenodo.org/badge/DOI/10.5281/zenodo.3413545.svg

This repository contains runscripts for the paper

"Consistent MPFA discretization for flow in the presence of gravity" by Michele Starnoni, Inga Berre, Eirik Keilegavlen, Jan Martin Nordbotten.

To run the examples contained in this repository a working version of PorePy v 1.1.0 (which can be downloaded from https://zenodo.org/record/3404634#.XX87YHUzbmF) must be installed on the computer. 

The PorePy install requires installations of further packages, see Install instructions in the PorePy repository.

This repository contains the following files:

piecewise_gravity_test: run script for Example 4.2 for grids  (a) and (b)

piecewise_uniform_flow: run script for Example 4.2 for grid (c)

upwind.py: run script for Example 3.3.2 and 5.2 using a standard upwind scheme

non_equilibrium_gcmpfa.py: run script for Example 5.2 using the GCMPFA method
