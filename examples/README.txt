Examples for SciGPU-GEMM Library
================================

This directory contains a number of examples of teh SciGPU-GEMM library in use:

simplecleaver    - A small program to test GEMMcleaver
simplemgemm      - A small program to test MGEMM
sweepsaltmgemm   - A program to benchmark MGEMM, testing matrices salted with progressively more 'large' values
sweepsizemgemm   - A program to benchmark MGEMM, varying the size of the matrices
sweepcutoffmgemm - A program to benchmark MGEMM on 'real' matrices

Note:
The binaries are built in
../bin/


Bugs:
These programs link against cutil, so the Makefiles may have to be edited to point to the appropriate place. At some point, this dependency will be removed.

The Makefiles are somewhat ugly, and heavily based on the common.mk supplied by NVIDIA in their SDK

(c) Copyright 2009 President and Fellows of Harvard College
