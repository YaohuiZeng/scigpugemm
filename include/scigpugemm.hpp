/*! \file
  General header file for SciGPU-GEMM library.
  Include this file to gain access to all of the SciGPU-GEMM
  classes

  \section Copyright

  This file is part of the SciGPU-GEMM Library

  SciGPU-GEMM is free software: you can redistribute it and/or modify
  it under the terms of the GNU Lesser General Public License as published by
  the Free Software Foundation, either version 3 of the License, or
  (at your option) any later version.

  SciGPU-GEMM is distributed in the hope that it will be useful,
  but WITHOUT ANY WARRANTY; without even the implied warranty of
  MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
  GNU Lesser General Public License for more details.

  You should have received a copy of the GNU Lesser General Public License
  along with SciGPU-GEMM.  If not, see <http://www.gnu.org/licenses/>.

  &copy; Copyright 2009 President and Fellows of Harvard College
*/

//! Namespace to hold all of the library routines
/*!
  The SciGPUgemm namespace holds all of the routines for the
  SciGPU-GEMM library.
  It is used to prevent namespace collisions with other packages.
*/
namespace SciGPUgemm {


/*! \mainpage Documentation for SciGPU-GEMM library

  The SciGPU-GEMM library enables easier use of GPU acceleration
  for matrix-matrix multiplication.
  Although \c SGEMM and \c DGEMM are provided in the CUBLAS
  library, there are two key limitations in using them
  - Most GPUs have somewhat limited onboard memory
  - Few GPUs have double precision hardware

  Although these are diminishing problems (and non-existent
  for new scientific clusters), they will often confront
  researchers attempting to run code on older hardware.
  In particular, the Aspuru group a Harvard University uses
  the BOINC project to run parameter space sweeps on thousands
  of machines worldwide.
  A large fraction of these machines will have CUDA-capable
  GPUs and the Aspuru group wishes to put these cards to
  use.
  Most of the Aspuru group's computations centre on
  matrix-matrix multiplications, which are ideally suited
  to GPU acceleration.
  However, their calculations often involve large matrices,
  and they often require more than single precision.
  This library was written to solve their problem.



  \section libcontents Library Contents

  The SciGPU-GEMM library contains three key routines
  for the end user, distributed into two classes,
  GEMMcleaver and MGEMM.
  They provide matrix-matrix multiplication functionality
  for <strong>column-major</strong> matrices.
  

  The GEMMcleaver class addresses the first problem
  noted above: the lack on memory available on
  many GPUs.
  When requested to perform a matrix-matrix multiplication,
  GEMMcleaver divides the multiplication into a series of
  sub-multiplications, each of which will fit into the
  memory available on the current GPU.
  This is described in detail in the section \ref cleaving.
  The GEMMcleaver class offers public methods,
  which implement \c SGEMM, \c DGEMM, \c CGEMM and \c ZGEMM.
  Usage is straightforward:
  \code
  #include "scigpugemm.hpp"
  using namespace SciGPUgemm;

  void mymultiplyfloat( ... ) {
    float *myA, *myB, *myC;
    GEMMcleaver myCleaver;
    
    ....
    myCleaver.sgemm( transposeA, transposeB,
                     m, n, k,
		     alpha, myA, lda, myB, ldb,
		     beta, myC, ldc );
  }

  void mymultiplydouble( ... ) {
    double *myA, *myB, *myC;
    GEMMcleaver myCleaver;
    
    ...
    myCleaver.dgemm( transposeA, transposeB,
                     m, n, k,
		     alpha, myA, lda, myB, ldb,
		     beta, myC, ldc );
  }
  \endcode
  The arguments to each method are identical to those
  in a standard \c *GEMM call.
  Note that everything is contained within the
  SciGPUgemm namespace.
  One important step not shown here is that <strong>a CUDA context
  must exist</strong> prior to any call into the SciGPU-GEMM library.
  This can be achieved with a single \c cudaMalloc (and \c cudaFree )
  call prior to calling into SciGPU-GEMM, if you have no other CUDA
  code.
  The library also contains C wrapper functions, available through
  the file scigpugemm_wrappers.h.


  The MGEMM class attempts to circumvent the lack of double
  precision hardware on most contemporary GPUs.
  It splits the matrices into `small' and `large' portions,
  and splits the calculation between the CPU and GPU.
  For more details, see the section \ref splitting.
  The key routine for users is MGEMM::mgemm.
  To user code, there is only a slight difference from a standard \c GEMM
  call:
  \code
  #include "scigpugemm.hpp"
  using namespace SciGPUgemm;

  void mymultiplymulti( ... ) {
    double *myA, *myB, *myC;
    const double myCutoff = 2;
    MGEMM myMultiply;

    ...

    myMultiply.mgemm( transposeA, transposeB,
                      m, n, k,
		      alpha, myA, lda, myB, ldb,
		      beta, myC, ldc,
		      myCutoff );
  }
  \endcode
  The arguments are the same, except for the additional of the \a myCutoff
  value.
  This determines how the matrices are split into `small' and `large'
  portions.
  Matrix elements with absolute values larger than \a myCutoff will be
  considered `large.'
  MGEMM uses the GEMMcleaver class to handle the GPU calculations,
  enabling MGEMM to function efficiently on devices with limited memory.
  The C wrapper function is mgemm(), also in the file scigpugemm_wrappers.h.
  
  
  The constructors for GEMMcleaver and MGEMM can both take an argument of
  type \c size_t, which is then used to limit the amount of memory
  used on the GPU.
  This value is <em>not</em> checked against the actual amount of 
  memory available on the GPU.
  By default, GEMMcleaver and MGEMM will use all available GPU memory.
  
  
  The library includes several classes beyond those described here
  (including a small amount of inheritance).
  Indeed, one could argue that an imperative approach would be simpler
  and better fit the likely uses.


  \section Installation

  The simplest way to get started is to run \c make in the \c scigpugemm/
  directory.
  This will build the library, the documentation, and the examples.
  The library is placed
  in the \c lib/ directory.
  The example binaries are compiled into the \c bin/ directory.
  
  The supplied \c Makefile assumes that the <tt>CPATH</tt> environment
  variable is set and includes the <tt>include/</tt> directory of
  the CUDA installation.
  You will need to ensure that your <tt>LIBRARY_PATH</tt> and
  <tt>LD_LIBRARY_PATH</tt> environment variables point to the
  <tt>lib/</tt> directory of your CUDA installation.
  The example programs also rely on CBLAS/ATLAS; the above environment
  variables will need to be set appropriately for these.
  
  \section Copying

  SciGPU-GEMM is free software: you can redistribute it and/or modify
  it under the terms of the GNU Lesser General Public License as published by
  the Free Software Foundation, either version 3 of the License, or
  (at your option) any later version.

  SciGPU-GEMM is distributed in the hope that it will be useful,
  but WITHOUT ANY WARRANTY; without even the implied warranty of
  MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
  GNU Lesser General Public License for more details.

  You should have received a copy of the GNU Lesser General Public License
  along with SciGPU-GEMM.  If not, see <http://www.gnu.org/licenses/>.

  \section Acknowledgments

  This library was based on suggestions by Alan Aspuru.
  Early implementations of some of the algorithms were written
  by Roberto Olivares-Amaya.
  Some hardware for this work was donated by NVIDIA, as part of the
  Harvard CUDA Center of Excellence

  \author Richard G. Edgar richard_edgar@harvard.edu
*/

};

#ifndef SciGPU_GEMM_H
#define SciGPU_GEMM_H

#include "gpuerror.hpp"
#include "densematrix.hpp"
#include "gemmcleaver.hpp"
#include "mgemm.hpp"


#endif
