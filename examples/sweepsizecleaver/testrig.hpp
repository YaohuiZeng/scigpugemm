// Runs a simple size sweep of the cleaver


/*
  Copyright

  (c) Copyright 2009 President and Fellows of Harvard College

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
*/

#include <cstdlib>
#include <cstring>
#include <iostream>
#include <iomanip>
#include <typeinfo>


#include "chronometer.hpp"
#include "gemm_gold.hpp"
#include "matrixutils.hpp"

#include "gemmcleaver.hpp"


#ifndef TESTRIG_H
#define TESTRIG_H


// ==============================================================================

template<typename T> void WriteStream( std::ostream &os, T data ) {
  // I can't work out how to set the field width of a stream for all writes, so...
  os << std::setw(15) << data;
}



// -------------------

template<typename T> void RunCleaver( const char transA, const char transB,
				      const int m, const int n, const int k,
				      const T alpha,
				      const T *A, const int lda,
				      const T *B, const int ldb,
				      const T beta,
				      T *C, const int ldc,
				      SciGPUgemm::GEMMcleaver &theCleaver ) {
  // Catch unimplemented types
  std::cerr << __PRETTY_FUNCTION__ << ": Unimplemented cleaver call for type ";
  std::cerr << typeid(alpha).name() << std::endl;
  exit( EXIT_FAILURE );
}


template<> void RunCleaver<double>( const char transA, const char transB,
				    const int m, const int n, const int k,
				    const double alpha,
				    const double *A, const int lda,
				    const double *B, const int ldb,
				    const double beta,
				    double *C, const int ldc,
				    SciGPUgemm::GEMMcleaver &theCleaver );

template<> void RunCleaver<float>( const char transA, const char transB,
				   const int m, const int n, const int k,
				   const float alpha,
				   const float *A, const int lda,
				   const float *B, const int ldb,
				   const float beta,
				   float *C, const int ldc,
				   SciGPUgemm::GEMMcleaver &theCleaver );



// ---------------

template<typename T> void RunOneTest( const unsigned int randSeed,
				      const size_t memAllowed,
				      const char transA, const char transB,
				      const int matrixSize,
				      const T min, const T max,
				      const T alpha, const T beta,
				      const unsigned int nRepeats,
				      std::ostream &os ) {
  // Routine to run a single comparision of the cleaver

  T *A, *B, *C;
  T *resBLAS, *resCleaver;

  int m, n, k, lda, ldb, ldc;
  int hA, wA, hB, wB, hC, wC;

  SciGPUgemm::GEMMcleaver testCleaver( memAllowed );
  

  Chronometer t_cpu, t_cleaver;

  // Initialise PRNG
  srand( randSeed );

  // Set the matrix sizes
  m = n = k = matrixSize;
  lda = wA = hA = ldb = wB = hB = ldc = wC = hC = matrixSize;

  // Allocate memory
  A = new T[lda*wA];
  B = new T[ldb*wB];
  C = new T[ldc*wC];

  resBLAS = new T[ldc*wC];
  resCleaver = new T[ldc*wC];

  // Fill with data
  RandomMatrix( A, lda, hA, wA, min, max );
  RandomMatrix( B, ldb, hB, wB, min, max );
  RandomMatrix( C, ldc, hC, wC, min, max );

  // Average over designated number of runs
  for( unsigned int iRepeat=0; iRepeat<nRepeats; iRepeat++ ) {
    memcpy( resBLAS, C, ldc*wC*sizeof(T) );
    memcpy( resCleaver, C, ldc*wC*sizeof(T) );

    // Run the CPU multiplication
    t_cpu.Start();
    gemm_gold( A, B, resBLAS,
	       alpha, beta,
	       transA, transB,
	       m, n, k, lda, ldb, ldc );
    t_cpu.Stop();

    // Run the GPU multiplication
    t_cleaver.Start();
    RunCleaver( transA, transB,
		m, n, k,
		alpha,
		A, lda, B, ldb,
		beta,
		resCleaver, ldc,
		testCleaver );
    t_cleaver.Stop();
  }

  // Output the results
  WriteStream( os, matrixSize );
  WriteStream( os, t_cpu.GetAverageTime() );
  WriteStream( os, t_cleaver.GetAverageTime() );
  os << std::endl;



  // Release memory
  delete[] A;
  delete[] B;
  delete[] C;
  delete[] resBLAS;
  delete[] resCleaver;
}


void WriteHeader( std::ostream &os );


#endif
