// Header file for the test rig

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

#ifndef TESTRIG_H
#define TESTRIG_H

#include <complex>
#include <cstdlib>
#include <cstring>

#include "matrixutils.hpp"
#include "gemm_gold.hpp"

#include "gemmcleaver.hpp"

// ==============================================================================

void SetupMatrices( const char transA, const char transB,
		    const int m, const int n, const int k,
		    const int lda, const int ldb, const int ldc,
		    int &hA, int &wA, int &hB, int &wB,
		    int &hC, int &wC );



// =============================================================================

template<typename T> T mynorm( const T& x ) {
  return( x*x );
}

template<typename T> T mynorm( const complex<T>& x ) {
  return( std::norm(x) );
}


template<typename T>
void RunOneTest( const unsigned int randSeed,
		 const size_t memAllowed,
		 const char transA, const char transB,
		 const int m, const int n, const int k,
		 const int lda, const int ldb, const int ldc,
		 const T min, const T max,
		 const T alpha, const T beta ) {
  
  T *A, *B, *C;
  T *resBLAS, *resCleaver;

  
  int hA, wA, hB, wB, hC, wC;

  SciGPUgemm::GEMMcleaver testCleave( memAllowed );
  

  // Initialise PRNG
  srand( randSeed );

  
  // Get the matrix sizes
  SetupMatrices( transA, transB, m, n, k, lda, ldb, ldc,
		 hA, wA, hB, wB, hC, wC );
  
  A = new T[lda*wA];
  B = new T[ldb*wB];
  C = new T[ldc*wC];

  resBLAS = new T[ldc*wC];
  resCleaver = new T[ldc*wC];

   // Fill with data
  RandomMatrix( A, lda, hA, wA, min, max );
  RandomMatrix( B, ldb, hB, wB, min, max );
  RandomMatrix( C, ldc, hC, wC, min, max );

  memcpy( resBLAS, C, ldc*wC*sizeof(T) );
  memcpy( resCleaver, C, ldc*wC*sizeof(T) );


  // Run the test
  testCleave.autogemm( transA, transB,
		       m, n, k,
		       alpha,
		       A, lda, B, ldb,
		       beta,
		       resCleaver, ldc);
  
  gemm_gold( A, B, resBLAS,
	     alpha, beta,
	     transA, transB,
	     m, n, k, lda, ldb, ldc );
  
  // Compare the results
 
  double err, ref;
  err = ref = 0;
  for( unsigned int j=0; j<wC; j++ ) {
    for( unsigned int i=0; i<hC; i++ ) {
      T currDiff = resBLAS[i+(j*ldc)] - resCleaver[i+(j*ldc)];
      
      err += mynorm( currDiff );
      ref += mynorm(resBLAS[i+(j*ldc)]);                    ;

    }
  }

#if 0
  PrintMatrix( resBLAS, ldc, hC, wC );
  std::cout << endl;
  PrintMatrix( resCleaver, ldc, hC, wC );
  std::cout << endl;
#endif


  std::cout << "Err L2 = " << sqrt( err / ref ) << std::endl;
  std::cout << "Test complete" << std::endl;

  // Release memory
  delete[] A;
  delete[] B;
  delete[] C;
  delete[] resBLAS;
  delete[] resCleaver;
}


#endif
