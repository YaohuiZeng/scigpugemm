// Runs a simple test of the cleaver

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
#include <string>
#include <vector>
#include <iostream>
#include <cstring>
#include <complex>
using namespace std;


/*
#include "matrixutils.hpp"
#include "matrixcompare.hpp"
#include "gemm_gold.hpp"
#include "gemmcleaver.hpp"
*/

// ==============================================================================





// ==============================================================================

#if 0
void RunOneTest( const unsigned int randSeed,
		 const size_t memAllowed,
		 const char transA, const char transB,
		 const int m, const int n, const int k,
		 const int lda, const int ldb, const int ldc,
		 const float min, const float max,
		 const float alpha, const float beta ) {
  // Routine to run a single comparision of GEMMcleaver
  float *A, *B, *C;
  float *resBLAS, *resCleaver;

  int hA, wA, hB, wB, hC, wC;

  SciGPUgemm::GEMMcleaver testCleave( memAllowed );
  MatrixCompare comparator;

  // Initialise PRNG
  srand( randSeed );

  
  // Get the matrix sizes
  SetupMatrices( transA, transB, m, n, k, lda, ldb, ldc,
		 hA, wA, hB, wB, hC, wC );


  // Allocate memory
  A = new float[lda*wA];
  B = new float[ldb*wB];
  C = new float[ldc*wC];

  cout << __FUNCTION__ << ": " << lda*wA << " " << ldb*wB << " " << ldc*wC << endl;

  resBLAS = new float[ldc*wC];
  resCleaver = new float[ldc*wC];

   // Fill with data
  RandomMatrix( A, lda, hA, wA, min, max );
  RandomMatrix( B, ldb, hB, wB, min, max );
  RandomMatrix( C, ldc, hC, wC, min, max );

  memcpy( resBLAS, C, ldc*wC*sizeof(float) );
  memcpy( resCleaver, C, ldc*wC*sizeof(float) );



  // Run the test
  testCleave.sgemm( transA, transB,
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
  comparator.CompareArrays( resBLAS, resCleaver,
			    ldc, hC, wC );

  cout << "Test complete" << endl;
  cout << comparator << endl;

  // Release memory
  delete[] A;
  delete[] B;
  delete[] C;
  delete[] resBLAS;
  delete[] resCleaver;

}
#endif


// =============================================================================

void SetupMatrices( const char transA, const char transB,
		    const int m, const int n, const int k,
		    const int lda, const int ldb, const int ldc,
		    int &hA, int &wA, int &hB, int &wB,
		    int &hC, int &wC ) {
  // Uses the input information to configure the matrices
  // Recall that GEMM does
  // C <- alpha * op(A) * op(B) + beta*C
  // where op(X) = X or X^T dependent on the flag
  // op(A) is an m*k matrix
  // op(B) is a k*n matrix
  // C is an m*n matrix
  // ld(a|b|c) are the leading dimensions of A, B and C
  // These are not redundant with m, n and k, since the
  // input matrices may not be fully dense (which allows in-place
  // slicing of matrices)
  // However, the leading dimension of a matrix must be at least
  // as great as its height (recall that we are using Fortran
  // column-major ordering)

  // Start with C, since it's easy
  hC = m;
  wC = n;
  if( ldc < hC ) {
    cerr << __FUNCTION__ << ": Error ldc too small" << endl;
    exit( EXIT_FAILURE );
  }

  // Next up is matrix A
  // Use fall-through to accept the same options as GEMM
  switch( transA ) {
  case 'N':
  case 'n':
    // We aren't transposing A
    hA = m;
    wA = k;
    break;

  case 'T':
  case 't':
  case 'C':
  case 'c':
    // Transposing
    hA = k;
    wA = m;
    break;

  default:
    cerr << __FUNCTION__ << ": Unrecognised transA" << endl;
    exit( EXIT_FAILURE );
  }

  // Verify that the sizes make sense
  if( lda < hA ) {
    cerr << __FUNCTION__ << ": Error lda too small" << endl;
    exit( EXIT_FAILURE );
  }

  // Matrix B is similar
  switch( transB ) {
  case 'N':
  case 'n':
    // We aren't transposing B
    hB = k;
    wB = n;
    break;

  case 'T':
  case 't':
  case 'C':
  case 'c':
    // Transposing
    hB = n;
    wB = k;
    break;

  default:
    cerr << __FUNCTION__ << ": Unrecognised transB" << endl;
    exit( EXIT_FAILURE );
  }

  // Verify that the sizes make sense
  if( ldb < hB ) {
    cerr << __FUNCTION__ << ": Error ldb too small" << endl;
    exit( EXIT_FAILURE );
  }
    

}
