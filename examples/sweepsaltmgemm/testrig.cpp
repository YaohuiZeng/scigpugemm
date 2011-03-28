// Runs a simple test of MGEMM

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
using namespace std;

#include "cudacheck.hpp"

#include "chronometer.hpp"

#include "matrixutils.hpp"
#include "matrixcompare.hpp"
#include "gemm_gold.hpp"
#include "mgemm.hpp"
#include "gemmcleaver.hpp"


// ==============================================================================

void SetupMatrices( const char transA, const char transB,
		    const int m, const int n, const int k,
		    const int lda, const int ldb, const int ldc,
		    int &hA, int &wA, int &hB, int &wB,
		    int &hC, int &wC );




// ==============================================================================

template<typename T> void WriteStream( ostream &os, T data ) {
  // I can't work out how to set the field width of a stream for all writes, so...
  os << setw(15) << data;
}




// ==============================================================================

void RunOneTest( const unsigned int randSeed,
		 const size_t memAllowed,
		 const char transA, const char transB,
		 const int m, const int n, const int k,
		 const int lda, const int ldb, const int ldc,
		 const double min, const double max,
		 const double minSalt, const double maxSalt,
		 const unsigned int nSalt,
		 const double alpha, const double beta,
		 const double cutOff,
		 const unsigned int nRepeats,
		 ostream &os ) {
  // Routine to run a single comparision of MGEMM
  double *A, *B, *C;
  double *resBLAS, *resMGEMM, *resCleaverD, *resCleaverS;

  int hA, wA, hB, wB, hC, wC;
  
  MatrixCompare compareCleaverS, compareCleaverD, compareMGEMM;

  SciGPUgemm::MGEMM testMGEMM( memAllowed );
  SciGPUgemm::GEMMcleaver testCleaver( memAllowed );

  const bool printMatrices = false;

  Chronometer t_cpu, t_mgemm, t_cleaverD, t_cleaverS;
  Chronometer t_random, t_salt;
  // Initialise PRNG
  srand( randSeed );

  
  // Get the matrix sizes
  SetupMatrices( transA, transB, m, n, k, lda, ldb, ldc,
		 hA, wA, hB, wB, hC, wC );


  // Allocate memory
  A = new double[lda*wA];
  B = new double[ldb*wB];
  C = new double[ldc*wC];

  resBLAS = new double[ldc*wC];
  resMGEMM = new double[ldc*wC];
  resCleaverD = new double[ldc*wC];
  resCleaverS = new double[ldc*wC];

  // Fill with data
  t_random.Start();
  RandomMatrix( A, lda, hA, wA, min, max );
  RandomMatrix( B, ldb, hB, wB, min, max );
  RandomMatrix( C, ldc, hC, wC, min, max );
  t_random.Stop();
  cout << __FUNCTION__ << ": Matrices randomised in " << t_random.GetTime() << " ms" << endl;

  t_salt.Start();
  SaltMatrix( A, lda, hA, wA, minSalt, maxSalt, nSalt );
  SaltMatrix( B, ldb, hB, wB, minSalt, maxSalt, nSalt );
  t_salt.Stop();
  cout << __FUNCTION__ << ": Matrices salted in " << t_salt.GetTime() << " ms" << endl;

 // Average over designated number of runs
  for( unsigned int iRepeat=0; iRepeat<nRepeats; iRepeat++ ) {
    // Set up result matrices
    memcpy( resBLAS, C, ldc*wC*sizeof(double) );
    memcpy( resMGEMM, C, ldc*wC*sizeof(double) );
    memcpy( resCleaverD, C, ldc*wC*sizeof(double) );
    memcpy( resCleaverS, C, ldc*wC*sizeof(double) );



    // Run the tests
    t_cleaverD.Start();
    testCleaver.dgemm( transA, transB,
		       m, n, k,
		       alpha,
		       A, lda, B, ldb,
		       beta,
		       resCleaverD, ldc );
    t_cleaverD.Stop();
#ifdef _DEBUG
    cout << "Done cleaverD" << endl;
#endif
    
    t_mgemm.Start();
    testMGEMM.mgemm( transA, transB,
		     m, n, k,
		     alpha,
		     A, lda, B, ldb,
		     beta,
		     resMGEMM, ldc,
		     cutOff );
    t_mgemm.Stop();
#ifdef _DEBUG
    cout << "Done MGEMM" << endl;
#endif
    
    // Note that 'CleaverS' uses MGEMM with the cutOff set to be very large
    // This will cause a fall-through to SGEMM on the GPU
    t_cleaverS.Start();
    testMGEMM.mgemm( transA, transB,
		     m, n, k,
		     alpha,
		     A, lda, B, ldb,
		     beta,
		     resCleaverS, ldc,
		     1.1*( maxSalt > max ? maxSalt : max ) );
    t_cleaverS.Stop();
#ifdef _DEBUG
    cout << "Done cleaverS" << endl;
#endif
    
    t_cpu.Start();
    gemm_gold( A, B, resBLAS,
	       alpha, beta,
	       transA, transB,
	       m, n, k, lda, ldb, ldc );
    t_cpu.Stop();
#ifdef _DEBUG
    cout << "Done CPU" << endl;
#endif
  }
    
  // Compare the results
  compareCleaverS.CompareArrays( resBLAS, resCleaverS, ldc, hC, wC );
  compareCleaverD.CompareArrays( resBLAS, resCleaverD, ldc, hC, wC );
  compareMGEMM.CompareArrays( resBLAS, resMGEMM, ldc, hC, wC );

  if( printMatrices ) {
    // Print matrix A
    cout << endl;
    cout << "Matrix A:" << endl;
    PrintMatrix( A, lda, hA, wA );
    cout << endl;
    
    // And B
    cout << "Matrix B:" << endl;
    PrintMatrix( B, ldb, hB, wB );
    cout << endl;
    
    cout << "Matrix resMGEMM" << endl;
    PrintMatrix( resMGEMM, ldc, hC, wC );
    cout << endl;
    
    cout << "Matrix resBLAS" << endl;
    PrintMatrix( resBLAS, ldc, hC, wC );
    cout << endl;
  }

  WriteStream( os, nSalt );
  WriteStream( os, t_cpu.GetAverageTime() );
  WriteStream( os, t_mgemm.GetAverageTime() );
  WriteStream( os, t_cleaverS.GetAverageTime() );
  WriteStream( os, t_cleaverD.GetAverageTime() );

  os << compareMGEMM;
  os << compareCleaverS;
  os << compareCleaverD;

  os << endl;

  // Release memory
  delete[] A;
  delete[] B;
  delete[] C;
  delete[] resBLAS;
  delete[] resMGEMM;
  delete[] resCleaverS;
  delete[] resCleaverD;


}



// =============================================================================

void WriteHeader( ostream &os ) {
  // Writes out the column headings
  // Needs to match up with the above

  vector<string> headings;
  vector<string> matrixCompareHeaders;
  vector<string> comparisons;

  // Compile the list
  // This is tedious, but worthwhile

  // Things which we will compare
  comparisons.push_back( "MGEMM" );
  comparisons.push_back( "CleaverS" );
  comparisons.push_back( "CleaverD" );

  // List of headers from MatrixCompare
  MatrixCompare::GetHeaders( matrixCompareHeaders );

  headings.push_back( "nSalt" );
  headings.push_back( "t_CPU" );
  headings.push_back( "t_MGEMM" );
  headings.push_back( "t_cleaverS" );
  headings.push_back( "t_cleaverD" );
  
  // Create the comparison headers
  for( unsigned int i=0; i<comparisons.size(); i++ ) {
    for( unsigned int j=0; j<matrixCompareHeaders.size(); j++ ) {
      string myString  = comparisons.at(i) + " " + matrixCompareHeaders.at(j);
      headings.push_back( myString );
    }
    
  }

  // Write it out
  for( unsigned int i=0; i<headings.size(); i++ ) {
    os << "# " << setw(2) << i+1 << ": " << headings.at(i) << endl;
  }
 
}



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
