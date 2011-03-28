// Runs a simple test of mgemm

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
#include <string>
#include <vector>
#include <iostream>
using namespace std;


#include "chronometer.hpp"


#include "matrixutils.hpp"
#include "matrixcompare.hpp"
#include "gemm_gold.hpp"
#include "mgemm.hpp"
#include "gemmcleaver.hpp"


// ==============================================================================

template<typename T> void WriteStream( ostream &os, T data ) {
  // I can't work out how to set the field width of a stream for all writes, so...
  os << setw(15) << data;
}




// ==============================================================================

void RunOneTest( const unsigned int randSeed,
		 const size_t memAllowed,
		 const char transA, const char transB,
		 const int matrixSize,
		 const double min, const double max,
		 const double minSalt, const double maxSalt,
		 const double fSalt,
		 const double alpha, const double beta,
		 const double cutOff,
		 const unsigned int nRepeats,
		 ostream &os ) {
  // Routine to run a single comparision of MGEMM
  double *A, *B, *C;
  double *resBLAS, *resMGEMM, *resCleaverD, *resCleaverS;
  
  int m, n, k, lda, ldb, ldc;
  int hA, wA, hB, wB, hC, wC;
  
  MatrixCompare compareCleaverS, compareCleaverD, compareMGEMM;

  SciGPUgemm::MGEMM testMGEMM( memAllowed );
  SciGPUgemm::GEMMcleaver testCleaver( memAllowed );

  const bool printMatrices = false;

  Chronometer t_cpu, t_mgemm, t_cleaverD, t_cleaverS;
  Chronometer t_random, t_salt;

  // Initialise PRNG
  srand( randSeed );

  
  // Set the matrix sizes
  m = n = k = matrixSize;
  lda = wA = hA = ldb = wB = hB = ldc = wC = hC = matrixSize;

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
  SaltMatrix( A, lda, hA, wA, minSalt, maxSalt, static_cast<int>(floor(fSalt*hA*wA)) );
  SaltMatrix( B, ldb, hB, wB, minSalt, maxSalt, static_cast<int>(floor(fSalt*hB*wB)) );
  t_salt.Stop();
  cout << __FUNCTION__ << ": Matrices salted with " << static_cast<int>(floor(fSalt*hA*wA))
       << " values in " << t_salt.GetTime() << " ms" << endl;

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
    
    t_mgemm.Start();
    testMGEMM.mgemm( transA, transB,
		     m, n, k,
		     alpha,
		     A, lda, B, ldb,
		     beta,
		     resMGEMM, ldc,
		     cutOff );
    t_mgemm.Stop();
    
    // Note that 'CleaverS' uses MGEMM with the cutOff set to be larger than
    // anything in the matrix
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
    
    t_cpu.Start();
    gemm_gold( A, B, resBLAS,
	       alpha, beta,
	       transA, transB,
	       m, n, k, lda, ldb, ldc );
    t_cpu.Stop();
  }
  

  // Make the comparisons
  compareCleaverS.CompareArrays( resBLAS, resCleaverS, ldc, hC, wC );
  compareCleaverD.CompareArrays( resBLAS, resCleaverD, ldc, hC, wC );
  compareMGEMM.CompareArrays( resBLAS, resMGEMM, ldc, hC, wC );

  // Optional print for debugging
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


  // Output the results
  WriteStream( os, matrixSize );
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

  headings.push_back( "matrixSize" );
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


