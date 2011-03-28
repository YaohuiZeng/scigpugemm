// Runs a simple test of the mgemm

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

#include "chronometer.hpp"


#include "matrixutils.hpp"
#include "matrixcompare.hpp"
#include "gemm_gold.hpp"
#include "mgemm.hpp"
#include "gemmcleaver.hpp"

#include "blasmatrix.hpp"

// ==============================================================================

template<typename T> void WriteStream( ostream &os, T data ) {
  // I can't work out how to set the field width of a stream for all writes, so...
  os << setw(15) << data;
}




// ==============================================================================

void RunOneTest( const size_t memAllowed,
		 const char transA, const char transB,
		 const BlasMatrix<double> &A, const BlasMatrix<double> &B,
		 const unsigned int nRepeats,
		 const double cutOff,
		 ostream &os ) {
  // Does a comparison of MGEMM

  const double alpha = 1;
  const double beta = 0;

  int m, n, k;
  double *resBLAS, *resMGEMM, *resCleaverD, *resCleaverS, *C;

  Chronometer t_cpu, t_mgemm, t_cleaverD, t_cleaverS;

  MatrixCompare compareCleaverS, compareCleaverD, compareMGEMM;

  SciGPUgemm::MGEMM testMGEMM( memAllowed );
  SciGPUgemm::GEMMcleaver testCleaver( memAllowed );

  cout << __FUNCTION__ << ": Testing cutOff = " << cutOff << endl;

  // Get the sizes
  m = n = k = -1;

  switch( transA ) {
  case 'n':
  case 'N':
    m = A.nRows;
    k = A.nCols;
    break;

  case 't':
  case 'T':
  case 'C':
  case 'c':
    m = A.nCols;
    k = A.nRows;
    break;
	
  default:
    cerr << __PRETTY_FUNCTION__ << ": Unrecognised transA" << endl;
    exit( EXIT_FAILURE );
  }

  switch( transB ) {
  case 'n':
  case 'N':
    n = B.nCols;
    if( static_cast<unsigned int>(k) != B.nRows ) {
      cerr << __FUNCTION__ << ": Matrices not conformable" << endl;
      exit( EXIT_FAILURE );
    }
    break;
    
  case 't':
  case 'T':
  case 'C':
  case 'c':
    n = B.nRows;
    if( static_cast<unsigned int>(k) != B.nCols ) {
      cerr << __FUNCTION__ << ": Matrices not conformable" << endl;
      exit( EXIT_FAILURE );
    }
    break;
    
  default:
    cerr << __PRETTY_FUNCTION__ << ": Unrecognised transB" << endl;
    exit( EXIT_FAILURE );
  }

  // Allocate the matrices
  resBLAS = new double[m*n];
  resMGEMM = new double[m*n];
  resCleaverD = new double[m*n];
  resCleaverS = new double[m*n];
  C = new double[m*n];

  RandomMatrix( C, m, m, n, -1.0, 1.0 );

  // Average over designated number of runs
  for( unsigned int iRepeat=0; iRepeat<nRepeats; iRepeat++ ) {
    // Set up result matrices
    memcpy( resBLAS, C, m*n*sizeof(double) );
    memcpy( resMGEMM, C, m*n*sizeof(double) );
    memcpy( resCleaverD, C, m*n*sizeof(double) );
    memcpy( resCleaverS, C, m*n*sizeof(double) );

    
    t_cleaverD.Start();
    testCleaver.dgemm( transA, transB,
		       m, n, k,
		       alpha,
		       A.matrix, A.nRows, B.matrix, B.nRows,
		       beta,
		       resCleaverD, m );
    t_cleaverD.Stop();

    // Note that 'CleaverS' uses MGEMM with the cutOff set to be larger than
    // anything in the matrix
    // This will cause a fall-through to SGEMM on the GPU
    t_cleaverS.Start();
    testMGEMM.mgemm( transA, transB,
		     m, n, k,
		     alpha, 
		     A.matrix, A.nRows, B.matrix, B.nRows,
		     beta,
		     resCleaverS, m,
		     1e128 );
    t_cleaverS.Stop();

    // Do 'real' MGEMM second, so the nAlarge, nBlarge values are correct
    t_mgemm.Start();
    testMGEMM.mgemm( transA, transB,
		     m, n, k,
		     alpha,
		     A.matrix, A.nRows, B.matrix, B.nRows,
		     beta,
		     resMGEMM, m,
		     cutOff );
    t_mgemm.Stop();
    
    t_cpu.Start();
    gemm_gold( A.matrix, B.matrix, resBLAS,
	       alpha, beta,
	       transA, transB,
	       m, n, k, A.nRows, B.nRows, m );
    t_cpu.Stop();

  }

  // Make the comparisons
  compareCleaverS.CompareArrays( resBLAS, resCleaverS, m, m, n );
  compareCleaverD.CompareArrays( resBLAS, resCleaverD, m, m, n );
  compareMGEMM.CompareArrays( resBLAS, resMGEMM, m, m, n );


  // Output the results
  WriteStream( os, cutOff );
  WriteStream( os, t_cpu.GetAverageTime() );
  WriteStream( os, t_mgemm.GetAverageTime() );
  WriteStream( os, t_cleaverS.GetAverageTime() );
  WriteStream( os, t_cleaverD.GetAverageTime() );
  WriteStream( os, testMGEMM.nAlarge );
  WriteStream( os, testMGEMM.nBlarge );

  os << compareMGEMM;
  os << compareCleaverS;
  os << compareCleaverD;
  os << endl;

  // Release memory
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

  headings.push_back( "cutOff" );
  headings.push_back( "t_CPU" );
  headings.push_back( "t_MGEMM" );
  headings.push_back( "t_cleaverS" );
  headings.push_back( "t_cleaverD" );
  headings.push_back( "nAlarge" );
  headings.push_back( "nBlarge" );
  
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
