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


#include "testrig.hpp"

#include <string>
#include <vector>
using namespace std;



// =============================================================================





template<> void RunCleaver<double>( const char transA, const char transB,
				    const int m, const int n, const int k,
				    const double alpha,
				    const double *A, const int lda,
				    const double *B, const int ldb,
				    const double beta,
				    double *C, const int ldc,
				    SciGPUgemm::GEMMcleaver &theCleaver ) {
  // Call through
  theCleaver.dgemm( transA, transB,
		    m, n, k,
		    alpha,
		    A, lda, B, ldb,
		    beta,
		    C, ldc );
}




template<> void RunCleaver<float>( const char transA, const char transB,
				   const int m, const int n, const int k,
				   const float alpha,
				   const float *A, const int lda,
				   const float *B, const int ldb,
				   const float beta,
				   float *C, const int ldc,
				   SciGPUgemm::GEMMcleaver &theCleaver ) {
  // Call through
  theCleaver.sgemm( transA, transB,
		    m, n, k,
		    alpha,
		    A, lda, B, ldb,
		    beta,
		    C, ldc );
}





// =============================================================================







void WriteHeader( ostream &os ) {
  // Writes out the column headings
  // Needs to match up with RunOneTest

  vector<string> headings;

  headings.push_back( "matrixSize" );
  headings.push_back( "t_CPU" );
  headings.push_back( "t_GPU" );
  
  // Write it out
  for( unsigned int i=0; i<headings.size(); i++ ) {
    os << "# " << setw(2) << i+1 << ": " << headings.at(i) << endl;
  }
 
}
