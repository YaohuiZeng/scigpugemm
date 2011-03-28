// Test Program for GEMMcleaver

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
#include <iostream>
#include <fstream>
#include <complex>
using namespace std;

#include <cuda_runtime.h>

#include "cudacheck.hpp"


#include "testrig.hpp"


// ================================================

void InitDevice( const int nDevice ) {
  
  int deviceCount;
  float *d_temp;
  cudaDeviceProp prop;

  // Verify that there are CUDA devices available
  CUDA_SAFE_CALL( cudaGetDeviceCount( &deviceCount ) );
  if( deviceCount == 0 ) {
    cerr << "No device supporting CUDA available!" << endl;
    exit( EXIT_FAILURE );
  }

  // Check desired device is in range
  if( nDevice > deviceCount-1 ) {
    cerr << "Invalid CUDA device" << endl;
    exit( EXIT_FAILURE );
  }
  
  CUDA_SAFE_CALL( cudaGetDeviceProperties( &prop, nDevice ) );
  cout << "Device name: " << prop.name <<endl;
  
  
  // Set the device
  CUDA_SAFE_CALL( cudaSetDevice( nDevice ) );

  // Do a quick malloc/free to make sure it's there
  CUDA_SAFE_CALL( cudaMalloc( (void**)&d_temp, 1 ) );
  if( d_temp == NULL ) {
    cerr << "Failed to initialse GPU" << endl;
    exit( EXIT_FAILURE );
  }
  CUDA_SAFE_CALL( cudaFree( d_temp ) );
}


// ================================================

int main( void ) {

  unsigned int randomSeed;
  int m, n, k;
  int lda, ldb, ldc;
  char transA, transB;
  float min, max;
  float alpha, beta;
  int iDevice;
  size_t memAllowed;
  char type;

  // -----------------------------------------
  cout << "GEMMcleaver Test" << endl;
  cout << "================" << endl << endl;

  cout << "Enter device number" << endl;
  cin >> iDevice;
  
  InitDevice( iDevice );

  cout << "Enter allowed memory consumption" << endl;
  cin >> memAllowed;

  cout << "Enter random number seed" << endl;
  cin >> randomSeed;

  
  // -----------------------------------------
  // Get the matrix sizes
  // Very basic checking performed

  cout << "Enter transA, transB" << endl;
  cin >> transA >> transB;

  cout << "Enter m, n, k" << endl;
  cin >> m >> n >> k;
  if( (m<1) || (n<1) || (k<1) ) {
    cerr << "Invalid Sizes" << endl;
    return( EXIT_FAILURE );
  }

  cout << "Enter lda, ldb, ldc" << endl;
  cin >> lda >> ldb >> ldc;
  if( (lda<1) || (ldb<1) || (ldc<1) ) {
    cerr << "Invalid Sizes" << endl;
    return( EXIT_FAILURE );
  }

  
  cout << "Enter alpha, beta" << endl;
  cin >> alpha >> beta;


  // ------------------------------------------
  // Get information about values to fill in
  cout << "Enter min, max values" << endl;
  cin >> min >> max;
  if( max <= min ) {
    cerr << "Invalid range" << endl;
    return( EXIT_FAILURE );
  }

  cout << "Enter type" << endl;
  cin >> type;
  

  // ============================================
  // Run a test
  switch( type ) {
  case 'f':
    RunOneTest<float>( randomSeed, memAllowed,
		       transA, transB,
		       m, n, k, lda, ldb, ldc,
		       min, max,
		       alpha, beta );
    break;

  case 'd':
    RunOneTest<double>( randomSeed, memAllowed,
			transA, transB,
			m, n, k, lda, ldb, ldc,
			min, max,
			alpha, beta );
    break;

  case 'c':
    RunOneTest< complex<float> >( randomSeed, memAllowed,
				  transA, transB,
				  m, n, k, lda, ldb, ldc,
				  complex<float>( min, min ), complex<float>( max, max ),
				  complex<float>( alpha, alpha ), complex<float>( beta, beta ) );
    break;

  case 'z':
    RunOneTest< complex<double> >( randomSeed, memAllowed,
				  transA, transB,
				  m, n, k, lda, ldb, ldc,
				  complex<double>( min, min ), complex<double>( max, max ),
				  complex<double>( alpha, alpha ), complex<double>( beta, beta ) );
    break;

  default:
    cerr << "Unrecognised type!" << endl;
    return( EXIT_FAILURE );

  }

  

  return( EXIT_SUCCESS );
}
