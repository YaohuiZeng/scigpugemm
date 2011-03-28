// Test Program for the Cleaver

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
#include <string>
using namespace std;

#include <cuda_runtime.h>

#include "cudacheck.hpp"


#include "testrig.hpp"



// ================================================

void InitDevice( const int nDevice, ofstream &os ) {

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
  os << "# Device name: " << prop.name <<endl;
  
  
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

  string outFilename;
  ofstream outputFile;

  unsigned int randomSeed, nRepeats;
  int minSize, maxSize, stepSize;
  char transA, transB;
  double min, max;
  double alpha, beta;
  int iDevice;
  size_t memAllowed;

  char dataType;
  
  // -----------------------------------------
  cout << "Cleaver Test" << endl;
  cout << "============" << endl << endl;
  
  cout << "Enter output filename" << endl;
  cin >> outFilename;
  
  outputFile.open( outFilename.c_str() );
  if( !outputFile.good() ) {
    cerr << "Failed to open output file " << outFilename << endl;
    return( EXIT_FAILURE );
  }
  
  cout << "Enter device number" << endl;
  cin >> iDevice;
  
  InitDevice( iDevice, outputFile );
  
  cout << "Enter allowed memory consumption" << endl;
  cin >> memAllowed;
  
  cout << "Enter random number seed" << endl;
  cin >> randomSeed;

  // -----------------------------------------
  // Get the matrix sizes
  // Very basic checking performed

  cout << "Enter transA, transB" << endl;
  cin >> transA >> transB;

  cout << "Enter minSize, maxSize, stepSize" << endl;
  cin >> minSize >> maxSize >> stepSize;
  if( (minSize<1) || (maxSize<minSize) ) {
    cerr << "Invalid Sizes" << endl;
    return( EXIT_FAILURE );
  }
  if( stepSize < 1 ) {
    cerr << "Invalid stepSize" << endl;
    return( EXIT_FAILURE );
  }
  
  cout << "Enter alpha, beta" << endl;
  cin >> alpha >> beta;

  cout << "Enter min, max values" << endl;
  cin >> min >> max;
  if( max <= min ) {
    cerr << "Invalid range" << endl;
    return( EXIT_FAILURE );
  }

  // ---------------------------------------------
  
  // Get the data type
  cout << "Enter data type" << endl;
  cin >> dataType;
  switch( dataType ) {
  case 'd':
  case 'f':
    break;

  default:
    cerr << "Unrecognised data type" << endl;
    return( EXIT_FAILURE );
  }

  
  // And number of repetitions
  cout << "Enter nRepeats" << endl;
  cin >> nRepeats;
  


  outputFile << "# Memory allowed = " << memAllowed << endl;
  outputFile << "# Random seed = " << randomSeed << endl;
  outputFile << "# transA, transB = " << transA << " " << transB << endl;
  outputFile << "# alpha, beta = " << alpha << " " << beta << endl;
  outputFile << "# min, max = " << min << " " << max << endl;
  outputFile << "# dataType = " << dataType << endl;

  WriteHeader( outputFile );
  
  // ============================================
  // Run a test
  for( int iSize=minSize; iSize<=maxSize; iSize += stepSize ) {

    switch( dataType ) {
    case 'f':
      RunOneTest<float>( randomSeed, memAllowed,
			 transA, transB,
			 iSize,
			 min, max,
			 alpha, beta,
			 nRepeats,
			 outputFile );
      break;

    case 'd':
      RunOneTest<double>( randomSeed, memAllowed,
			  transA, transB,
			  iSize,
			  min, max,
			  alpha, beta,
			  nRepeats,
			  outputFile );
      break;

    default:
      cerr << "Unrecognised data type" << endl;
      return( EXIT_FAILURE );
    }
      
    cout << "Done iSize = " << iSize << endl;
  }

  return( EXIT_SUCCESS );
}
